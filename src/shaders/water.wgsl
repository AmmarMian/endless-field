// River water. A camera-following ribbon laid along the procedural river; the surface is
// opaque but shades what lies beneath it analytically (the riverbed height is known), so
// we get depth-based absorption, caustics and shore foam without a refraction pass.
import { Globals } from "./lib/globals.wgsl";
import { riverCenter, riverHalfWidth, riverInfo, riverSpeed, riverWater, terrainHeight } from "./lib/terrain.wgsl";
import { SkyParams, applyFog, skyColor } from "./lib/atmosphere.wgsl";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";

struct WaterParams {
  center: vec2f,
  extent: f32,
  pad: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> W: WaterParams;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var pebbles: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) flow: vec2f,
  @location(2) across: f32,
  @location(3) rapids: f32,
  @location(4) speed: f32,
  // Unsigned distance from the centerline in half-widths: continuous everywhere, unlike
  // `across`, whose sign flips where the closest point jumps between meander branches.
  @location(5) dist: f32,
  // White water boiling in the plunge pool just below a drop.
  @location(6) plunge: f32,
}

fn sky() -> SkyParams {
  return SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
}

// A camera-centered grid; each vertex snaps its height to the nearest point on the river.
// Fragments outside the channel are discarded; the banks hide the remaining edge.
@vertex
fn vs_main(@location(0) g: vec2f) -> VOut {
  let xz = W.center + g * W.extent;
  let r = riverInfo(xz);
  let px = r.w;
  let slope = (riverCenter(px + 0.75) - riverCenter(px - 0.75)) / 1.5;
  let tangent = normalize(vec2f(1.0, slope));
  let downhill = select(-1.0, 1.0, riverInfoWater(px + 4.0) < riverInfoWater(px - 4.0));
  // Signed distance across the channel in half-widths (negative on the left bank).
  let toPoint = xz - vec2f(px, riverCenter(px));
  let sideSign = select(-1.0, 1.0, dot(toPoint, vec2f(-tangent.y, tangent.x)) >= 0.0);
  var out: VOut;
  out.world = vec3f(xz.x, r.y, xz.y);
  out.pos = G.viewProj * vec4f(out.world, 1.0);
  out.flow = tangent * downhill;
  out.across = sideSign * min(r.x / r.z, 4.0);
  // Steep stretches (cascades between the pools of the mountain stream) become white water.
  let drop = (riverWater(px - 3.0) - riverWater(px + 3.0)) / 6.0;
  out.rapids = smoothstep(0.12, 0.45, abs(drop));
  // Upstream of this point (within ~10 m) the water fell: a plunge pool.
  var fell = 0.0;
  for (var k = 1; k <= 3; k = k + 1) {
    let up = px - downhill * f32(k) * 3.5;
    let du = (riverWater(up - 3.0) - riverWater(up + 3.0)) / 6.0;
    fell = max(fell, smoothstep(0.12, 0.45, abs(du)) * (1.0 - f32(k - 1) * 0.25));
  }
  out.plunge = fell * (1.0 - out.rapids);
  // Channel-center depth drives the Manning speed; the fragment shader slows it at the banks.
  out.speed = riverSpeed(px, 1.7 * mix(1.0, 0.45, smoothstep(0.0, 1.0, out.rapids)));
  out.dist = min(r.x / r.z, 4.0);
  return out;
}

fn riverInfoWater(x: f32) -> f32 {
  return riverInfo(vec2f(x, riverCenter(x))).y;
}

// `p` is in flow space (x downstream, meters); `s` is how far the water has travelled.
// Layers advect at slightly different speeds so the pattern evolves instead of sliding.
fn waveHeight(p: vec2f, s: f32, t: f32, detail: f32) -> f32 {
  return simplex2d((p - vec2f(s, 0.0)) * vec2f(0.3, 0.45)) * 0.5
    + simplex2d((p - vec2f(s * 1.2, 0.0)) * 1.1 + vec2f(0.0, 7.0)) * 0.22
    + (simplex2d((p - vec2f(s * 1.35, 0.0)) * 3.1 + vec2f(t * 0.3, -3.0)) * 0.12
    + simplex2d((p - vec2f(s * 1.5, 0.0)) * 7.0 + vec2f(-t * 0.6, t * 0.9)) * 0.05) * detail;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  if (frag.dist > 1.6 || abs(frag.across) > 1.6) {
    discard;
  }
  let s = sky();
  let p = frag.world;
  let t = G.time;
  let flow = normalize(frag.flow);
  let side = vec2f(-flow.y, flow.x);
  // Faster in the middle of the channel, lazy at the banks.
  // Velocity profile across the channel: fastest mid-stream, still moving at the banks.
  let speed = frag.speed * mix(1.0, 0.55, smoothstep(0.3, 1.1, abs(frag.across)));
  let travel = t * speed;
  let uv = vec2f(dot(p.xz, flow), dot(p.xz, side));
  let camDist = distance(G.camPos, p);
  let grazing = 1.0 - abs(normalize(G.camPos - p).y);
  let detail = (1.0 - smoothstep(4.0, 30.0, camDist)) * (1.0 - grazing * 0.6);
  // Two-phase flow mapping: two copies of the ripple field are advected by the local speed
  // and reset half a cycle apart, cross-faded so the surface evolves as it flows instead of
  // sliding as a rigid band (and shear across the channel never stretches it).
  let period = 1.8;
  let ph0 = fract(t / period);
  let ph1 = fract(t / period + 0.5);
  let w0 = 1.0 - abs(ph0 * 2.0 - 1.0);
  let d0 = ph0 * period * speed;
  let d1 = ph1 * period * speed;
  let j0 = vec2f(0.0, 0.0);
  let j1 = vec2f(17.3, 9.1);
  let e = 0.08;
  let a0 = waveHeight(uv + j0, d0, t, detail);
  let a1 = waveHeight(uv + j1, d1, t, detail);
  let h0 = mix(a1, a0, w0);
  let hx = mix(waveHeight(uv + j1 + vec2f(e, 0.0), d1, t, detail), waveHeight(uv + j0 + vec2f(e, 0.0), d0, t, detail), w0) - h0;
  let hy = mix(waveHeight(uv + j1 + vec2f(0.0, e), d1, t, detail), waveHeight(uv + j0 + vec2f(0.0, e), d0, t, detail), w0) - h0;
  // Crossfading halves the contrast at mid-blend; compensate so ripples stay even.
  let blendGain = 1.0 / sqrt(w0 * w0 + (1.0 - w0) * (1.0 - w0));
  // Rain beats the wind ripples flat; its own rings take over.
  let amp = 0.075 * mix(1.0, 0.45, smoothstep(10.0, 60.0, camDist)) * (1.0 + frag.rapids * 3.0) * blendGain * (1.0 - 0.65 * G.rain);
  let gradLocal = vec2f(hx, hy) / e * amp;
  var grad = flow * gradLocal.x + side * gradLocal.y;
  // Rain on the river: each 0.6 m cell gets a drop at its own moment; the ring spreads and fades.
  if (G.rain > 0.01 && camDist < 60.0) {
    for (var oy = 0; oy < 2; oy = oy + 1) {
      for (var ox = 0; ox < 2; ox = ox + 1) {
        let cell = floor(p.xz / 0.6) + vec2f(f32(ox), f32(oy)) - 0.5;
        let hc = fract(sin(dot(cell, vec2f(127.1, 311.7))) * 43758.5453);
        let hc2 = fract(hc * 91.7);
        let age = fract(t * (0.9 + hc2 * 0.6) + hc);
        let center = (cell + 0.5 + (vec2f(hc, hc2) - 0.5) * 0.5) * 0.6;
        let dv = p.xz - center;
        let d = length(dv);
        let radius = age * 0.35;
        let ring = exp(-pow((d - radius) * 40.0, 2.0)) * (1.0 - age) * step(hc2, G.rain);
        grad = grad + dv / max(d, 1e-3) * ring * 0.6 * (1.0 - smoothstep(20.0, 60.0, camDist));
      }
    }
  }
  let n = normalize(vec3f(-grad.x, 1.0, -grad.y));

  let v = normalize(G.camPos - p);
  let ndv = max(dot(n, v), 0.0);
  var fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);

  // Reflection: the opposite bank (grass and reeds) where the reflected ray is below its
  // skyline, the sky above it.
  let r = reflect(-v, n);
  var refl = skyColor(normalize(vec3f(r.x, max(r.y, 0.01), r.z)), s);
  let hw = riverHalfWidth(p.x);
  let towardSide = dot(normalize(r.xz + vec2f(1e-5)), side);
  let acrossM = frag.across * hw;
  let toBank = select(hw + acrossM, hw - acrossM, towardSide > 0.0) / max(abs(towardSide), 0.15);
  let bankAngle = atan2(2.2, max(toBank, 0.5));
  let rAngle = asin(clamp(r.y, -1.0, 1.0));
  let bankLit = vec3f(0.30, 0.24, 0.08) * (G.sunColor * 0.45 + s.zenith * 0.35);
  let bankColor = mix(bankLit, vec3f(0.01, 0.012, 0.02), G.night * 0.8);
  refl = mix(refl, bankColor, 1.0 - smoothstep(bankAngle - 0.04, bankAngle + 0.04, rAngle));
  let sunSpec = pow(max(dot(r, G.sunDir), 0.0), 1200.0) * 45.0 + pow(max(dot(r, G.sunDir), 0.0), 120.0) * 0.35;
  refl = refl + G.sunColor * sunSpec;

  // Refraction: riverbed pebbles seen through the water column.
  let bedY = terrainHeight(p.xz);
  let depth = max(p.y - bedY, 0.0);
  let path = depth / max(v.y, 0.12);
  // Screen-space footprint of one meter of surface: fine features fade before they alias.
  let footprint = length(fwidth(uv));
  let aa = 1.0 - smoothstep(0.08, 0.4, footprint);
  let bedUv = p.xz * 0.45 + n.xz * min(depth, 0.6) * 0.25 * aa;
  let bed = textureSample(pebbles, samp, bedUv).rgb * 0.8;
  let caust = pow(1.0 - abs(simplex2d(p.xz * 1.1 + flow * t * 0.6) + simplex2d(p.xz * 1.6 - flow * t * 0.45)) * 0.5, 7.0);
  let bedLit = bed * (G.sunColor * max(G.sunDir.y, 0.1) * (0.6 + caust * 2.2) + s.zenith * 0.4);
  let absorb = exp(-path * vec3f(0.42, 0.14, 0.11));
  let scatter = vec3f(0.025, 0.085, 0.075) * (G.sunColor * 0.5 + s.zenith * 0.6);
  let refr = bedLit * absorb + scatter * (1.0 - absorb);

  // Reflections fade out in the last few centimeters so the waterline melts into the beach.
  fresnel = fresnel * smoothstep(0.0, 0.3, depth);
  var col = mix(refr, refl, fresnel);

  let edge = smoothstep(0.35, 1.0, abs(frag.across));
  // Foam, used sparingly: a thin fringe right at the waterline, and the occasional line of
  // bubbles drifting downstream (these sell the current more than anything else).
  let shore = 1.0 - smoothstep(0.0, 0.07, depth);
  let lace = smoothstep(0.45, 0.8, simplex2d((uv - vec2f(travel * 0.6, 0.0)) * vec2f(1.2, 3.5)) * 0.5 + 0.5);
  let lineN = simplex2d((uv - vec2f(travel, 0.0)) * vec2f(0.08, 1.6) + vec2f(0.0, 11.0)) * 0.5 + 0.5;
  let bubbles = smoothstep(0.55, 0.85, simplex2d((uv - vec2f(travel, 0.0)) * 4.0) * 0.5 + 0.5) * aa;
  let streak = smoothstep(0.93, 0.985, lineN) * bubbles;
  // White water. On the drops the water is fully aerated: streaks stretched along the flow,
  // racing downhill. Below each drop it boils up in the plunge pool and breaks apart.
  let fast = travel * 1.8;
  let streakN = simplex2d((uv - vec2f(fast, 0.0)) * vec2f(0.22, 1.7)) * 0.55
    + simplex2d((uv - vec2f(fast * 1.3, 0.0)) * vec2f(0.6, 3.6) + vec2f(5.0, 0.0)) * 0.3 * aa
    + simplex2d((uv - vec2f(fast * 1.6, 0.0)) * vec2f(1.4, 7.0) + vec2f(-3.0, 2.0)) * 0.15 * aa;
  let white = smoothstep(-0.35, 0.35, streakN) * frag.rapids;
  let boilN = simplex2d((uv - vec2f(travel * 0.7, 0.0)) * 0.9 + vec2f(t * 0.35, -t * 0.25)) * 0.6
    + simplex2d((uv - vec2f(travel, 0.0)) * 2.4 - vec2f(t * 0.5, 0.0)) * 0.4 * aa;
  let boil = smoothstep(0.0, 0.6, boilN * 0.5 + 0.5 - (1.0 - frag.plunge) * 0.6) * frag.plunge;
  let foam = clamp(shore * lace * 0.6 + streak * 0.7, 0.0, 1.0) * 0.6;
  let foamCol = (G.sunColor * 0.75 + s.zenith * 0.55) * 0.95;
  col = mix(col, foamCol, foam);
  // Aerated water still shows a little blue-green body between the streaks.
  let rapidBody = mix(foamCol * 0.55, foamCol, smoothstep(-0.2, 0.6, streakN));
  col = mix(col, rapidBody, frag.rapids * 0.85);
  col = mix(col, foamCol * mix(0.8, 1.0, white), clamp(white * 0.6 + boil * 0.75, 0.0, 1.0));

  col = applyFog(col, p, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}
