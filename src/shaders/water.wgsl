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
  out.rapids = smoothstep(0.04, 0.35, abs(drop));
  // Channel-center depth drives the Manning speed; the fragment shader slows it at the banks.
  out.speed = riverSpeed(px, 1.7 * mix(1.0, 0.45, smoothstep(0.0, 1.0, out.rapids)));
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
  if (abs(frag.across) > 1.6) {
    discard;
  }
  let s = sky();
  let p = frag.world;
  let t = G.time;
  let flow = normalize(frag.flow);
  let side = vec2f(-flow.y, flow.x);
  // Faster in the middle of the channel, lazy at the banks.
  // Velocity profile across the channel: fastest mid-stream, near still at the edges.
  let speed = frag.speed * mix(1.0, 0.25, smoothstep(0.2, 1.0, abs(frag.across)));
  let travel = t * speed;
  let uv = vec2f(dot(p.xz, flow), dot(p.xz, side));
  let camDist = distance(G.camPos, p);
  let grazing = 1.0 - abs(normalize(G.camPos - p).y);
  let detail = (1.0 - smoothstep(4.0, 30.0, camDist)) * (1.0 - grazing * 0.6);
  let e = 0.08;
  let h0 = waveHeight(uv, travel, t, detail);
  let hx = waveHeight(uv + vec2f(e, 0.0), travel, t, detail) - h0;
  let hy = waveHeight(uv + vec2f(0.0, e), travel, t, detail) - h0;
  let amp = 0.075 * mix(1.0, 0.45, smoothstep(10.0, 60.0, camDist)) * (1.0 + frag.rapids * 3.0);
  let gradLocal = vec2f(hx, hy) / e * amp;
  let grad = flow * gradLocal.x + side * gradLocal.y;
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
  let bedUv = p.xz * 0.45 + n.xz * depth * 0.25;
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
  let bubbles = smoothstep(0.55, 0.85, simplex2d((uv - vec2f(travel, 0.0)) * 4.0) * 0.5 + 0.5);
  let streak = smoothstep(0.93, 0.985, lineN) * bubbles;
  let churn = smoothstep(0.3, 0.75, simplex2d((uv - vec2f(travel * 1.3, 0.0)) * vec2f(0.6, 1.8)) * 0.5 + 0.5 + frag.rapids * 0.35);
  let foam = clamp(shore * lace * 0.6 + streak * 0.7 + churn * frag.rapids * 1.4, 0.0, 1.0) * mix(0.5, 0.9, frag.rapids);
  col = mix(col, (G.sunColor * 0.75 + s.zenith * 0.5) * 0.9, foam);

  col = applyFog(col, p, G.camPos, G.fogDensity, s);
  return vec4f(col, 1.0);
}
