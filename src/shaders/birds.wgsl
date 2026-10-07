// Meadow birds (model: tools/blender/model_bird.py): painted plumage from vertex colours.
// Wings flap about the shoulder; the outer hand bends further than the arm.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Bird {
  // xyz = position, w = heading (yaw)
  pos: vec4f,
  // x = flap angle, y = wing spread, z = pitch, w = seed (in [0, 1); 1 + k: glowing with k)
  pose: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> birds: array<Bird>;

override SHOULDER: f32 = 0.022;
const SCALE = 1.35;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) ao: f32,
  @location(4) @interpolate(flat) part: u32,
  @location(5) @interpolate(flat) glow: f32,
}

fn rotZ(v: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let b = birds[ii];
  let part = u32(round(p.w));
  var lp = p.xyz;
  var ln = n.xyz;
  if (part == 3u && t.x != 0.0) {
    // Wing: rotate about the shoulder (an axis along the body); the hand bends further.
    let side = sign(t.x);
    let d = abs(t.x);
    let a = b.pose.x * (smoothstep(0.0, 0.012, d) + 0.6 * smoothstep(0.035, 0.075, d)) * side;
    let pivot = vec2f(side * SHOULDER, 0.008);
    let r = rotZ(lp.xy - pivot, a) + pivot;
    lp = vec3f(r, lp.z);
    ln = vec3f(rotZ(ln.xy, a), ln.z);
  }
  lp *= SCALE;
  // Pitch about x (positive dips the head), then heading about y.
  let cp = cos(-b.pose.z);
  let sp = sin(-b.pose.z);
  lp = vec3f(lp.x, lp.y * cp + lp.z * sp, -lp.y * sp + lp.z * cp);
  ln = vec3f(ln.x, ln.y * cp + ln.z * sp, -ln.y * sp + ln.z * cp);
  let cy = cos(b.pos.w);
  let sy = sin(b.pos.w);
  lp = vec3f(lp.x * cy + lp.z * sy, lp.y, -lp.x * sy + lp.z * cy);
  ln = vec3f(ln.x * cy + ln.z * sy, ln.y, -ln.x * sy + ln.z * cy);
  let world = b.pos.xyz + lp;
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = ln;
  // Painted colours are sRGB; a little per-bird variation in warmth.
  let warm = 0.92 + 0.16 * fract(b.pose.w * 7.3);
  out.albedo = pow(e.rgb, vec3f(2.2)) * vec3f(warm, 1.0, 2.0 - warm);
  out.ao = e.a;
  out.part = part;
  out.glow = max(b.pose.w - 1.0, 0.0);
  return out;
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  let v = normalize(G.camPos - frag.world);
  let l = G.sunDir;
  // Soft wrapped light for down and feathers; thin wing and tail feathers glow when backlit.
  let wrap = clamp((dot(n, l) + 0.35) / 1.35, 0.0, 1.0);
  let thin = select(0.0, 0.5, frag.part == 3u || frag.part == 4u);
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 4.0) * thin;
  var col = frag.albedo * (ambientSky(n, s) * 0.75 * frag.ao + G.sunColor * (wrap * 0.85 + back));
  if (frag.part == 2u) {
    // Eyes: a bright wet glint.
    let h = normalize(l + v);
    col += G.sunColor * pow(max(dot(n, h), 0.0), 80.0) * 2.0;
  }
  if (frag.glow > 0.0) {
    // A spirit bird: lit from within, with a bright turquoise rim.
    let rim = pow(1.0 - abs(dot(n, v)), 2.0);
    col += frag.albedo * frag.glow * 0.35 + vec3f(0.45, 0.95, 1.0) * rim * frag.glow * 0.35;
  }
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}

// ---- Halo: the light a glowing bird gives off (additive, around its body) ----

struct HOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) k: f32,
}

@vertex
fn vs_halo(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> HOut {
  let b = birds[ii];
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let toCam = G.camPos - b.pos.xyz;
  let d = length(toCam);
  let view = toCam / d;
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), view));
  let up = cross(view, right);
  // Grows a little with distance, so the light still reads from far off.
  let size = 0.7 + d * 0.008;
  var out: HOut;
  out.pos = G.viewProj * vec4f(b.pos.xyz + vec3f(0.0, 0.05, 0.0) + (right * c.x + up * c.y) * size, 1.0);
  out.uv = c;
  out.k = max(b.pose.w - 1.0, 0.0) * (0.85 + 0.15 * sin(G.time * 3.0));
  return out;
}

@fragment
fn fs_halo(frag: HOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let light = exp(-r2 * 5.0) * 0.9 + exp(-r2 * 28.0) * 1.6;
  let col = mix(vec3f(1.0, 0.75, 0.35), vec3f(0.45, 0.9, 1.0), exp(-r2 * 6.0));
  return vec4f(col * light * frag.k * mix(0.35, 0.5, G.night), 0.0);
}
