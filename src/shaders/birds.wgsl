// Small meadow birds: a faceted body, tail and two-jointed wings built in the vertex shader.
// The CPU sets each bird's pose (position, heading, pitch, flap angle, wing spread).
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Bird {
  // xyz = position, w = heading (yaw)
  pos: vec4f,
  // x = flap angle, y = wing spread (0 folded .. 1 open), z = pitch, w = seed
  pose: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> birds: array<Bird>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) part: u32,
  @location(3) @interpolate(flat) seed: f32,
}

const SCALE = 1.7;

fn bodyVertex(i: u32) -> vec3f {
  // 8 faces between nose/tail-end and a ring of back, right, belly, left.
  let ring = array<vec3f, 4>(vec3f(0.0, 0.04, 0.0), vec3f(0.035, 0.0, 0.01), vec3f(0.0, -0.035, 0.0), vec3f(-0.035, 0.0, 0.01));
  let face = i / 3u;
  let k = i % 3u;
  let tip = select(vec3f(0.0, 0.01, 0.13), vec3f(0.0, 0.005, -0.1), face >= 4u);
  let r = face % 4u;
  if (k == 0u) {
    return tip;
  }
  let a = select(r, (r + 1u) % 4u, (k == 1u) != (face >= 4u));
  let b = select((r + 1u) % 4u, r, (k == 1u) != (face >= 4u));
  return select(ring[b], ring[a], k == 1u);
}

/** Wing point: `u` along the span (0 root .. 1 tip), `v` across the chord (0 front .. 1 back). */
fn wingPoint(u: f32, v: f32, side: f32, flap: f32, spread: f32) -> vec3f {
  let span = 0.22 * mix(0.25, 1.0, spread);
  let chord = mix(0.07, 0.03, u) ;
  // Two joints: the hand bends further than the arm.
  let arm = min(u, 0.5) * span;
  let hand = max(u - 0.5, 0.0) * span;
  let a1 = flap;
  let a2 = flap * 1.5;
  var p = vec3f(arm * cos(a1) + hand * cos(a2), arm * sin(a1) + hand * sin(a2), 0.03 - v * chord - u * 0.05 * (1.0 - spread));
  // Folded wings lie back along the body.
  p = mix(vec3f(0.03, 0.02, 0.02 - u * 0.13 - v * 0.02), p, spread);
  return vec3f(p.x * side, p.y + 0.012, p.z);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let b = birds[ii];
  var local: vec3f;
  var part = 0u;
  if (vi < 24u) {
    local = bodyVertex(vi);
  } else if (vi < 27u) {
    let tail = array<vec3f, 3>(vec3f(0.0, 0.01, -0.07), vec3f(-0.045, 0.0, -0.17), vec3f(0.045, 0.0, -0.17));
    local = tail[vi - 24u];
    part = 1u;
  } else {
    // Two wings x two segments x two triangles.
    let w = vi - 27u;
    let side = select(1.0, -1.0, w >= 12u);
    let q = w % 12u;
    let seg = f32(q / 6u);
    let c = q % 6u;
    let uv = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0))[c];
    local = wingPoint((seg + uv.x) * 0.5, uv.y, side, b.pose.x, b.pose.y);
    part = 2u;
  }
  local *= SCALE;
  // Pitch about x, then heading about y.
  let cp = cos(b.pose.z);
  let sp = sin(b.pose.z);
  local = vec3f(local.x, local.y * cp + local.z * sp, -local.y * sp + local.z * cp);
  let cy = cos(b.pos.w);
  let sy = sin(b.pos.w);
  local = vec3f(local.x * cy + local.z * sy, local.y, -local.x * sy + local.z * cy);
  let world = b.pos.xyz + local;
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = vec3f(0.0, 1.0, 0.0);
  out.part = part;
  out.seed = b.pose.w;
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(cross(dpdx(frag.world), dpdy(frag.world)));
  let v = normalize(G.camPos - frag.world);
  if (dot(n, v) < 0.0) {
    n = -n;
  }
  // Warm brown sparrow tones, paler underneath; wings a little darker.
  let tone = fract(frag.seed * 7.3);
  var base = mix(vec3f(0.32, 0.22, 0.14), vec3f(0.42, 0.33, 0.24), tone);
  base = mix(base, vec3f(0.62, 0.56, 0.48), smoothstep(0.2, -0.6, n.y) * f32(frag.part == 0u));
  base *= select(1.0, 0.75, frag.part == 2u);
  let l = G.sunDir;
  var col = base * (ambientSky(n, s) * 0.8 + G.sunColor * (max(dot(n, l), 0.0) * 0.8 + 0.06));
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}

