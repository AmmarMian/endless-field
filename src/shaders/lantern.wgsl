// Stone lanterns along the path (model: tools/blender/model_lantern.py). Weathered granite
// with lichen and moss; paper windows that glow like a candle at night.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { rotateYaw } from "./lib/tree.wgsl";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";

struct Lantern {
  // xyz = base, w = scale
  root: vec4f,
  // xy = (cos, sin) yaw, z = index (flicker phase), w = unused
  rot: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> lanterns: array<Lantern>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) @interpolate(flat) part: u32,
  @location(4) e: vec4f,
  @location(5) @interpolate(flat) phase: f32,
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let inst = lanterns[ii];
  let world = inst.root.xyz + rotateYaw(p.xyz * inst.root.w, inst.rot.xy);
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = rotateYaw(n.xyz, inst.rot.xy);
  out.uv = t;
  out.part = u32(round(p.w));
  out.e = e;
  out.phase = inst.rot.z;
  return out;
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  let l = G.sunDir;
  let wp = frag.world;
  if (frag.part == 1u) {
    // Shoji paper on a kumiko lattice (u: around the drum, v: height).
    let bars = max(smoothstep(0.86, 0.95, abs(fract(frag.uv.x * 18.0) * 2.0 - 1.0)), smoothstep(0.86, 0.95, abs(fract(frag.uv.y * 22.0) * 2.0 - 1.0)));
    let paper = mix(vec3f(0.86, 0.79, 0.62), vec3f(0.25, 0.17, 0.1), bars * 0.85);
    let t = G.time;
    let k = frag.phase;
    let ki = u32(k);
    let lamp = G.lamps[ki / 4u][ki % 4u];
    let flicker = 0.82 + 0.1 * sin(t * 7.3 + k * 1.7) + 0.08 * sin(t * 13.1 + k * 4.3);
    // Candle light through paper; the lattice bars stay dark against it.
    let glow = vec3f(1.0, 0.6, 0.26) * (3.2 * flicker) * mix(1.0, 0.25, bars);
    var col = paper * (ambientSky(n, s) * 0.4 + G.sunColor * wrapDiffuse(n, l, 0.5) * 0.5) * (1.0 - G.night * 0.9);
    // Lit lanterns glow (a little even by day, fully at night); a flare when just touched.
    col = col + glow * (lamp * mix(0.18, 1.0, G.night) + max(lamp - 1.0, 0.0) * 1.5);
    col = applyFog(col, wp, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
    return vec4f(col, 1.0);
  }
  // Granite: two scales of grain, darker weathering streaks running down from the roof.
  let grain = simplex2d(wp.xz * 9.0 + wp.y * 7.0) * 0.5 + 0.5;
  let mottle = simplex2d(vec2f(wp.x + wp.z, wp.y) * 2.2) * 0.5 + 0.5;
  let streak = smoothstep(0.55, 0.9, simplex2d(vec2f((wp.x - wp.z) * 6.0, wp.y * 0.6)) * 0.5 + 0.5);
  var stone = vec3f(0.37, 0.36, 0.33) * mix(0.75, 1.15, grain) * mix(0.85, 1.1, mottle) * (1.0 - streak * 0.25);
  // Pale lichen rosettes and moss on upward, sheltered faces.
  let lichen = smoothstep(0.62, 0.72, simplex2d(wp.xz * 5.0 + wp.y * 3.0) * 0.5 + 0.5);
  stone = mix(stone, vec3f(0.62, 0.62, 0.5), lichen * 0.6);
  let mossN = simplex2d(wp.xz * 3.5 + wp.y * 2.0) * 0.5 + 0.5;
  let moss = smoothstep(0.3, 0.7, frag.e.y * 1.3 * mossN);
  stone = mix(stone, vec3f(0.12, 0.18, 0.05) * mix(0.8, 1.2, grain), moss);
  let ao = frag.e.z;
  var col = stone * (ambientSky(n, s) * 0.6 * ao + G.sunColor * wrapDiffuse(n, l, 0.3) * 0.9 * ao);
  // Candle light spilling out of the windows onto the platform and roof soffit (the band of
  // the lantern's height around the fire box).
  let lampS = G.lamps[u32(frag.phase) / 4u][u32(frag.phase) % 4u];
  col = col + stone * vec3f(1.0, 0.58, 0.24) * G.night * 0.9 * lampS * (1.0 - smoothstep(0.55, 0.85, frag.e.x)) * smoothstep(0.35, 0.6, frag.e.x) * ao;
  col = applyFog(col, wp, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}
