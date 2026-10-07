// The two vermilion torii at the ends of the lantern path (model: tools/blender/
// model_shrine.py --torii-only). The material comes from the part id in p.w.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { rotateYaw } from "./lib/tree.wgsl";
import { LANTERN_COLOR, lanternFirst, lanternTerm } from "./lib/path.wgsl";
import { SNOW, seasonWeights } from "./lib/season.wgsl";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";

struct Gate {
  // xyz = base, w = glow (the lantern path completed)
  origin: vec4f,
  // xy = (cos, sin) rotation
  rot: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> gates: array<Gate>;
@group(0) @binding(2) var samp: sampler;
// Poly Haven (CC0) surfaces: hinoki planks (under the lacquer) and rock (the base stones).
@group(0) @binding(3) var woodDiff: texture_2d<f32>;
@group(0) @binding(4) var woodNor: texture_2d<f32>;
@group(0) @binding(5) var rockDiff: texture_2d<f32>;
@group(0) @binding(6) var rockNor: texture_2d<f32>;

// Triplanar sample of a color texture (world-space, `scale` repeats per meter).
fn tri(t: texture_2d<f32>, p: vec3f, w: vec3f, scale: f32) -> vec3f {
  return (textureSample(t, samp, p.zy * scale).rgb * w.x + textureSample(t, samp, p.xz * scale).rgb * w.y + textureSample(t, samp, p.xy * scale).rgb * w.z);
}

// Triplanar normal mapping: each projection's tangent-space normal (OpenGL convention)
// tilts the surface normal within its own plane (UDN blend).
fn triNormal(t: texture_2d<f32>, p: vec3f, n: vec3f, w: vec3f, scale: f32, strength: f32) -> vec3f {
  let tx = textureSample(t, samp, p.zy * scale).xy * 2.0 - 1.0;
  let ty = textureSample(t, samp, p.xz * scale).xy * 2.0 - 1.0;
  let tz = textureSample(t, samp, p.xy * scale).xy * 2.0 - 1.0;
  let perturb = vec3f(0.0, tx.y, tx.x) * w.x + vec3f(ty.x, 0.0, ty.y) * w.y + vec3f(tz.x, tz.y, 0.0) * w.z;
  return normalize(n + perturb * strength);
}

fn lampLight(p: vec3f) -> vec3f {
  let k0 = lanternFirst(p, G.lampPos[1].w, G.lampPos[2].w, G.night);
  if (k0 < -50) {
    return vec3f(0.0);
  }
  let count = i32(G.lampPos[0].w);
  var sum = 0.0;
  for (var k = max(k0, 0); k < min(k0 + 5, count); k = k + 1) {
    sum = sum + lanternTerm(p, G.lampPos[k].xyz, G.lamps[u32(k) / 4u][u32(k) % 4u], f32(k), G.time);
  }
  return LANTERN_COLOR * sum * G.night;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) part: u32,
  @location(3) e: vec4f,
  @location(4) local: vec3f,
  @location(5) @interpolate(flat) glow: f32,
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let gate = gates[ii];
  let world = gate.origin.xyz + rotateYaw(p.xyz, gate.rot.xy);
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = rotateYaw(n.xyz, gate.rot.xy);
  out.part = u32(round(p.w));
  out.e = e;
  out.local = p.xyz;
  out.glow = gate.origin.w;
  return out;
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  let wp = frag.world;
  let lp = frag.local;
  let v = normalize(G.camPos - wp);
  let l = G.sunDir;
  let grain = simplex2d(vec2f(lp.x * 3.0 + lp.z * 2.0, lp.y * 40.0)) * 0.5 + 0.5;
  // Triplanar weights from the (geometric) normal, sharpened so faces take one projection.
  var tw = pow(abs(n), vec3f(6.0));
  tw = tw / (tw.x + tw.y + tw.z);
  // All textures are sampled in uniform control flow (a WGSL rule); materials pick from them.
  let wood = tri(woodDiff, wp, tw, 0.8);
  let rock = tri(rockDiff, wp, tw, 1.3);
  let nWood = triNormal(woodNor, wp, n, tw, 0.8, 0.8);
  let nRock = triNormal(rockNor, wp, n, tw, 1.3, 0.9);
  var albedo = vec3f(0.5);
  var spec = 0.0;
  switch (frag.part) {
    case 0u: {
      // Vermilion lacquer over hinoki: the grain shows through, worn paler where rain runs.
      let lum = dot(wood, vec3f(0.3, 0.59, 0.11));
      let wear = smoothstep(0.6, 0.9, simplex2d(vec2f(lp.x + lp.z, lp.y * 0.8) * 3.0) * 0.5 + 0.5);
      albedo = mix(vec3f(0.6, 0.09, 0.03), vec3f(0.52, 0.2, 0.11), wear * 0.45) * (0.65 + 0.85 * lum);
      n = nWood;
      spec = 0.18;
    }
    case 1u: {
      albedo = vec3f(0.025, 0.022, 0.02);
      spec = 0.45;
    }
    default: {
      // Base stones: grey, finely tooled.
      let g = dot(rock, vec3f(0.33));
      albedo = mix(vec3f(g), rock, 0.25) * 1.05;
      n = nRock;
    }
  }
  // Winter: snow on the upward faces (roof, lintel, plinths, the foxes' heads).
  let sw = seasonWeights(G.season);
  albedo = mix(albedo, SNOW, sw.z * smoothstep(0.55, 0.85, n.y) * 0.85);
  let ao = frag.e.z;
  var col = albedo * (ambientSky(n, s) * 0.6 * ao + G.sunColor * wrapDiffuse(n, l, 0.3) * 0.9);
  let h = normalize(v + l);
  col = col + G.sunColor * spec * pow(max(dot(n, h), 0.0), 60.0) * 0.6;
  col = col + albedo * lampLight(wp);
  // The lantern path completed: the vermilion gates glow from within, breathing slowly.
  if (frag.part == 0u && frag.glow > 0.0) {
    let breathe = 0.75 + 0.25 * sin(G.time * 1.3 + wp.x * 0.05);
    col += vec3f(1.0, 0.36, 0.12) * frag.glow * breathe * mix(0.35, 1.2, G.night);
  }
  col = applyFog(col, wp, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}
