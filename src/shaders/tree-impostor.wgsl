// Far-tree impostor: a camera-facing billboard that blends the two nearest baked views.
// The atlas holds `frames` orthographic views around the tree (albedo + tree-space normals).
import { SNOW, seasonHash, seasonWeights, springLeaf } from "./lib/season.wgsl";
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFogPre, morningFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { TreeInstance, autumnLeaf, lodKeep, rotateYaw, treeSway, windGust } from "./lib/tree.wgsl";

struct ImpostorParams {
  frames: f32,
  cols: f32,
  rows: f32,
  size: f32,
  centerY: f32,
  height: f32,
  leafTint: f32,
  deciduous: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> trees: array<TreeInstance>;
@group(0) @binding(2) var<uniform> imp: ImpostorParams;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var albedoAtlas: texture_2d<f32>;
@group(0) @binding(5) var normalAtlas: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  // Morning fog evaluated per vertex (see morningFog).
  @location(14) fog: vec4f,
  @location(1) uv: vec2f,
  @location(2) @interpolate(flat) frames: vec2f,
  @location(3) blend: f32,
  @location(4) @interpolate(flat) rot: vec2f,
  // x = autumn amount, y = tree seed
  @location(5) @interpolate(flat) autumn: vec2f,
  @location(6) @interpolate(flat) lodFade: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = trees[ii];
  let scale = inst.root.w;
  var corners = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
  let c = corners[vi];

  let root = inst.root.xyz;
  let toCam = G.camPos - root;
  let flat = normalize(vec3f(toCam.x, 0.0, toCam.z) + vec3f(1e-5, 0.0, 0.0));
  let right = normalize(cross(-flat, vec3f(0.0, 1.0, 0.0)));
  // View direction in the tree's own frame selects the baked view.
  let localDir = rotateYaw(flat, vec2f(inst.rot.x, -inst.rot.y));
  let angle = atan2(localDir.x, localDir.z);
  let f = fract(angle / 6.2831853) * imp.frames;
  let f0 = floor(f);
  let f1 = f0 + 1.0;

  let size = imp.size * scale;
  let center = root + vec3f(0.0, imp.centerY * scale, 0.0);
  // uv: x right, y down (atlas convention); quad grows from the bottom.
  var world = center + right * (c.x - 0.5) * size + vec3f(0.0, (c.y - 0.5) * size, 0.0);
  // Push the card toward the camera a little so it does not clip into hillsides.
  world = world + flat * size * 0.15;
  let heightN = clamp((world.y - root.y) / (imp.height * scale), 0.0, 1.2);
  let gust = windGust(root.xz, G.time, G.windDir);
  world = world + treeSway(heightN, heightN, G.windDir, G.windStrength * (0.55 + 0.9 * gust), G.time, inst.rot.z) * scale;

  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.fog = morningFog(out.world, G.camPos, SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor), vec4f(G.mist, G.mistBase, G.canopy, G.time));
  out.uv = vec2f(c.x, 1.0 - c.y);
  out.frames = vec2f(f0 % imp.frames, f1 % imp.frames);
  out.blend = f - f0;
  out.rot = inst.rot.xy;
  out.autumn = vec2f(inst.env.x * imp.deciduous, inst.rot.z);
  out.lodFade = inst.rot.w;
  return out;
}

fn atlasUv(frame: f32, uv: vec2f) -> vec2f {
  let col = frame % imp.cols;
  let row = floor(frame / imp.cols);
  return (vec2f(col, row) + clamp(uv, vec2f(0.002), vec2f(0.998))) / vec2f(imp.cols, imp.rows);
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  if (!lodKeep(frag.lodFade, frag.pos.xy)) {
    discard;
  }
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  let uv0 = atlasUv(frag.frames.x, frag.uv);
  let uv1 = atlasUv(frag.frames.y, frag.uv);
  let a0 = textureSample(albedoAtlas, samp, uv0);
  let a1 = textureSample(albedoAtlas, samp, uv1);
  let n0 = textureSample(normalAtlas, samp, uv0).xyz;
  let n1 = textureSample(normalAtlas, samp, uv1).xyz;
  let albedo = mix(a0, a1, frag.blend);
  var a = albedo.a;
  a = clamp((a - 0.5) / max(fwidth(a), 1e-4) + 0.5, 0.0, 1.0);
  if (a < 0.01) {
    discard;
  }
  let nl = normalize(mix(n0, n1, frag.blend) * 2.0 - 1.0);
  let n = rotateYaw(nl, frag.rot);
  let v = normalize(G.camPos - frag.world);
  let l = G.sunDir;
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 2.5) * 0.6;
  // Bark stays as baked; only leafy (green-dominant) texels turn.
  let leafy = smoothstep(0.0, 0.08, albedo.g - albedo.r);
  let sw = seasonWeights(G.season);
  // Winter thins the baked foliage in clumps (bare crowns from afar); bark stays.
  let clump = seasonHash(dot(floor(frag.uv * 40.0), vec2f(1.0, 57.0)) + frag.autumn.y * 31.0);
  if (leafy > 0.5 && clump < (sw.y * 0.25 + sw.z * 0.85) * imp.deciduous) {
    discard;
  }
  let autumnAmt = max(frag.autumn.x, (sw.y + sw.z * 0.6) * imp.deciduous);
  var leafCol = autumnLeaf(albedo.rgb, frag.autumn.y, autumnAmt);
  leafCol = mix(leafCol, springLeaf(leafCol), sw.w * 0.75 * imp.deciduous);
  leafCol = mix(leafCol, SNOW * 0.85, sw.z * (1.0 - imp.deciduous) * smoothstep(0.1, 0.7, nl.y) * 0.8);
  let rgb = mix(albedo.rgb, leafCol, leafy) * imp.leafTint;
  var col = rgb * (ambientSky(n, s) * 0.6 + G.sunColor * (wrapDiffuse(n, l, 0.5) + back) * 0.8);
  col = applyFogPre(col, frag.world, G.camPos, G.fogDensity, s, frag.fog);
  return vec4f(col, a);
}
