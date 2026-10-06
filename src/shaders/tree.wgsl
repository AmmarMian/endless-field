// Tree meshes (LOD0/LOD1). One vertex stage; `fs_bark` / `fs_leaves` are selected per draw.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { TreeInstance, autumnLeaf, lodKeep, rotateYaw, treeSway } from "./lib/tree.wgsl";

struct TreeParams {
  height: f32,
  leafTint: f32,
  // 1 for broadleaf species that turn in autumn, 0 for evergreens.
  deciduous: f32,
  pad1: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> trees: array<TreeInstance>;
@group(0) @binding(2) var<uniform> tree: TreeParams;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var trunkDiff: texture_2d<f32>;
@group(0) @binding(5) var trunkNor: texture_2d<f32>;
@group(0) @binding(6) var branchesDiff: texture_2d<f32>;
@group(0) @binding(7) var branchesNor: texture_2d<f32>;
@group(0) @binding(8) var leaves: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) extra: vec4f,
  @location(4) @interpolate(flat) seed: f32,
  @location(5) localY: f32,
  @location(6) shade: f32,
  @location(7) @interpolate(flat) autumn: f32,
  @location(8) @interpolate(flat) lodFade: f32,
}

fn sky() -> SkyParams {
  return SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
}

@vertex
fn vs_main(
  @location(0) p: vec4f,
  @location(1) n: vec4f,
  @location(2) t: vec2f,
  @location(3) e: vec4f,
  @builtin(instance_index) ii: u32,
) -> VOut {
  let inst = trees[ii];
  let scale = inst.root.w;
  let seed = inst.rot.z;
  var local = p.xyz;
  let heightN = clamp(local.y / tree.height, 0.0, 1.2);
  var world = inst.root.xyz + rotateYaw(local * scale, inst.rot.xy);
  world = world + treeSway(e.x, heightN, G.windDir, G.windStrength, G.time, seed) * scale;
  // Leaf flutter: small, fast, per-card phase.
  if (e.w > 0.9) {
    let ph = e.y * 6.2831 + G.time * (5.0 + e.y * 3.0);
    world = world + rotateYaw(n.xyz, inst.rot.xy) * sin(ph) * 0.025 * scale * (0.4 + G.windStrength);
  }
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = rotateYaw(n.xyz, inst.rot.xy);
  out.uv = t;
  out.extra = e;
  out.seed = seed;
  out.localY = local.y * scale;
  // Under the canopy (lower ~60% of the tree) the forest's shade applies; crowns stay lit.
  let under = 1.0 - smoothstep(0.35, 0.75, heightN);
  out.shade = mix(1.0, inst.env.y, under);
  out.autumn = inst.env.x * tree.deciduous;
  out.lodFade = inst.rot.w;
  return out;
}

// Cotangent frame from screen derivatives (no tangents stored in the mesh).
fn perturb(n: vec3f, world: vec3f, uv: vec2f, tn: vec3f) -> vec3f {
  let dp1 = dpdx(world);
  let dp2 = dpdy(world);
  let duv1 = dpdx(uv);
  let duv2 = dpdy(uv);
  let dp2perp = cross(dp2, n);
  let dp1perp = cross(n, dp1);
  let tg = dp2perp * duv1.x + dp1perp * duv2.x;
  let bt = dp2perp * duv1.y + dp1perp * duv2.y;
  let invmax = inverseSqrt(max(dot(tg, tg), dot(bt, bt)) + 1e-12);
  return normalize(tg * invmax * tn.x - bt * invmax * tn.y + n * tn.z);
}

@fragment
fn fs_bark(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  if (!lodKeep(frag.lodFade, frag.pos.xy)) {
    discard;
  }
  let s = sky();
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  // Sample both bark sets (uniform control flow), then pick trunk or branches.
  let isTrunk = frag.extra.w < 0.25;
  let albedo = select(textureSample(branchesDiff, samp, frag.uv).rgb, textureSample(trunkDiff, samp, frag.uv).rgb, isTrunk);
  let tn = select(textureSample(branchesNor, samp, frag.uv).xyz, textureSample(trunkNor, samp, frag.uv).xyz, isTrunk) * 2.0 - 1.0;
  n = perturb(n, frag.world, frag.uv, tn);
  let l = G.sunDir;
  // Darken toward the ground where grass and roots crowd the trunk.
  let ao = mix(0.45, 1.0, smoothstep(0.0, 2.5, frag.localY));
  var col = albedo * (ambientSky(n, s) * 0.55 * ao * mix(0.5, 1.0, frag.shade) + G.sunColor * wrapDiffuse(n, l, 0.2) * 0.9 * frag.shade * frag.shade);
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec3f(G.mist, G.mistBase, G.canopy));
  return vec4f(col, 1.0);
}

@fragment
fn fs_leaves(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  if (!lodKeep(frag.lodFade, frag.pos.xy)) {
    discard;
  }
  let s = sky();
  let texel = textureSample(leaves, samp, frag.uv);
  // Crisp alpha-to-coverage: sharpen alpha to a ~1px ramp, and boost it in smaller mips so
  // distant foliage keeps its coverage instead of thinning out.
  let dims = vec2f(textureDimensions(leaves));
  let duv = max(length(dpdx(frag.uv * dims)), length(dpdy(frag.uv * dims)));
  let mip = max(log2(duv), 0.0);
  var a = texel.a * (1.0 + mip * 0.3);
  a = clamp((a - 0.5) / max(fwidth(a), 1e-4) + 0.5, 0.0, 1.0);
  if (a < 0.01) {
    discard;
  }
  var n = normalize(frag.normal);
  let v = normalize(G.camPos - frag.world);
  if (dot(n, v) < 0.0 && !front) {
    n = -n;
  }
  let l = G.sunDir;
  let ao = frag.extra.z;
  let hue = 0.92 + 0.16 * fract(frag.seed * 7.13);
  let albedo = autumnLeaf(texel.rgb * hue * tree.leafTint, frag.seed, frag.autumn);
  let diff = wrapDiffuse(n, l, 0.5);
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 2.5) * 0.9;
  var col = albedo * (ambientSky(n, s) * 0.6 * ao * mix(0.55, 1.0, frag.shade) + G.sunColor * (diff * ao + back * (0.4 + 0.6 * ao)) * 0.85 * frag.shade);
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec3f(G.mist, G.mistBase, G.canopy));
  return vec4f(col, a);
}
