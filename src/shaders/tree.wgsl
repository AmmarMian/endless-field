// Tree meshes (LOD0/LOD1). One vertex stage; `fs_bark` / `fs_leaves` are selected per draw.
import { SNOW, seasonHash, seasonWeights, springLeaf } from "./lib/season.wgsl";
import { Globals, TRAIL_LEN } from "./lib/globals.wgsl";
import { LANTERN_COLOR, lanternFirst, lanternTerm } from "./lib/path.wgsl";
import { SkyParams, applyFogPre, morningFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { TreeInstance, autumnLeaf, lodKeep, rotateYaw, treeSway, windGust } from "./lib/tree.wgsl";

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

// Warm light from the lit path lanterns at night (see lib/path.wgsl).
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
  // Morning fog evaluated per vertex (see morningFog).
  @location(14) fog: vec4f,
  // Warm light from the path lanterns at night (per vertex).
  @location(13) lamp: vec3f,
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
  // Wind in three layers: the whole tree leans with the passing gusts; branches sway
  // independently across the crown (spatially coherent, growing toward the crown's edge);
  // leaves flutter fast, each vertex on its own phase so cards twist rather than slide.
  let gust = windGust(inst.root.xz, G.time, G.windDir);
  let strength = G.windStrength * (0.55 + 0.9 * gust);
  world = world + treeSway(e.x, heightN, G.windDir, strength, G.time, seed) * scale;
  let radial = length(local.xz) / max(tree.height * 0.5, 0.1);
  let reach = clamp(radial * 0.7 + heightN * 0.5, 0.0, 1.5);
  let bp = local * 0.35 + vec3f(seed * 17.0);
  let branch = vec3f(
    sin(G.time * 1.3 + bp.x + bp.y * 0.7) + 0.5 * sin(G.time * 2.3 + bp.z * 1.3),
    0.35 * sin(G.time * 1.7 + bp.y + bp.z),
    sin(G.time * 1.1 + bp.z + bp.x * 0.6) + 0.5 * sin(G.time * 2.9 + bp.x * 1.1),
  );
  let windW = vec3f(G.windDir.x, 0.0, G.windDir.y);
  world = world + (branch * 0.05 + windW * 0.06 * gust) * reach * strength * scale;
  // The player's wind: passing through or beside a tree pushes its leaves and twigs away from
  // the trail and sets the leaves rustling (as the grass parts).
  var rustle = 0.0;
  if (distance(inst.root.xz, G.playerPos.xz) < 25.0) {
    var push = vec3f(0.0);
    for (var i = 0u; i < TRAIL_LEN; i = i + 1u) {
      let tp = G.trail[i];
      if (tp.w <= 0.001) {
        continue;
      }
      let d = world - tp.xyz;
      let d2 = dot(d, d);
      let radius = 3.2 + G.gust * 2.0;
      push = push + d / max(sqrt(d2), 0.1) * exp(-d2 / (radius * radius)) * tp.w;
    }
    let pl = length(push);
    rustle = min(pl, 1.5);
    world = world + push * 0.22 * clamp(0.3 + reach, 0.0, 1.3) * select(0.35, 1.0, e.w > 0.9);
  }
  if (e.w > 0.9) {
    let ph = e.y * 6.2831 + dot(local, vec3f(3.1, 2.3, 2.7));
    let flutter = sin(G.time * (7.0 + e.y * 5.0) + ph) * 0.7 + sin(G.time * 12.5 + ph * 1.7) * 0.3;
    let wn = rotateYaw(n.xyz, inst.rot.xy);
    world = world + wn * flutter * 0.045 * scale * (0.25 + strength * 1.1 + rustle * 1.6) * clamp(0.4 + reach, 0.0, 1.2);
  }
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.lamp = lampLight(out.world);
  out.fog = morningFog(out.world, G.camPos, SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor), vec4f(G.mist, G.mistBase, G.canopy, G.time));
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
  col = col + albedo * frag.lamp;
  col = applyFogPre(col, frag.world, G.camPos, G.fogDensity, s, frag.fog);
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
  // Seasons: broadleaves turn in autumn, drop their leaves (stochastically, per card) into
  // winter and flush light green in spring; evergreens carry snow on their upper needles.
  let sw = seasonWeights(G.season);
  let fall = (sw.y * 0.25 + sw.z * 0.93) * tree.deciduous;
  if (seasonHash(frag.extra.y * 97.0 + frag.seed * 13.0) < fall) {
    discard;
  }
  let autumnAmt = max(frag.autumn, (sw.y + sw.z * 0.6) * tree.deciduous);
  var albedo = autumnLeaf(texel.rgb * hue * tree.leafTint, frag.seed, autumnAmt);
  albedo = mix(albedo, springLeaf(albedo), sw.w * 0.75 * tree.deciduous);
  albedo = mix(albedo, SNOW * 0.85, sw.z * (1.0 - tree.deciduous) * smoothstep(0.1, 0.7, normalize(frag.normal).y) * 0.8);
  let diff = wrapDiffuse(n, l, 0.5);
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 2.5) * 0.9;
  var col = albedo * (ambientSky(n, s) * 0.6 * ao * mix(0.55, 1.0, frag.shade) + G.sunColor * (diff * ao + back * (0.4 + 0.6 * ao)) * 0.85 * frag.shade);
  col = col + albedo * frag.lamp;
  col = applyFogPre(col, frag.world, G.camPos, G.fogDensity, s, frag.fog);
  return vec4f(col, a);
}
