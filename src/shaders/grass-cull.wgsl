// Grass placement + culling + animation. One thread per world-anchored grid cell around the
// camera; surviving blades are appended to `blades` and counted into indirect draw `args`.
import { Globals, TRAIL_LEN } from "./lib/globals.wgsl";
import { mountainCorridor, mountainHeight, riverInfo, riverUpper, riverValley, terrainBroad, terrainHeightR, terrainNormalM } from "./lib/terrain.wgsl";
import { bedMask } from "./lib/beds.wgsl";
import { Blade, LifeCell, fieldKind, lifeIndex, lifeKey } from "./lib/field.wgsl";
import { pcg2d, unitFloat } from "@vgpu/wgsl-std/hash";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";
import { biome } from "./lib/biome.wgsl";
import { sunflowerField } from "./lib/sunflowers.wgsl";
import { PATH_WIDTH, pathDistance } from "./lib/path.wgsl";
import { seasonWeights } from "./lib/season.wgsl";

struct CullParams {
  // Integer cell of the grid center (camera snapped to the cell size).
  centerCell: vec2i,
  gridSize: u32,
  spacing: f32,
  // Blades exist where rInner <= distance <= rOuter, crossfading over `fade` meters.
  rInner: f32,
  rOuter: f32,
  fade: f32,
  widthScale: f32,
  heightScale: f32,
  // Probability falloff for the far ring (keeps blade count bounded with distance).
  thinStart: f32,
  thinEnd: f32,
  thinMin: f32,
  // Nested LOD: blade identity lives on the finest grid (baseSpacing); this ring holds every
  // k-th blade of it, and every kNext-th blade continues into the next, coarser ring.
  baseSpacing: f32,
  k: i32,
  kNext: i32,
  widthNext: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> P: CullParams;
@group(0) @binding(2) var<storage, read_write> blades: array<Blade>;
@group(0) @binding(3) var<storage, read_write> args: array<atomic<u32>, 4>;
@group(0) @binding(4) var<storage, read> life: array<LifeCell>;
@group(0) @binding(5) var mtnTex: texture_2d_array<f32>;
@group(0) @binding(6) var mtnSamp: sampler;

fn sampleLife(xz: vec2f) -> f32 {
  let cell = vec2i(floor(xz));
  let c = life[lifeIndex(cell)];
  return select(0.0, c.life, c.key == lifeKey(cell));
}

// Which of the 3x3 child cells represents coarse cell `c` at `level` (random per cell).
fn subPick(c: vec2i, level: i32) -> vec2i {
  let hh = pcg2d(bitcast<vec2u>(c) ^ vec2u(u32(level) * 0x632BE5ABu, 0x7F4A7C15u));
  return vec2i(i32(hh.x % 3u), i32(hh.y % 3u));
}

fn inFrustum(c: vec3f, r: f32) -> bool {
  for (var i = 0u; i < 6u; i = i + 1u) {
    let p = G.frustum[i];
    if (dot(p.xyz, c) + p.w < -r) {
      return false;
    }
  }
  return true;
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.gridSize || id.y >= P.gridSize) {
    return;
  }
  let half = i32(P.gridSize / 2u);
  let cell = P.centerCell + vec2i(id.xy) - vec2i(half);
  // Identity on the finest grid: the same blade has the same position, height and rotation in
  // every ring, so moving between rings never swaps one set of blades for another. A coarse
  // cell is represented by a randomly chosen fine blade (picked level by level), so coarse
  // rings are scattered over their whole cells instead of lining up in rows.
  let level = i32(round(log2(f32(P.k)) / log2(3.0)));
  var cellN = cell;
  for (var l = level; l > 0; l = l - 1) {
    cellN = cellN * 3 + subPick(cellN, l);
  }
  let h = pcg2d(bitcast<vec2u>(cellN) ^ vec2u(0x9E3779B9u, 0x85EBCA6Bu));
  let h2 = pcg2d(h);
  let jitter = vec2f(unitFloat(h.x), unitFloat(h.y));
  let xz = (vec2f(cellN) + jitter) * P.baseSpacing;

  let dist = length(xz - G.camPos.xz);
  // Inside rInner the finer ring owns every one of this ring's blades; past rOuter the
  // coarser ring takes over the ones that continue.
  if (dist > P.rOuter || dist < P.rInner) {
    return;
  }
  // This blade continues into the next ring if it is its parent cell's chosen representative.
  let parent = vec2i(floor(vec2f(cell) / 3.0));
  let continues = P.kNext > 0 && all(subPick(parent, level + 1) == cell - parent * 3);
  let edge = smoothstep(P.rOuter - P.fade, P.rOuter, dist);
  let dither = unitFloat(h.y ^ h2.x);
  // Blades that do not continue shrink back into the ground across the outer band (each at
  // its own threshold); the ones that continue stay and widen toward the next ring's width.
  let presence = select(1.0 - edge, 1.0, continues);
  if (dither > presence) {
    return;
  }
  let fade = select(smoothstep(dither * 0.75, dither * 0.75 + 0.25, presence), 1.0, continues);
  let widthMul = select(1.0, mix(1.0, P.widthNext / P.widthScale, edge), continues);
  let keep = 1.0;

  // Meadow structure: large patches of tall grass and shorter lawns.
  let patchN = simplex2d(xz * 0.018) * 0.5 + 0.5;
  let detail = simplex2d(xz * 0.11 + vec2f(31.0, 7.0)) * 0.5 + 0.5;
  let r1 = unitFloat(h2.y);
  // Field kinds: tall seed-head meadows, ordinary grass, short lawns; plus hue patches.
  let kindN = simplex2d(xz * 0.0055 + vec2f(113.0, -41.0) + vec2f(simplex2d(xz * 0.013), simplex2d(xz * 0.013 + 9.0)) * 0.9);
  let hueN = simplex2d(xz * 0.011 + vec2f(-7.0, 59.0) + vec2f(simplex2d(xz * 0.027 + 3.0), simplex2d(xz * 0.027 - 11.0)) * 0.8);
  let bio = biome(xz);
  let kind = fieldKind(xz, kindN + bio.y * 0.35 - bio.x * 0.3 - bio.z * 0.6, hueN + bio.x * 0.35 + bio.z * 0.5);
  var height = mix(0.45, 1.15, patchN) * mix(0.7, 1.2, r1) * mix(0.85, 1.1, detail);
  height = height * mix(1.0, 1.75, kind.x) * mix(1.0, 0.4, kind.y) * mix(1.0, 0.7, bio.z);
  height = height * P.heightScale * fade;
  // Winter snow presses the grass down; spring grass is young and shorter.
  let sw = seasonWeights(G.season);
  height = height * (1.0 - 0.32 * sw.z - 0.15 * sw.w) * (1.0 - 0.14 * G.wet);
  // The lantern path is bare; grass shortens toward its edge (trampled).
  let pd = pathDistance(xz);
  if (pd < PATH_WIDTH + 1.6) {
    let onPath = 1.0 - smoothstep(PATH_WIDTH - 0.2, PATH_WIDTH + 0.4, pd);
    // Bare in the middle (at ~120 blades/m^2 even a few percent would show), thinning at the edges.
    if (onPath > 0.98 || unitFloat(h.y ^ 0x2545F491u) < onPath) {
      return;
    }
    height = height * mix(0.45, 1.0, smoothstep(PATH_WIDTH, PATH_WIDTH + 1.6, pd));
  }
  // Inside the sunflower field only sparse, short weeds grow between the rows.
  let sf = sunflowerField(xz);
  if (sf > 0.01) {
    if (unitFloat(h.x ^ 0x68E31DA4u) < sf * 0.8) {
      return;
    }
    height = height * mix(1.0, 0.45, sf);
  }
  // River: open water has no grass; reeds grow in clumps along the edge (some in the
  // shallows); the field thins out over an irregular beach before taking over.
  let river = riverInfo(xz);
  let hwR = river.z;
  if (river.x < hwR * 0.78) {
    return;
  }
  let edgeJitter = simplex2d(xz * 0.3 + vec2f(5.0, 2.0)) * hwR * 0.25;
  let clump = smoothstep(0.1, 0.5, simplex2d(xz * 0.16 + vec2f(31.0, -4.0)));
  let reed = clump * (1.0 - smoothstep(hwR * 1.25, hwR * 1.6, river.x)) * select(0.0, 1.0, river.x > hwR * 0.78) * (1.0 - smoothstep(0.1, 0.35, riverUpper(river.w)));
  let isReed = reed >= 0.4;
  let shoreKeep = smoothstep(hwR * 1.15, hwR * 2.0, river.x + edgeJitter);
  if (reed < 0.4 && unitFloat(h.x ^ 0x68e31da4u) > shoreKeep) {
    return;
  }
  height = select(height * mix(0.55, 1.0, shoreKeep), height * 1.6, reed >= 0.4);

  // Shade under the forest canopy: only sparse, shorter tufts survive.
  if (bio.z > 0.05 && unitFloat(h.y ^ 0x9e3779b9u) < bio.z * 1.2) {
    return;
  }
  height = height * mix(1.0, 0.6, bio.z);

  // Flower beds: the grass thins and shortens so wildflowers grow up through it.
  let bed = bedMask(xz).x;
  if (bed > 0.0 && unitFloat(h2.y ^ 0x27d4eb2du) < bed * 0.5) {
    return;
  }
  height = height * mix(1.0, 0.55, bed);
  let seedKind = ((kind.x > 0.3 && unitFloat(h.y ^ 0x5bd1e995u) < 0.35 * kind.x) || (reed >= 0.4 && unitFloat(h.y ^ 0x1b873593u) < 0.25));

  let baseY = terrainHeightR(xz, river);
  let mRaw = mountainHeight(xz, mtnTex, mtnSamp, 0.0);
  let y = riverValley(baseY + mRaw * mountainCorridor(river), river, xz);
  let mtn = max(y - (terrainBroad(xz) * 0.6 + 22.0), 0.0) * smoothstep(0.0, 30.0, y - baseY + mRaw);
  let root = vec3f(xz.x, y, xz.y);
  // Bent blades can lean up to ~1.6x their height sideways: cull with a sphere that covers it.
  if (!inFrustum(root + vec3f(0.0, height * 0.5, 0.0), height * 1.7 + 0.8)) {
    return;
  }
  // Alpine ecology: turf shortens with altitude; rock and snow leave only sparse tufts.
  let groundN = terrainNormalM(xz, y, 1.5, river.w, mtnTex, mtnSamp);
  // Distant rings: wide sparse blades on steep ground read as spikes on ridgelines and
  // curtains on valley walls; the ground shading carries those areas instead.
  // Far away, blades on steep ground or high mountain turf shrink away gradually with distance
  // (the ground shading carries those areas) - the same in every ring, so nothing pops.
  let steepFar = select(1.0, 1.0 - smoothstep(22.0, 45.0, dist), groundN.y < 0.82);
  let mtnFar = select(1.0, 1.0 - smoothstep(90.0, 140.0, dist), mtn > 15.0);
  let farFade = steepFar * mtnFar;
  if (farFade <= 0.01) {
    return;
  }
  height = height * farFade;
  var alpine = 0.0;
  if (mtn > 1.0) {
    alpine = smoothstep(60.0, 150.0, mtn);
    let steep = 1.0 - smoothstep(0.62, 0.82, groundN.y);
    let snow = smoothstep(190.0, 230.0, mtn + simplex2d(xz * 0.004) * 35.0) * smoothstep(0.45, 0.75, groundN.y);
    let bare = max(smoothstep(0.45, 0.8, steep), snow);
    if (unitFloat(h2.x ^ 0x2545f491u) < bare * 0.97) {
      return;
    }
    height = height * mix(1.0, 0.32, alpine);
  }
  let alpineGreen = alpine * 0.55;
  let isSeed = seedKind && alpine < 0.3;
  let hueQ = round(clamp(select(mix(kind.z, 0.7, alpine), 0.8, reed >= 0.4), -1.0, 1.0) * 7.0) + 7.0;

  // Wind: ripple waves (~8 m/s, ~12 m wavelength) riding on slow broad gusts, plus flutter.
  let wdir = G.windDir;
  let t = G.time;
  let along = dot(xz, wdir);
  let ripple = simplex2d(vec2f(along * 0.085 - t * 0.68, dot(xz, vec2f(-wdir.y, wdir.x)) * 0.03));
  let broad = simplex2d(xz * 0.014 - wdir * t * 0.09);
  let gust = smoothstep(-0.6, 0.9, broad * 0.6 + ripple * 0.55);
  let flutter = simplex2d(xz * 0.5 + vec2f(t * 3.1, -t * 2.3));
  var bend = wdir * G.windStrength * (0.1 + 0.85 * gust) + vec2f(-wdir.y, wdir.x) * flutter * 0.16 * G.windStrength;
  // Raindrops knock the blades about (only while it rains: two noise lookups per blade).
  if (G.rain > 0.001) {
    bend = bend + vec2f(simplex2d(xz * 3.1 + vec2f(t * 7.0, 0.0)), simplex2d(xz * 3.1 + vec2f(0.0, t * 6.3))) * 0.22 * G.rain;
  }

  // The petal stream parts the grass as it passes.
  let toPlayer = distance(xz, G.playerPos.xz);
  // Animals in the grass: blades lean away from them and lie lower around them (hares,
  // deer, foxes stay in sight, and the grass parts as they run).
  if (distance(xz, G.camPos.xz) < 90.0) {
    for (var i = 0u; i < 16u; i = i + 1u) {
      let c = G.critters[i];
      if (c.w <= 0.0) {
        continue;
      }
      let d = xz - c.xz;
      let dl = length(d);
      if (dl < c.w * 2.6) {
        height = height * mix(0.3, 1.0, smoothstep(c.w * 0.5, c.w * 2.6, dl));
        bend = bend + d / max(dl, 0.05) * (1.0 - smoothstep(c.w * 0.8, c.w * 2.6, dl)) * 0.9;
      }
    }
  }
  // A resting bird: the grass under and around it lies pressed down, a little form in the
  // meadow it can be seen in.
  if (G.rest > 0.0) {
    height = height * mix(1.0, mix(0.16, 1.0, smoothstep(0.4, 3.2, toPlayer)), G.rest);
  }
  if (toPlayer < 40.0) {
    var push = vec2f(0.0);
    for (var i = 0u; i < TRAIL_LEN; i = i + 1u) {
      let tp = G.trail[i];
      if (tp.w <= 0.001) {
        continue;
      }
      let w = tp.w;
      let d = xz - tp.xz;
      let d2 = dot(d, d);
      let radius = 2.6 + G.gust * 1.5;
      let fall = exp(-d2 / (radius * radius));
      let above = tp.y - y;
      let vert = 1.0 - smoothstep(1.2, 5.5, above);
      let dirAway = d / max(sqrt(d2), 0.05);
      push = push + dirAway * fall * vert * w;
    }
    bend = bend + push * 1.6;
  }
  let bl = length(bend);
  if (bl > 1.6) {
    bend = bend * (1.6 / bl);
  }

  let angle = unitFloat(h.x ^ h2.y) * 6.2831853;
  var width = mix(0.035, 0.06, unitFloat(h2.x ^ h.y)) * P.widthScale * widthMul;
  width = width * mix(1.0, 0.7, kind.x) * select(1.0, 0.75, isReed);
  // Waterside plants stay green even before the land is restored.
  let lifeV = max(max(sampleLife(xz), select(0.0, 0.4, isReed)), alpineGreen);

  let idx = atomicAdd(&args[1], 1u);
  if (idx >= arrayLength(&blades)) {
    return;
  }
  var b: Blade;
  b.root = vec4f(root, height);
  b.shape = vec4f(cos(angle), sin(angle), width, lifeV);
  let packedKind = min(patchN, 0.999) + select(0.0, 2.0, isSeed) + 4.0 * hueQ;
  b.bend = vec4f(bend * select(1.0, 1.25, isSeed), r1, packedKind);
  b.ground = vec4f(groundN, 0.0);
  blades[idx] = b;
}
