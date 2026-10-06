// Grass placement + culling + animation. One thread per world-anchored grid cell around the
// camera; surviving blades are appended to `blades` and counted into indirect draw `args`.
import { Globals, TRAIL_LEN } from "./lib/globals.wgsl";
import { mountainCorridor, mountainHeight, riverInfo, riverUpper, riverValley, terrainBroad, terrainHeightR, terrainNormalM } from "./lib/terrain.wgsl";
import { bedMask } from "./lib/beds.wgsl";
import { Blade, LifeCell, fieldKind, lifeIndex, lifeKey } from "./lib/field.wgsl";
import { pcg2d, unitFloat } from "@vgpu/wgsl-std/hash";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";
import { biome } from "./lib/biome.wgsl";

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
  let h = pcg2d(bitcast<vec2u>(cell) ^ vec2u(0x9E3779B9u, 0x85EBCA6Bu));
  let h2 = pcg2d(h);
  let jitter = vec2f(unitFloat(h.x), unitFloat(h.y));
  let xz = (vec2f(cell) + jitter) * P.spacing;

  let dist = length(xz - G.camPos.xz);
  if (dist > P.rOuter || dist < P.rInner - P.fade) {
    return;
  }
  // Thin the far ring stochastically; survivors widen to keep coverage.
  let keep = mix(1.0, P.thinMin, smoothstep(P.thinStart, P.thinEnd, dist));
  let r0 = unitFloat(h2.x);
  if (r0 > keep) {
    return;
  }
  // LOD crossfade by dithering blades in/out at full height (shrinking both rings in the
  // overlap leaves a visible trough). The outermost edge still shrinks into the ground.
  let fadeOut = 1.0 - smoothstep(P.rOuter - P.fade, P.rOuter, dist);
  let fadeIn = smoothstep(P.rInner - P.fade, P.rInner, dist);
  let dither = unitFloat(h.y ^ h2.x);
  let isFarRing = P.rInner > 0.0;
  // Inner edges crossfade with the previous ring; the outer edge thins out stochastically
  // over a long band so the field never ends in a visible line.
  let longFade = 1.0 - smoothstep(P.rOuter - P.fade * 8.0, P.rOuter, dist);
  let presence = select(fadeOut, fadeIn * longFade, isFarRing);
  if (dither > presence) {
    return;
  }
  let fade = select(1.0, mix(0.6, 1.0, longFade), isFarRing);

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
  if (!inFrustum(root + vec3f(0.0, height * 0.5, 0.0), height * 0.8 + 0.6)) {
    return;
  }
  // Alpine ecology: turf shortens with altitude; rock and snow leave only sparse tufts.
  let groundN = terrainNormalM(xz, y, 1.5, river.w, mtnTex, mtnSamp);
  // Distant rings: wide sparse blades on steep ground read as spikes on ridgelines and
  // curtains on valley walls; the ground shading carries those areas instead.
  if (P.rInner > 0.0 && (groundN.y < 0.82 || (P.rInner > 100.0 && mtn > 15.0))) {
    return;
  }
  var alpine = 0.0;
  if (mtn > 1.0) {
    alpine = smoothstep(60.0, 150.0, mtn);
    let steep = 1.0 - smoothstep(0.62, 0.82, groundN.y);
    let snow = smoothstep(235.0, 280.0, mtn + simplex2d(xz * 0.004) * 40.0) * smoothstep(0.45, 0.75, groundN.y);
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

  // The petal stream parts the grass as it passes.
  let toPlayer = distance(xz, G.playerPos.xz);
  if (toPlayer < 40.0) {
    var push = vec2f(0.0);
    for (var i = 0u; i < TRAIL_LEN; i = i + 1u) {
      let tp = G.trail[i];
      if (tp.w <= 0.001) {
        continue;
      }
      let d = xz - tp.xz;
      let d2 = dot(d, d);
      let radius = 2.6 + G.gust * 1.5;
      let fall = exp(-d2 / (radius * radius));
      let above = tp.y - y;
      let vert = 1.0 - smoothstep(1.2, 5.5, above);
      let dirAway = d / max(sqrt(d2), 0.05);
      push = push + dirAway * fall * vert * tp.w;
    }
    bend = bend + push * 1.6;
  }
  let bl = length(bend);
  if (bl > 1.6) {
    bend = bend * (1.6 / bl);
  }

  let angle = unitFloat(h.x ^ h2.y) * 6.2831853;
  var width = mix(0.035, 0.06, unitFloat(h2.x ^ h.y)) * P.widthScale / sqrt(keep);
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
