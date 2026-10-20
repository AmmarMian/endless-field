// Terrain height field. Mirrored by src/world/height.ts (up to float precision), so gameplay
// on the CPU and rendering on the GPU agree on where the ground is.
import { sunflowerLevel } from "./sunflowers.wgsl";

export fn hash2i(p: vec2i) -> u32 {
  var v = bitcast<vec2u>(p) * 1664525u + 1013904223u;
  v.x = v.x + v.y * 1664525u;
  v.y = v.y + v.x * 1664525u;
  v = v ^ (v >> vec2u(16u));
  v.x = v.x + v.y * 1664525u;
  v.y = v.y + v.x * 1664525u;
  v = v ^ (v >> vec2u(16u));
  return v.x ^ v.y;
}

fn grad(c: vec2i, f: vec2f) -> f32 {
  let a = f32(hash2i(c)) * (6.28318530718 / 4294967296.0);
  return dot(vec2f(cos(a), sin(a)), f);
}

// Gradient noise in roughly [-0.7, 0.7].
// World seed: every landscape noise is read at an offset (0, 0 is the original world).
override SEED_X: f32 = 0.0;
override SEED_Z: f32 = 0.0;

export fn gnoise(pIn: vec2f) -> f32 {
  let p = pIn + vec2f(SEED_X, SEED_Z);
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = grad(i, f);
  let b = grad(i + vec2i(1, 0), f - vec2f(1.0, 0.0));
  let c = grad(i + vec2i(0, 1), f - vec2f(0.0, 1.0));
  let d = grad(i + vec2i(1, 1), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

export fn terrainBroad(xz: vec2f) -> f32 {
  return gnoise(xz * (1.0 / 700.0)) * 52.0;
}

fn terrainBase(xz: vec2f) -> f32 {
  // Broad swells shape the land; rounded domes and rolling octaves ride on top.
  var h = terrainBroad(xz);
  let dome = max(gnoise(xz * (1.0 / 160.0) + vec2f(41.0, -17.0)) + 0.08, 0.0);
  h = h + dome * dome * 70.0;
  var p = xz * (1.0 / 210.0);
  var amp = 22.0;
  for (var i = 0; i < 4; i = i + 1) {
    h = h + gnoise(p) * amp;
    // Rotate each octave so lattice artifacts never line up.
    p = vec2f(p.x * 1.6 - p.y * 1.2, p.x * 1.2 + p.y * 1.6) + vec2f(17.3, -9.1);
    amp = amp * 0.42;
  }
  return h;
}

// ---- River: one endless meandering river running roughly along +x.
export const RIVER_Z: f32 = -170.0;
// The river valley's offset for this world (set by the host from the seed).
override RIVER_DZ: f32 = 0.0;

// The river springs from a mountain massif to the west and runs endlessly east.
// Mountains (and the mountain source of the river) live on the mountains-dev branch; here the
// river is a lowland river along its whole course.
const MOUNTAINS = false;
export const SOURCE_X: f32 = -1.0e6;
const UPPER_LEN: f32 = 1500.0;

// 1 at the spring, 0 once the river has come down to the lowland.
export fn riverUpper(x: f32) -> f32 {
  return 1.0 - smoothstep(SOURCE_X, SOURCE_X + UPPER_LEN, x);
}

// Valley wander + irregular, skewed meander loops (wavelength ~13x the channel width).
export fn riverCenter(x: f32) -> f32 {
  let valley = RIVER_Z + RIVER_DZ + 110.0 * gnoise(vec2f(x / 700.0 + 3.1, 0.7));
  let phase = x * (6.2831853 / 230.0) + 1.9 * gnoise(vec2f(x / 420.0, 5.5));
  // Sinuosity follows slope: steep mountain reaches run nearly straight, big meanders form
  // only on the flat floodplain.
  let amp = (46.0 + 18.0 * gnoise(vec2f(x / 520.0, 2.2))) * (1.0 - 0.85 * riverUpper(x));
  return valley + amp * (sin(phase) + 0.24 * sin(2.0 * phase + 0.8));
}

// Height gained toward the source, as a staircase of pools and short cascades.
export fn riverRise(x: f32) -> f32 {
  // Step-pool profile: long calm pools broken by short, steep drops (cascades / falls).
  let s = x / 60.0;
  let stepped = (floor(s) + smoothstep(0.86, 0.98, fract(s))) * 60.0;
  return 140.0 * pow(riverUpper(stepped), 2.2);
}

// Width follows discharge, which accumulates downstream of the spring (w ~ Q^0.5).
export fn riverHalfWidth(x: f32) -> f32 {
  let q = clamp((x - SOURCE_X) / (UPPER_LEN * 1.6), 0.0, 1.0);
  return (7.5 + 2.5 * gnoise(vec2f(x / 110.0, 9.3))) * mix(0.2, 1.0, pow(q, 0.6));
}

// Manning's equation, v = (1/n) R^(2/3) S^(1/2): surface speed from the local water slope
// and depth, with a floor for the near-flat lowland reaches.
export fn riverSpeed(x: f32, depth: f32) -> f32 {
  let slope = abs(riverWater(x - 4.0) - riverWater(x + 4.0)) / 8.0;
  let r = clamp(depth, 0.05, 2.0);
  return clamp((1.0 / 0.035) * pow(r, 0.6667) * sqrt(max(slope, 0.0004)), 0.35, 7.0);
}

// Water surface: the broad terrain swell in the lowland, climbing into the mountains upstream,
// and settling to sea level over the last reach before the estuary.
export fn riverWater(x: f32) -> f32 {
  let c = riverCenter(x);
  return mix(SEA_Y, terrainBroad(vec2f(x, c)) * 0.6 - 3.0 + riverRise(x), smoothstep(MOUTH_X + 30.0, MOUTH_X + 600.0, x));
}

// Channel bed: deepest mid-stream, shallower in the mountain stream.
export fn riverBed(d: f32, water: f32, hw: f32, upper: f32) -> f32 {
  let inC = clamp(d / hw, 0.0, 1.0);
  return water - 0.12 - 3.2 * mix(1.0, 0.45, upper) * pow(1.0 - inC * inC, 1.4);
}

fn riverSlope(x: f32) -> f32 {
  return (riverCenter(x + 0.75) - riverCenter(x - 0.75)) / 1.5;
}

// x = distance to the centerline (closest point by Gauss-Newton on the curve), y = water
// height at that point, z = half width there, w = along-river coordinate of that point.
export fn riverInfo(xz: vec2f) -> vec4f {
  var px = xz.x;
  var c = riverCenter(px);
  // Cheap reject: the curve is never steeper than ~3, so far points skip the projection.
  if (abs(xz.y - c) > 420.0) {
    return vec4f(1e4, riverWater(px), riverHalfWidth(px), px);
  }
  for (var i = 0; i < 3; i = i + 1) {
    let sl = riverSlope(px);
    let t = ((xz.x - px) + (xz.y - c) * sl) / (1.0 + sl * sl);
    px = px + clamp(t, -60.0, 60.0);
    c = riverCenter(px);
  }
  // Nothing upstream of the spring or past the mouth: distance is measured to the end point.
  px = max(px, max(SOURCE_X, MOUTH_X));
  c = riverCenter(px);
  let d = distance(xz, vec2f(px, c));
  return vec4f(d, riverWater(px), riverHalfWidth(px), px);
}

// River info for a point near one whose closest river coordinate `px` is already known:
// only the distance is recomputed (used for normals and neighbor samples).
export fn riverInfoAt(xz: vec2f, px: f32) -> vec4f {
  return vec4f(distance(xz, vec2f(px, riverCenter(px))), riverWater(px), riverHalfWidth(px), px);
}

export fn terrainHeight(xz: vec2f) -> f32 {
  return terrainHeightR(xz, riverInfo(xz));
}

export fn terrainHeightR(xz: vec2f, r: vec4f) -> f32 {
  return coastal(xz, r, inlandHeight(xz, r));
}

fn inlandHeight(xz: vec2f, r: vec4f) -> f32 {
  // The upper valley climbs with the river: the land is raised by the river's rise (fading
  // over ~700 m) before the banks are shaped, so cross-sections stay monotonic.
  let h = sunflowerLevel(xz, terrainBase(xz)) + riverRise(r.w) * (1.0 - smoothstep(r.z * 2.0, 700.0, r.x));
  let hw = r.z;
  if (r.x > hw * 7.0) {
    return h;
  }
  // Channel: deepest mid-stream, shelving to a shallow pebbly edge, then a gentle beach
  // and floodplain that blends back into the hills.
  let bed = riverBed(r.x, r.y, hw, riverUpper(r.w));
  let beach = r.y - 0.12 + smoothstep(hw * 0.95, hw * 1.9, r.x) * 0.7 + max(r.x - hw * 1.9, 0.0) * 0.12;
  let profile = select(beach, bed, r.x < hw);
  let w = 1.0 - smoothstep(hw * 2.0, hw * 7.0, r.x);
  return mix(h, profile, w);
}

// ---- Sea: the land ends in a southern coast (beach, then dunes), and the river reaches the
// sea through an estuary. The river flows west and ends at MOUTH_X, in the estuary's head.
export const SEA_Y: f32 = -6.0;
export const MOUTH_X: f32 = -760.0;
// riverCenter(MOUTH_X), set by the host (it moves with the river from the world seed).
override MOUTH_Z: f32 = -165.0;

// The open coastline, z as a function of x (south is +z): trending north to the west.
export fn coastLine(x: f32) -> f32 {
  return 470.0 + 0.16 * clamp(x, -1500.0, 2500.0) + 55.0 * gnoise(vec2f(x / 600.0, 13.7)) + 16.0 * gnoise(vec2f(x / 150.0, -4.2));
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

// x = signed distance inland from the waterline (m; negative at sea), y = how exposed the
// water is to the open sea (1 along the open coast .. 0 up the estuary: waves die there).
export fn coastInfo(xz: vec2f) -> vec2f {
  let zc = coastLine(xz.x);
  let inEstuary = abs(xz.x - MOUTH_X + 60.0) < 420.0 && xz.y > MOUTH_Z - 300.0;
  // Far inland (most of the world) nothing below matters.
  if (zc - xz.y > 450.0 && !inEstuary) {
    return vec2f(zc - xz.y, 0.0);
  }
  // Far out at sea, likewise.
  if (xz.y - zc > 450.0) {
    return vec2f(zc - xz.y, 1.0);
  }
  let slope = (coastLine(xz.x + 2.0) - coastLine(xz.x - 2.0)) * 0.25;
  let open = (zc - xz.y) / sqrt(1.0 + slope * slope);
  var est = 1e4;
  if (inEstuary) {
    // The estuary: a channel from the river's end curving out to the coast, widening seaward.
    let zEnd = coastLine(MOUTH_X - 50.0) + 60.0;
    let len = zEnd - MOUTH_Z;
    let t = clamp((xz.y - MOUTH_Z) / len, 0.0, 1.0);
    let cx = MOUTH_X - 75.0 * sin(t * 2.4);
    let dcx = -75.0 * 2.4 * cos(t * 2.4) / len;
    let w = mix(10.0, 140.0, pow(t, 1.6)) + 9.0 * gnoise(xz * 0.018 + vec2f(5.0, 1.0)) * t;
    est = length(vec2f((xz.x - cx) / sqrt(1.0 + dcx * dcx), max(MOUTH_Z - xz.y, 0.0))) - w;
  }
  return vec2f(smin(open, est, 45.0), 1.0 - smoothstep(0.0, 160.0, open));
}

// How much of a stretch of coast is dunes (the rest is low beach and rocky ground).
export fn duneAmount(xz: vec2f) -> f32 {
  return smoothstep(-0.15, 0.25, gnoise(xz / 420.0 + vec2f(-3.3, 8.8)));
}

// Ground across the shore at signed distance `s`: seabed shelving out with a sandbar, the
// beach, then a band of dunes.
export fn coastProfile(s: f32, xz: vec2f, open: f32) -> f32 {
  if (s < 0.0) {
    let d = -s;
    let b = (d - 55.0) / 14.0;
    let bar = 0.5 * exp(-b * b) * open;
    return SEA_Y - (2.0 * (1.0 - exp(-d / 40.0)) + d * 0.012) + bar;
  }
  var h = SEA_Y + min(s, 60.0) * 0.045 + max(s - 60.0, 0.0) * 0.015;
  // Dunes: a ridge behind the beach broken into hummocks, with blowouts between.
  let ridge = smoothstep(30.0, 65.0, s) * (1.0 - smoothstep(105.0, 200.0, s));
  let hum = gnoise(xz / 17.0 + vec2f(1.7, -4.4)) * 0.5 + 0.5;
  let big = gnoise(xz / 55.0 + vec2f(-6.1, 2.3)) * 0.5 + 0.5;
  h = h + ridge * (1.2 + 3.2 * big + 4.5 * hum * hum) * duneAmount(xz);
  return h;
}

// Shapes the ground toward the coast. The river's last reach keeps its channel (its water is
// at sea level there) and carves on into the estuary's bed.
export fn coastal(xz: vec2f, r: vec4f, h: f32) -> f32 {
  let c = coastInfo(xz);
  if (c.x > 330.0) {
    return h;
  }
  let p = coastProfile(c.x, xz, c.y);
  let shaped = mix(h, p, select(1.0 - smoothstep(90.0, 330.0, c.x), 1.0, c.x < 0.0));
  let channel = min(h, p + max(c.x, 0.0) * 0.2);
  return mix(channel, shaped, smoothstep(r.z * 1.5, r.z * 5.0, r.x));
}

export fn terrainNormal(xz: vec2f, e: f32) -> vec3f {
  let hx = terrainHeight(xz + vec2f(e, 0.0)) - terrainHeight(xz - vec2f(e, 0.0));
  let hz = terrainHeight(xz + vec2f(0.0, e)) - terrainHeight(xz - vec2f(0.0, e));
  return normalize(vec3f(-hx, 2.0 * e, -hz));
}

// Cheaper normal when the height at `xz` is already known (forward differences).
export fn terrainNormalFrom(xz: vec2f, h: f32, e: f32) -> vec3f {
  let hx = terrainHeight(xz + vec2f(e, 0.0)) - h;
  let hz = terrainHeight(xz + vec2f(0.0, e)) - h;
  return normalize(vec3f(-hx, e, -hz));
}

// ---- Mountains: eroded massifs authored in Blender (tools/blender/build_mountains.py),
// stamped onto a cell grid far from spawn. Heights come from a 2D-array texture passed in by
// the entry shader (pure modules cannot declare bindings). Mirrored by src/world/height.ts.
export const MTN_CELL: f32 = 1700.0;
export const MTN_EXTENT: f32 = 2400.0;

fn mtnHash(c: vec2i) -> u32 {
  return hash2i(c + vec2i(-911, 4517));
}

// Range mask: zero near spawn, and broken into ranges by very low-frequency noise.
export fn mountainZone(xz: vec2f) -> f32 {
  if (!MOUNTAINS) {
    return 0.0;
  }
  let dist = length(xz);
  if (dist < 600.0) {
    return 0.0;
  }
  // Mountains rise only to the west (where the river springs), broken into ranges.
  let west = 1.0 - smoothstep(-1500.0, -800.0, xz.x);
  let ranges = smoothstep(-0.35, 0.1, gnoise(xz / 3200.0 + vec2f(11.0, -7.0)));
  var zone = west * ranges;
  // The river's upper valley always runs between mountains.
  let up = riverUpper(xz.x);
  if (up > 0.0) {
    let upperValley = up * (1.0 - smoothstep(250.0, 900.0, abs(xz.y - riverCenter(xz.x))));
    zone = max(zone, upperValley * smoothstep(600.0, 1000.0, dist));
  }
  return zone;
}

export fn mountainHeight(xz: vec2f, tex: texture_2d_array<f32>, samp: sampler, lod: f32) -> f32 {
  // No massif reaches within ~250 m of spawn (zones start at 600 m, stamps span 1200 m).
  if (!MOUNTAINS || length(xz) < 250.0) {
    return 0.0;
  }
  let g = xz / MTN_CELL - 0.5;
  let base = vec2i(floor(g));
  var h = 0.0;
  for (var k = 0; k < 4; k = k + 1) {
    let c = base + vec2i(k & 1, k >> 1);
    let hs = mtnHash(c);
    let center = (vec2f(c) + 0.5 + (vec2f(f32(hs & 0xFFu), f32((hs >> 8u) & 0xFFu)) / 255.0 - 0.5) * 0.24) * MTN_CELL;
    if (any(abs(xz - center) >= vec2f(MTN_EXTENT * 0.5))) {
      continue;
    }
    let zone = mountainZone(center);
    if (zone < 0.05) {
      continue;
    }
    var uv = (xz - center) / MTN_EXTENT;
    if (abs(uv.x) >= 0.5 || abs(uv.y) >= 0.5) {
      continue;
    }
    let rot = (hs >> 16u) & 3u;
    if (rot == 1u) { uv = vec2f(-uv.y, uv.x); }
    if (rot == 2u) { uv = -uv; }
    if (rot == 3u) { uv = vec2f(uv.y, -uv.x); }
    let layer = i32((hs >> 18u) % 3u);
    let scale = 200.0 + f32((hs >> 20u) & 0xFFu) / 255.0 * 160.0;
    let v = textureSampleLevel(tex, samp, uv + 0.5, layer, lod).r;
    h = max(h, v * scale * zone);
  }
  // The source massif, always present at the head of the river.
  let src = vec2f(SOURCE_X - 160.0, riverCenter(SOURCE_X));
  let suv = (xz - src) / MTN_EXTENT;
  if (abs(suv.x) < 0.5 && abs(suv.y) < 0.5) {
    h = max(h, textureSampleLevel(tex, samp, suv + 0.5, 0, lod).r * 340.0);
  }
  return h;
}

// Full terrain including mountains; mountains fade out along the river corridor.
// In the mountains the river cuts a V-shaped valley: ground is clamped below a "cut" profile
// rising steeply from the banks and above a "fill" that keeps the stream from floating.
export fn riverValley(natural: f32, r: vec4f, xz: vec2f) -> f32 {
  let upper = riverUpper(r.w);
  let uw = smoothstep(0.0, 0.1, upper);
  if (uw <= 0.0) {
    return natural;
  }
  let bed = riverBed(r.x, r.y, r.z, upper);
  // Valley walls: slope and small bumps vary along the valley so they read as hillsides.
  let wallN = gnoise(xz / 45.0 + vec2f(3.3, 8.1));
  let bumps = gnoise(xz / 9.0) * 1.6 * smoothstep(r.z * 1.5, r.z * 4.0, r.x);
  let cut = select(r.y + 0.3 + max(r.x - r.z * 1.15, 0.0) * (0.5 + 0.25 * wallN) + bumps, bed, r.x < r.z);
  let fill = select(r.y + 0.3 - max(r.x - r.z * 1.6, 0.0) * 0.28, bed, r.x < r.z);
  return mix(natural, clamp(natural, fill, max(cut, fill)), uw);
}

export fn mountainCorridor(r: vec4f) -> f32 {
  return mix(smoothstep(r.z * 4.0, r.z * 14.0, r.x), 1.0, smoothstep(0.0, 0.1, riverUpper(r.w)));
}

export fn terrainHeightMR(xz: vec2f, r: vec4f, tex: texture_2d_array<f32>, samp: sampler) -> f32 {
  return riverValley(terrainHeightR(xz, r) + mountainHeight(xz, tex, samp, 0.0) * mountainCorridor(r), r, xz);
}

export fn terrainHeightM(xz: vec2f, tex: texture_2d_array<f32>, samp: sampler) -> f32 {
  return terrainHeightMR(xz, riverInfo(xz), tex, samp);
}

// Normal from forward differences, reusing the river projection of the center point.
export fn terrainNormalM(xz: vec2f, h: f32, e: f32, px: f32, tex: texture_2d_array<f32>, samp: sampler) -> vec3f {
  let ax = xz + vec2f(e, 0.0);
  let az = xz + vec2f(0.0, e);
  let hx = terrainHeightMR(ax, riverInfoAt(ax, px), tex, samp) - h;
  let hz = terrainHeightMR(az, riverInfoAt(az, px), tex, samp) - h;
  return normalize(vec3f(-hx, e, -hz));
}
