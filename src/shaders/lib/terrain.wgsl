// Terrain height field. Mirrored by src/world/height.ts (up to float precision), so gameplay
// on the CPU and rendering on the GPU agree on where the ground is.

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
export fn gnoise(p: vec2f) -> f32 {
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

// The river springs from a mountain massif to the west and runs endlessly east.
export const SOURCE_X: f32 = -1900.0;
const UPPER_LEN: f32 = 1500.0;

// 1 at the spring, 0 once the river has come down to the lowland.
export fn riverUpper(x: f32) -> f32 {
  return 1.0 - smoothstep(SOURCE_X, SOURCE_X + UPPER_LEN, x);
}

// Valley wander + irregular, skewed meander loops (wavelength ~13x the channel width).
export fn riverCenter(x: f32) -> f32 {
  let valley = RIVER_Z + 110.0 * gnoise(vec2f(x / 700.0 + 3.1, 0.7));
  let phase = x * (6.2831853 / 230.0) + 1.9 * gnoise(vec2f(x / 420.0, 5.5));
  // Sinuosity follows slope: steep mountain reaches run nearly straight, big meanders form
  // only on the flat floodplain.
  let amp = (46.0 + 18.0 * gnoise(vec2f(x / 520.0, 2.2))) * (1.0 - 0.85 * riverUpper(x));
  return valley + amp * (sin(phase) + 0.24 * sin(2.0 * phase + 0.8));
}

// Height gained toward the source, as a staircase of pools and short cascades.
export fn riverRise(x: f32) -> f32 {
  let s = x / 95.0;
  let stepped = (floor(s) + smoothstep(0.72, 0.97, fract(s))) * 95.0;
  return 175.0 * pow(riverUpper(stepped), 2.2);
}

// Width follows discharge, which accumulates downstream of the spring (w ~ Q^0.5).
export fn riverHalfWidth(x: f32) -> f32 {
  let q = clamp((x - SOURCE_X) / (UPPER_LEN * 1.6), 0.0, 1.0);
  return (7.5 + 2.5 * gnoise(vec2f(x / 110.0, 9.3))) * mix(0.28, 1.0, sqrt(q));
}

// Manning's equation, v = (1/n) R^(2/3) S^(1/2): surface speed from the local water slope
// and depth, with a floor for the near-flat lowland reaches.
export fn riverSpeed(x: f32, depth: f32) -> f32 {
  let slope = abs(riverWater(x - 4.0) - riverWater(x + 4.0)) / 8.0;
  let r = clamp(depth, 0.05, 2.0);
  return clamp((1.0 / 0.035) * pow(r, 0.6667) * sqrt(max(slope, 0.0004)), 0.35, 7.0);
}

// Water surface: the broad terrain swell in the lowland, climbing into the mountains upstream.
export fn riverWater(x: f32) -> f32 {
  let c = riverCenter(x);
  return terrainBroad(vec2f(x, c)) * 0.6 - 3.0 + riverRise(x);
}

// Channel bed: deepest mid-stream, shallower in the mountain stream.
export fn riverBed(d: f32, water: f32, hw: f32, upper: f32) -> f32 {
  let inC = clamp(d / hw, 0.0, 1.0);
  return water - 0.12 - 1.7 * mix(1.0, 0.45, upper) * pow(1.0 - inC * inC, 1.4);
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
  // Nothing upstream of the spring: distance is measured to the source point there.
  px = max(px, SOURCE_X);
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
  let h = terrainBase(xz);
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
  let dist = length(xz);
  if (dist < 600.0) {
    return 0.0;
  }
  let spawn = smoothstep(800.0, 1400.0, dist);
  let ranges = smoothstep(-0.12, 0.22, gnoise(xz / 3200.0 + vec2f(11.0, -7.0)));
  var zone = spawn * ranges;
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
  if (length(xz) < 250.0) {
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
    let scale = 260.0 + f32((hs >> 20u) & 0xFFu) / 255.0 * 220.0;
    let v = textureSampleLevel(tex, samp, uv + 0.5, layer, lod).r;
    h = max(h, v * scale * zone);
  }
  // The source massif, always present at the head of the river.
  let src = vec2f(SOURCE_X - 160.0, riverCenter(SOURCE_X));
  let suv = (xz - src) / MTN_EXTENT;
  if (abs(suv.x) < 0.5 && abs(suv.y) < 0.5) {
    h = max(h, textureSampleLevel(tex, samp, suv + 0.5, 0, lod).r * 450.0);
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
  let cut = select(r.y + 0.3 + max(r.x - r.z * 1.15, 0.0) * (0.85 + 0.45 * wallN) + bumps, bed, r.x < r.z);
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
