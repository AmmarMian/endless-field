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

// Valley wander + irregular, skewed meander loops (wavelength ~13x the channel width).
export fn riverCenter(x: f32) -> f32 {
  let valley = RIVER_Z + 110.0 * gnoise(vec2f(x / 700.0 + 3.1, 0.7));
  let phase = x * (6.2831853 / 230.0) + 1.9 * gnoise(vec2f(x / 420.0, 5.5));
  let amp = 46.0 + 18.0 * gnoise(vec2f(x / 520.0, 2.2));
  return valley + amp * (sin(phase) + 0.24 * sin(2.0 * phase + 0.8));
}

export fn riverHalfWidth(x: f32) -> f32 {
  return 7.5 + 2.5 * gnoise(vec2f(x / 110.0, 9.3));
}

// Water surface height: only the broad terrain swell, so the river slopes gently.
export fn riverWater(x: f32) -> f32 {
  let c = riverCenter(x);
  return terrainBroad(vec2f(x, c)) * 0.6 - 3.0;
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
  let d = distance(xz, vec2f(px, c));
  return vec4f(d, riverWater(px), riverHalfWidth(px), px);
}

export fn terrainHeight(xz: vec2f) -> f32 {
  let h = terrainBase(xz);
  let r = riverInfo(xz);
  let hw = r.z;
  if (r.x > hw * 7.0) {
    return h;
  }
  // Channel: deepest mid-stream, shelving to a shallow pebbly edge, then a gentle beach
  // and floodplain that blends back into the hills.
  let inC = clamp(r.x / hw, 0.0, 1.0);
  let bed = r.y - 0.12 - 1.7 * pow(1.0 - inC * inC, 1.4);
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
