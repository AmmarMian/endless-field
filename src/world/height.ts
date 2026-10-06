import { SUNFLOWERS, sunflowerFlat } from "./sunflower-field";
// CPU mirror of src/shaders/lib/terrain.wgsl. Keep the two in sync.

const TAU = Math.PI * 2;

export function hash2i(px: number, py: number): number {
  let x = (Math.imul(px >>> 0, 1664525) + 1013904223) >>> 0;
  let y = (Math.imul(py >>> 0, 1664525) + 1013904223) >>> 0;
  x = (x + Math.imul(y, 1664525)) >>> 0;
  y = (y + Math.imul(x, 1664525)) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  y = (y ^ (y >>> 16)) >>> 0;
  x = (x + Math.imul(y, 1664525)) >>> 0;
  y = (y + Math.imul(x, 1664525)) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  y = (y ^ (y >>> 16)) >>> 0;
  return (x ^ y) >>> 0;
}

function grad(cx: number, cy: number, fx: number, fy: number): number {
  const a = hash2i(cx, cy) * (TAU / 4294967296);
  return Math.cos(a) * fx + Math.sin(a) * fy;
}

/** World seed offsets (mirror of the SEED_X / SEED_Z overrides in terrain.wgsl). */
let seedX = 0;
let seedZ = 0;

/** Selects the landscape: seed 0 is the original world. Call before anything samples heights. */
export function setWorldSeed(seed: number): void {
  if (!seed) {
    seedX = seedZ = 0;
    return;
  }
  const h = hash2i(seed, seed * 7 + 3);
  seedX = ((h & 0xffff) / 65536) * 900 + 37.17;
  seedZ = (((h >>> 16) & 0xffff) / 65536) * 900 + 53.91;
}

/** Override constants for every pipeline that samples the landscape on the GPU. */
export function worldConstants(): Record<string, number> {
  return { SEED_X: seedX, SEED_Z: seedZ, SF_LEVEL: SUNFLOWERS.level };
}

export function gnoise(xIn: number, yIn: number): number {
  const x = xIn + seedX;
  const y = yIn + seedZ;
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
  const a = grad(ix, iy, fx, fy);
  const b = grad(ix + 1, iy, fx - 1, fy);
  const c = grad(ix, iy + 1, fx, fy - 1);
  const d = grad(ix + 1, iy + 1, fx - 1, fy - 1);
  const ab = a + (b - a) * ux;
  const cd = c + (d - c) * ux;
  return ab + (cd - ab) * uy;
}

export function terrainBroad(x: number, z: number): number {
  return gnoise(x / 700, z / 700) * 52;
}

export function terrainBase(x: number, z: number): number {
  let h = terrainBroad(x, z);
  const dome = Math.max(gnoise(x / 160 + 41, z / 160 - 17) + 0.08, 0);
  h += dome * dome * 70;
  let px = x / 210;
  let py = z / 210;
  let amp = 22;
  for (let i = 0; i < 4; i++) {
    h += gnoise(px, py) * amp;
    const nx = px * 1.6 - py * 1.2 + 17.3;
    const ny = px * 1.2 + py * 1.6 - 9.1;
    px = nx;
    py = ny;
    amp *= 0.42;
  }
  return h;
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

// ---- River (mirror of the WGSL river functions).
export const RIVER_Z = -170;

// Mountains (and the mountain source of the river) live on the mountains-dev branch; here the
// river is a lowland river along its whole course.
const MOUNTAINS = false;
export const SOURCE_X = -1e6;
const UPPER_LEN = 1500;

export function riverUpper(x: number): number {
  return 1 - smoothstep(SOURCE_X, SOURCE_X + UPPER_LEN, x);
}

export function riverCenter(x: number): number {
  const valley = RIVER_Z + 110 * gnoise(x / 700 + 3.1, 0.7);
  const phase = x * ((Math.PI * 2) / 230) + 1.9 * gnoise(x / 420, 5.5);
  const amp = (46 + 18 * gnoise(x / 520, 2.2)) * (1 - 0.85 * riverUpper(x));
  return valley + amp * (Math.sin(phase) + 0.24 * Math.sin(2 * phase + 0.8));
}

export function riverRise(x: number): number {
  // Step-pool profile: long calm pools broken by short, steep drops (cascades / falls).
  const s = x / 60;
  const stepped = (Math.floor(s) + smoothstep(0.86, 0.98, s - Math.floor(s))) * 60;
  return 140 * Math.pow(riverUpper(stepped), 2.2);
}

export function riverHalfWidth(x: number): number {
  const q = Math.min(1, Math.max(0, (x - SOURCE_X) / (UPPER_LEN * 1.6)));
  return (7.5 + 2.5 * gnoise(x / 110, 9.3)) * (0.2 + 0.8 * Math.pow(q, 0.6));
}

export function riverWater(x: number): number {
  return terrainBroad(x, riverCenter(x)) * 0.6 - 3 + riverRise(x);
}

export function riverBed(d: number, water: number, hw: number, upper: number): number {
  const inC = Math.min(1, Math.max(0, d / hw));
  return water - 0.12 - 1.7 * (1 + (0.45 - 1) * upper) * Math.pow(1 - inC * inC, 1.4);
}

function riverSlope(x: number): number {
  return (riverCenter(x + 0.75) - riverCenter(x - 0.75)) / 1.5;
}

/** [distance to centerline, water height, half width, along-river x of the closest point] */
export function riverInfo(x: number, z: number): [number, number, number, number] {
  let px = x;
  let c = riverCenter(px);
  if (Math.abs(z - c) > 420) return [1e4, riverWater(px), riverHalfWidth(px), px];
  for (let i = 0; i < 3; i++) {
    const sl = riverSlope(px);
    const t = (x - px + (z - c) * sl) / (1 + sl * sl);
    px += Math.max(-60, Math.min(60, t));
    c = riverCenter(px);
  }
  px = Math.max(px, SOURCE_X);
  c = riverCenter(px);
  return [Math.hypot(x - px, z - c), riverWater(px), riverHalfWidth(px), px];
}

export function terrainHeight(x: number, z: number): number {
  const [d, water, hw, px] = riverInfo(x, z);
  const flat = sunflowerFlat(x, z);
  const base = terrainBase(x, z);
  const h = base + (SUNFLOWERS.level - base) * flat + riverRise(px) * (1 - smoothstep(hw * 2, 700, d));
  if (d > hw * 7) return h;
  const bed = riverBed(d, water, hw, riverUpper(px));
  const beach = water - 0.12 + smoothstep(hw * 0.95, hw * 1.9, d) * 0.7 + Math.max(d - hw * 1.9, 0) * 0.12;
  const profile = d < hw ? bed : beach;
  const w = 1 - smoothstep(hw * 2, hw * 7, d);
  return h + (profile - h) * w;
}

export function terrainNormal(x: number, z: number, e = 0.75): [number, number, number] {
  const hx = terrainHeight(x + e, z) - terrainHeight(x - e, z);
  const hz = terrainHeight(x, z + e) - terrainHeight(x, z - e);
  const nx = -hx;
  const ny = 2 * e;
  const nz = -hz;
  const l = Math.hypot(nx, ny, nz);
  return [nx / l, ny / l, nz / l];
}

// ---- Mountains (mirror of the WGSL mountain functions; data from public/assets/mountains).
export const MTN_CELL = 1700;
export const MTN_EXTENT = 2400;
let mtnData: Float32Array[] = [];
let mtnSize = 1;

export function setMountains(layers: Float32Array[], size: number): void {
  mtnData = layers;
  mtnSize = size;
}

export function mountainZone(x: number, z: number): number {
  if (!MOUNTAINS) return 0;
  const dist = Math.hypot(x, z);
  if (dist < 600) return 0;
  const west = 1 - smoothstep(-1500, -800, x);
  const ranges = smoothstep(-0.35, 0.1, gnoise(x / 3200 + 11, z / 3200 - 7));
  let zone = west * ranges;
  const up = riverUpper(x);
  if (up > 0) zone = Math.max(zone, up * (1 - smoothstep(250, 900, Math.abs(z - riverCenter(x)))) * smoothstep(600, 1000, dist));
  return zone;
}

/** Bilinear sample matching a linear clamp-to-edge GPU sampler (texel centers at +0.5). */
function sampleLayer(layer: number, u: number, v: number): number {
  const d = mtnData[layer];
  if (!d) return 0;
  const n = mtnSize;
  const x = Math.min(n - 1, Math.max(0, u * n - 0.5));
  const y = Math.min(n - 1, Math.max(0, v * n - 0.5));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(n - 1, x0 + 1);
  const y1 = Math.min(n - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const a = d[y0 * n + x0] + (d[y0 * n + x1] - d[y0 * n + x0]) * fx;
  const b = d[y1 * n + x0] + (d[y1 * n + x1] - d[y1 * n + x0]) * fx;
  return a + (b - a) * fy;
}

export function mountainHeight(x: number, z: number): number {
  if (!MOUNTAINS || Math.hypot(x, z) < 250) return 0;
  const gx = x / MTN_CELL - 0.5;
  const gz = z / MTN_CELL - 0.5;
  const bx = Math.floor(gx);
  const bz = Math.floor(gz);
  let h = 0;
  for (let k = 0; k < 4; k++) {
    const cx = bx + (k & 1);
    const cz = bz + (k >> 1);
    const hs = hash2i(cx - 911, cz + 4517);
    const ox = ((hs & 0xff) / 255 - 0.5) * 0.24;
    const oz = (((hs >>> 8) & 0xff) / 255 - 0.5) * 0.24;
    const centerX = (cx + 0.5 + ox) * MTN_CELL;
    const centerZ = (cz + 0.5 + oz) * MTN_CELL;
    if (Math.abs(x - centerX) >= MTN_EXTENT / 2 || Math.abs(z - centerZ) >= MTN_EXTENT / 2) continue;
    const zone = mountainZone(centerX, centerZ);
    if (zone < 0.05) continue;
    let u = (x - centerX) / MTN_EXTENT;
    let v = (z - centerZ) / MTN_EXTENT;
    if (Math.abs(u) >= 0.5 || Math.abs(v) >= 0.5) continue;
    const rot = (hs >>> 16) & 3;
    if (rot === 1) [u, v] = [-v, u];
    else if (rot === 2) [u, v] = [-u, -v];
    else if (rot === 3) [u, v] = [v, -u];
    const layer = ((hs >>> 18) >>> 0) % 3;
    const scale = 200 + (((hs >>> 20) & 0xff) / 255) * 160;
    h = Math.max(h, sampleLayer(layer, u + 0.5, v + 0.5) * scale * zone);
  }
  const sx = SOURCE_X - 160;
  const sz = riverCenter(SOURCE_X);
  const su = (x - sx) / MTN_EXTENT;
  const sv = (z - sz) / MTN_EXTENT;
  if (Math.abs(su) < 0.5 && Math.abs(sv) < 0.5) h = Math.max(h, sampleLayer(0, su + 0.5, sv + 0.5) * 340);
  return h;
}

function riverValley(natural: number, d: number, water: number, hw: number, px: number, x: number, z: number): number {
  const upper = riverUpper(px);
  const uw = smoothstep(0, 0.1, upper);
  if (uw <= 0) return natural;
  const bed = riverBed(d, water, hw, upper);
  const wallN = gnoise(x / 45 + 3.3, z / 45 + 8.1);
  const bumps = gnoise(x / 9, z / 9) * 1.6 * smoothstep(hw * 1.5, hw * 4, d);
  const cut = d < hw ? bed : water + 0.3 + Math.max(d - hw * 1.15, 0) * (0.5 + 0.25 * wallN) + bumps;
  const fill = d < hw ? bed : water + 0.3 - Math.max(d - hw * 1.6, 0) * 0.28;
  const clamped = Math.min(Math.max(natural, fill), Math.max(cut, fill));
  return natural + (clamped - natural) * uw;
}

/** Terrain including mountains (what gameplay should use everywhere). */
export function terrainHeightM(x: number, z: number): number {
  const [d, water, hw, px] = riverInfo(x, z);
  const corridor = smoothstep(hw * 4, hw * 14, d) + (1 - smoothstep(hw * 4, hw * 14, d)) * smoothstep(0, 0.1, riverUpper(px));
  return riverValley(terrainHeight(x, z) + mountainHeight(x, z) * corridor, d, water, hw, px, x, z);
}
