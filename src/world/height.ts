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

export function gnoise(x: number, y: number): number {
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

function terrainBase(x: number, z: number): number {
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

// ---- River (mirror of the WGSL river functions).
export const RIVER_Z = -170;

export function riverCenter(x: number): number {
  const valley = RIVER_Z + 110 * gnoise(x / 700 + 3.1, 0.7);
  const phase = x * ((Math.PI * 2) / 230) + 1.9 * gnoise(x / 420, 5.5);
  const amp = 46 + 18 * gnoise(x / 520, 2.2);
  return valley + amp * (Math.sin(phase) + 0.24 * Math.sin(2 * phase + 0.8));
}

export function riverHalfWidth(x: number): number {
  return 7.5 + 2.5 * gnoise(x / 110, 9.3);
}

export function riverWater(x: number): number {
  return terrainBroad(x, riverCenter(x)) * 0.6 - 3;
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
  return [Math.hypot(x - px, z - c), riverWater(px), riverHalfWidth(px), px];
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

export function terrainHeight(x: number, z: number): number {
  const h = terrainBase(x, z);
  const [d, water, hw] = riverInfo(x, z);
  if (d > hw * 7) return h;
  const inC = Math.min(1, d / hw);
  const bed = water - 0.12 - 1.7 * Math.pow(1 - inC * inC, 1.4);
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
