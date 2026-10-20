// CPU mirror of src/shaders/lib/swell.wgsl (keep the two in sync).
import { SEA_Y, coastInfo, terrainHeightM } from "./height";

const L = [110, 71, 43, 29];
const A = [0.62, 0.38, 0.2, 0.11];
const H = [0.12, -0.28, 0.45, -0.6];
const P = [0, 1.7, 4.1, 2.6];

/** The swell's height above SEA_Y at (x, z), time t, scaled by k. */
export function swellHeight(x: number, z: number, t: number, k = 1): number {
  let y = 0;
  for (let i = 0; i < 4; i++) {
    const dx = Math.sin(H[i]);
    const dz = -Math.cos(H[i]);
    const kw = (Math.PI * 2) / L[i];
    const w = Math.sqrt(9.81 * kw);
    y += A[i] * k * Math.sin(kw * (dx * x + dz * z) - w * t + P[i]);
  }
  return y;
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** How much swell rides at a point (it dies in the shallows and up the estuary): as sea.wgsl. */
export function swellAmount(x: number, z: number): number {
  const [s, open] = coastInfo(x, z);
  if (s > 0) return 0;
  const still = SEA_Y - (s > -220 ? terrainHeightM(x, z) : SEA_Y - 30);
  return smoothstep(0.5, 2.8, still) * (0.15 + 0.85 * open);
}

/** The sea's surface height at (x, z), or null over land. */
export function seaSurface(x: number, z: number, t: number): number | null {
  const [s] = coastInfo(x, z);
  if (s > 0) return null;
  return SEA_Y + swellHeight(x, z, t, swellAmount(x, z));
}
