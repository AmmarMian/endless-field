// CPU mirror of src/shaders/lib/biome.wgsl. Keep the two in sync.
import { gnoise } from "./height";

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Returns [grove, meadow, forest] amounts in [0, 1]; all 0 means rolling savanna. */
export function biome(x: number, z: number): [number, number, number] {
  const wx = gnoise(x * 0.0011 + 3.7, z * 0.0011 + 1.3) * 380;
  const wz = gnoise(x * 0.0011 - 8.1, z * 0.0011 + 4.4) * 380;
  const m = gnoise((x + wx) / 1150 + 21, (z + wz) / 1150 - 13);
  const east = smoothstep(380, 900, x);
  const forest = Math.max(smoothstep(0.36, 0.48, m), east);
  return [smoothstep(0.1, 0.3, m) * (1 - east), smoothstep(0.08, 0.28, -m) * (1 - east), forest];
}
