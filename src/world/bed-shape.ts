// CPU mirror of src/shaders/lib/beds.wgsl. Keep the two in sync.
import { gnoise, hash2i, mountainZone } from "./height";
import { biome } from "./biome";

export const BED_CELL = 64;
export const BED_SPECIES = 5;

export interface BedInfo {
  cx: number;
  cz: number;
  x: number;
  z: number;
  radius: number;
  species: number;
}

export function bedInCell(cx: number, cz: number): BedInfo | null {
  const h = hash2i(cx + 7919, cz - 104729);
  const h2 = hash2i(cx - 31337, cz + 2203);
  const [grove, meadow, forest] = biome((cx + 0.5) * BED_CELL, (cz + 0.5) * BED_CELL);
  if ((h & 0xffff) / 65536 >= 0.3 * (1 + meadow * 0.9 - grove * 0.6 - forest * 0.4)) return null;
  if (mountainZone((cx + 0.5) * BED_CELL, (cz + 0.5) * BED_CELL) > 0.05) return null;
  const u = ((h >>> 16) & 0xff) / 255;
  const v = ((h >>> 24) & 0xff) / 255;
  return {
    cx,
    cz,
    x: cx * BED_CELL + 14 + u * (BED_CELL - 28),
    z: cz * BED_CELL + 14 + v * (BED_CELL - 28),
    radius: 6 + ((h2 & 0xff) / 255) * 4,
    species: (h2 >>> 8) % BED_SPECIES,
  };
}

/** Inside amount in [0, 1] for a point within this bed's cell. */
export function bedInside(b: BedInfo, x: number, z: number): number {
  const wobble = 1 + 0.28 * gnoise(x * 0.12 + b.cx * 3.1, z * 0.12 + b.cz * 3.1);
  const d = Math.hypot(x - b.x, z - b.z) / (b.radius * wobble);
  const t = Math.min(1, Math.max(0, (d - 0.45) / 0.55));
  return 1 - t * t * (3 - 2 * t);
}
