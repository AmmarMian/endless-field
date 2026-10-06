import { biome } from "./biome";
import { riverInfo, terrainBroad, terrainHeightM } from "./height";

/** Environmental conditions at a point, the inputs of every species' niche. */
export interface Eco {
  height: number;
  /** Meters above the surrounding plain (drives temperature). */
  elevation: number;
  /** 1 warm lowland .. 0 alpine cold. */
  temperature: number;
  /** 0 dry .. 1 wet. */
  moisture: number;
  /** 0 flat .. 1 vertical. */
  slope: number;
  /** -1 shaded (facing away from the sun) .. 1 sun-facing. */
  exposure: number;
  /** The eastern forest realm. */
  forest: number;
}

const SUN_XZ = [0.36, -0.93]; // horizontal direction toward the afternoon sun

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

export function ecology(x: number, z: number): Eco {
  const h = terrainHeightM(x, z);
  const e = 4;
  const hx = terrainHeightM(x + e, z) - h;
  const hz = terrainHeightM(x, z + e) - h;
  const nlen = Math.hypot(hx, e, hz);
  const ny = e / nlen;
  const slope = 1 - ny;
  // Aspect: slopes whose downhill faces the sun receive more radiation and dry out.
  const downX = -hx / nlen;
  const downZ = -hz / nlen;
  const exposure = Math.max(-1, Math.min(1, (downX * SUN_XZ[0] + downZ * SUN_XZ[1]) * 4));

  const elevation = Math.max(0, h - (terrainBroad(x, z) * 0.6 + 22));
  const temperature = clamp01(1 - elevation / 230);

  // Water: proximity to the river, and hollows where runoff gathers (local relief).
  const [d, , hw] = riverInfo(x, z);
  const riverWet = Math.exp(-Math.max(d - hw, 0) / 90);
  const r = 70;
  const ring = (terrainHeightM(x + r, z) + terrainHeightM(x - r, z) + terrainHeightM(x, z + r) + terrainHeightM(x, z - r)) / 4;
  const hollow = Math.max(-1, Math.min(1, (ring - h) / 18));
  const [grove, meadow, forest] = biome(x, z);
  const east = Math.min(1, Math.max(0, (x - 380) / 520));
  const moisture = clamp01(
    0.35 + 0.45 * riverWet + 0.2 * hollow + 0.3 * grove + 0.45 * forest - 0.3 * meadow - 0.15 * Math.max(exposure, 0) + 0.1 * (1 - temperature),
  );
  return { height: h, elevation, temperature, moisture, slope, exposure, forest: east };
}

function bell(v: number, center: number, width: number): number {
  const t = (v - center) / width;
  return Math.exp(-t * t);
}

/** Species order matches SPECIES in trees.ts: jacaranda, island, spruce. */
export function treeSuitability(eco: Eco): number[] {
  const steep = clamp01((eco.slope - 0.25) / 0.3);
  const cold = eco.temperature < 0.28 ? 0 : 1;
  const jacaranda = clamp01((eco.temperature - 0.75) / 0.2) * bell(eco.moisture, 0.42, 0.22) * (1 - steep) * (1 - eco.forest) + eco.forest * 0.3;
  // The eastern forest is mixed: broadleaves (autumn color) among the evergreen spruce.
  const island = clamp01((eco.temperature - 0.6) / 0.25) * bell(eco.moisture, 0.68, 0.22) * (1 - steep * 0.6) * (1 - eco.forest * 0.7) + eco.forest * 0.75;
  const spruce = (bell(eco.temperature, 0.55, 0.3) * clamp01((eco.moisture - 0.4) / 0.3) + eco.forest * 0.9) * (1 - steep * 0.3);
  return [jacaranda * cold, island * cold, spruce * cold];
}

/** Expected canopy cover (0..1): moist, mild ground grows forest; dry or cold stays open. */
export function treeDensity(eco: Eco): number {
  if (eco.temperature < 0.28) return 0;
  return clamp01((eco.moisture - 0.45) * 1.8) * clamp01((eco.temperature - 0.28) / 0.25) + eco.forest * 0.85;
}
