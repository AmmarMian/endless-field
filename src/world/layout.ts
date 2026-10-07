import { ecology } from "./ecology";
import { RIVER, riverCenter, riverInfo } from "./height";
import { PATH, pathDistance, pathZ } from "./lantern-path";
import { SUNFLOWERS, sunflowerNear } from "./sunflower-field";

/** Where the kingfisher's river course begins (x along the river). */
export const RACE = { startX: 70, length: 11 * 36 + 20 };

function rng(seed: number): () => number {
  let s = (seed * 2654435761) >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Within the explorable window (the map's extent), with a margin. */
function inWorld(x: number, z: number, m: number): boolean {
  return x > -1300 + m && x < 1900 - m && z > -1100 + m && z < 900 - m;
}

/**
 * Places the landmarks for a world seed: the sunflower field, the lantern path (with its
 * torii and avenue) and the river race. Seed 0 keeps the original layout; other seeds put
 * each one somewhere fitting (open land, clear of the river and of each other), unrelated to
 * where the others are. Call after setWorldSeed and before anything reads the landmarks.
 */
export function placeLandmarks(seed: number): void {
  if (!seed) return;
  const r = rng(seed);
  const rand = (a: number, b: number) => a + r() * (b - a);
  const spawn: [number, number] = [0, 30];

  // The river valley: shifted north or south, but never through the spawn meadow.
  for (let k = 0; k < 40; k++) {
    RIVER.dz = rand(-420, 380);
    let near = 1e9;
    for (let x = -80; x <= 80; x += 10) near = Math.min(near, Math.abs(riverCenter(x) - spawn[1]));
    if (near > 90) break;
  }

  // Sunflowers: open, gentle farmland away from the river.
  for (let k = 0; k < 120; k++) {
    const a = r() * Math.PI * 2;
    const d = rand(180, 650);
    const x = spawn[0] + Math.cos(a) * d;
    const z = spawn[1] + Math.sin(a) * d;
    if (!inWorld(x, z, 150)) continue;
    let ok = riverInfo(x, z)[0] > 140;
    for (let i = 0; ok && i < 10; i++) {
      const b = (i / 10) * Math.PI * 2;
      const px = x + Math.cos(b) * 95;
      const pz = z + Math.sin(b) * 95;
      const [rd, , hw] = riverInfo(px, pz);
      if (rd < hw + 18 || ecology(px, pz).forest > 0.3) ok = false;
    }
    if (!ok) continue;
    const t = r() * Math.PI * 2;
    SUNFLOWERS.x = x;
    SUNFLOWERS.z = z;
    SUNFLOWERS.dir = [Math.cos(t), Math.sin(t)];
    break;
  }

  // The lantern path: 440 m on land, never across the river, clear of the field.
  for (let k = 0; k < 200; k++) {
    const x0 = rand(-900, 1000);
    const zBase = rand(-800, 700);
    const p1 = rand(0, Math.PI * 2);
    const p2 = rand(0, Math.PI * 2);
    PATH.x0 = x0;
    PATH.x1 = x0 + 440;
    PATH.zBase = zBase;
    PATH.p1 = p1;
    PATH.p2 = p2;
    let ok = inWorld(x0, pathZ(x0), 60) && inWorld(PATH.x1, pathZ(PATH.x1), 60);
    let forest = 0;
    for (let x = x0 - 10; ok && x <= PATH.x1 + 10; x += 15) {
      const z = pathZ(x);
      const [rd, , hw] = riverInfo(x, z);
      if (rd < hw + 25 || sunflowerNear(x, z, 30)) ok = false;
      forest += ecology(x, z).forest > 0.5 ? 1 : 0;
    }
    if (ok && forest < 6) break;
  }
  PATH.lanternX0 = PATH.x0 + 10;

  // The kingfisher's course: a reach of river clear of the path and the field.
  for (let k = 0; k < 80; k++) {
    const sx = rand(-900, 1100);
    let ok = inWorld(sx, riverCenter(sx), 40) && inWorld(sx + RACE.length, riverCenter(sx + RACE.length), 40);
    for (let x = sx; ok && x <= sx + RACE.length; x += 12) {
      const z = riverCenter(x);
      if (pathDistance(x, z) < 25 || sunflowerNear(x, z, 25)) ok = false;
    }
    if (ok) {
      RACE.startX = sx;
      break;
    }
  }
}
