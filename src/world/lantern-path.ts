// The lantern path (mirror of src/shaders/lib/path.wgsl): a gravel path winding from the
// spawn meadow east to the forest edge, lined with an avenue of trees and stone lanterns.
// No imports, so height / placement modules can use it freely.

/** Where the path runs (x0..x1, centerline base z and wave phases come from the world seed). */
export const PATH = { x0: 30, x1: 470, width: 1.3, lanternX0: 40, lanternOffset: 1.6, treeOffset: 6.2, treeStep: 12, zBase: 70, p1: 0.4, p2: 1.7 };

/**
 * Lanterns are spaced by beats: at cruise speed (7.5 m/s) one lantern every 0.8 s (75 bpm),
 * in phrases of four followed by a one-beat rest. They sit just inside the reach of the
 * path's center line, alternating sides, so following the path plays the whole sequence.
 */
export const BEAT_METERS = 6;
const PHRASE = 4;
export const MAX_LANTERNS = 64;

/** Path centerline z at x. */
export function pathZ(x: number): number {
  return PATH.zBase + 16 * Math.sin(x / 68 + PATH.p1) + 7 * Math.sin(x / 29 + PATH.p2);
}

function pathSlope(x: number): number {
  return (16 / 68) * Math.cos(x / 68 + PATH.p1) + (7 / 29) * Math.cos(x / 29 + PATH.p2);
}

/** Unit normal of the centerline (pointing +z side) at x. */
export function pathNormal(x: number): [number, number] {
  const s = pathSlope(x);
  const l = Math.hypot(1, s);
  return [-s / l, 1 / l];
}

/** Distance from the centerline (perpendicular approximation), and how far along (0..1). */
export function pathDistance(x: number, z: number): number {
  if (x < PATH.x0 - 6 || x > PATH.x1 + 6) return 1e9;
  const s = pathSlope(x);
  const d = Math.abs(z - pathZ(x)) / Math.hypot(1, s);
  // Rounded ends.
  const over = Math.max(PATH.x0 - x, x - PATH.x1, 0);
  return Math.hypot(d, over);
}

export interface Lantern {
  x: number;
  z: number;
  /** Faces the path. */
  yaw: number;
  variant: number;
  index: number;
}

let lanternCache: Lantern[] | null = null;

/** All lanterns, in path order (placed by arc length; see BEAT_METERS). */
export function lanterns(): Lantern[] {
  if (lanternCache) return lanternCache;
  const out: Lantern[] = [];
  let x = PATH.lanternX0;
  let travelled = 0;
  let beat = 0;
  let nextAt = 0;
  while (x < PATH.x1 - 4 && out.length < MAX_LANTERNS) {
    if (travelled >= nextAt) {
      // Beats 0-3 of each 5-beat phrase hold a lantern; beat 4 is a rest.
      if (beat % (PHRASE + 1) < PHRASE) {
        const k = out.length;
        const side = k % 2 === 0 ? 1 : -1;
        const [nx, nz] = pathNormal(x);
        out.push({ x: x + nx * side * PATH.lanternOffset, z: pathZ(x) + nz * side * PATH.lanternOffset, yaw: Math.atan2(-nx * side, -nz * side), variant: k % 7 === 5 ? 1 : 0, index: k });
      }
      beat++;
      nextAt += BEAT_METERS;
    }
    const dx = 0.05;
    travelled += dx * Math.hypot(1, pathSlope(x));
    x += dx;
  }
  lanternCache = out;
  return out;
}

export function lanternAt(k: number): Lantern {
  return lanterns()[k];
}

export function lanternCount(): number {
  return lanterns().length;
}

/** Avenue trees: between lanterns, on both sides. */
export function avenueTrees(): { x: number; z: number; k: number; side: number }[] {
  const out: { x: number; z: number; k: number; side: number }[] = [];
  for (let k = 0; ; k++) {
    const x = PATH.lanternX0 + PATH.treeStep * (k + 0.5);
    if (x > PATH.x1 - 10) break;
    const [nx, nz] = pathNormal(x);
    for (const side of [1, -1]) out.push({ x: x + nx * side * PATH.treeOffset, z: pathZ(x) + nz * side * PATH.treeOffset, k, side });
  }
  return out;
}

/** True within `margin` of the path (keeps beds, flowers and random trees off it). */
export function pathNear(x: number, z: number, margin: number): boolean {
  return pathDistance(x, z) < margin;
}
