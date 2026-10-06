// The lantern path (mirror of src/shaders/lib/path.wgsl): a gravel path winding from the
// spawn meadow east to the forest edge, lined with an avenue of trees and stone lanterns.
// No imports, so height / placement modules can use it freely.

export const PATH = { x0: 30, x1: 470, width: 1.3, lanternStep: 10, lanternX0: 40, lanternOffset: 2.3, treeOffset: 6.2 };

/** Path centerline z at x. */
export function pathZ(x: number): number {
  return 70 + 16 * Math.sin(x / 68 + 0.4) + 7 * Math.sin(x / 29 + 1.7);
}

function pathSlope(x: number): number {
  return (16 / 68) * Math.cos(x / 68 + 0.4) + (7 / 29) * Math.cos(x / 29 + 1.7);
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

/** Lantern k: every lanternStep meters along x, alternating sides. */
export function lanternAt(k: number): Lantern {
  const x = PATH.lanternX0 + k * PATH.lanternStep;
  const side = k % 2 === 0 ? 1 : -1;
  const [nx, nz] = pathNormal(x);
  return { x: x + nx * side * PATH.lanternOffset, z: pathZ(x) + nz * side * PATH.lanternOffset, yaw: Math.atan2(-nx * side, -nz * side), variant: k % 5 === 3 ? 1 : 0, index: k };
}

export function lanternCount(): number {
  return Math.floor((PATH.x1 - PATH.lanternX0) / PATH.lanternStep) + 1;
}

/** Avenue trees: between lanterns, on both sides. */
export function avenueTrees(): { x: number; z: number; k: number; side: number }[] {
  const out: { x: number; z: number; k: number; side: number }[] = [];
  for (let k = 0; ; k++) {
    const x = PATH.lanternX0 + PATH.lanternStep * (k + 0.5);
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
