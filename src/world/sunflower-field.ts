// Sunflower field geometry (mirror of src/shaders/lib/sunflowers.wgsl). No imports, so the
// height field can level the ground under the field without a module cycle.

/** Center, row direction, half extents (along, across the rows), drill-row spacing, ground level. */
export const SUNFLOWERS = { x: -62, z: -400, dir: [0.3902, -0.9207] as [number, number], rx: 80, rz: 50, row: 0.76, level: -22.1 };

export function sunflowerLocal(x: number, z: number): [number, number] {
  const dx = x - SUNFLOWERS.x;
  const dz = z - SUNFLOWERS.z;
  const [ux, uz] = SUNFLOWERS.dir;
  return [dx * ux + dz * uz, dx * -uz + dz * ux];
}

/** Superellipse distance from the center: 1 at the field edge (with a ragged wobble). */
function edgeDistance(a: number, b: number): number {
  const qx = Math.abs(a) / SUNFLOWERS.rx;
  const qz = Math.abs(b) / SUNFLOWERS.rz;
  return Math.pow(qx ** 6 + qz ** 6, 1 / 6) + 0.03 * Math.sin(a * 0.11 + 1.3) + 0.03 * Math.sin(b * 0.17);
}

function smooth(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** 1 inside the field, easing to 0 over its ragged edge. */
export function sunflowerField(x: number, z: number): number {
  const [a, b] = sunflowerLocal(x, z);
  if (Math.abs(a) > SUNFLOWERS.rx * 1.2 || Math.abs(b) > SUNFLOWERS.rz * 1.2) return 0;
  return 1 - smooth(0.95, 1.0, edgeDistance(a, b));
}

/** How strongly the ground is leveled (farmland is graded): full inside, easing out beyond. */
export function sunflowerFlat(x: number, z: number): number {
  const [a, b] = sunflowerLocal(x, z);
  if (Math.abs(a) > SUNFLOWERS.rx * 1.6 || Math.abs(b) > SUNFLOWERS.rz * 1.8) return 0;
  return (1 - smooth(0.9, 1.45, edgeDistance(a, b))) * 0.85;
}

/** True within `margin` meters of the field's bounding box (keeps beds, trees and props out). */
export function sunflowerNear(x: number, z: number, margin: number): boolean {
  const [a, b] = sunflowerLocal(x, z);
  return Math.abs(a) < SUNFLOWERS.rx + margin && Math.abs(b) < SUNFLOWERS.rz + margin;
}
