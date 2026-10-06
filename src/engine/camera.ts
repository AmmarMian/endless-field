import { mat4, type Mat4 } from "wgpu-matrix";

export type Vec3 = [number, number, number];

/**
 * Perspective camera with a reversed-Z, infinite-far projection: depth 1 at the near plane,
 * approaching 0 at infinity. Pair with `depth: { compare: "greater" }` and `clearDepth: 0`.
 * That keeps precision even across a kilometre of rolling hills.
 */
export class Camera {
  readonly position: Vec3 = [0, 2, 0];
  readonly target: Vec3 = [0, 2, -1];
  fovY = (55 * Math.PI) / 180;
  aspect = 16 / 9;
  near = 0.08;

  readonly view: Mat4 = mat4.identity();
  readonly projection: Mat4 = mat4.identity();
  readonly viewProj: Mat4 = mat4.identity();
  readonly invViewProj: Mat4 = mat4.identity();
  /** Six planes (xyz normal pointing inward, w offset), flattened for the GPU. */
  readonly frustum = new Float32Array(24);
  readonly frustumViews = Array.from({ length: 6 }, (_, i) => this.frustum.subarray(i * 4, i * 4 + 4));

  update(): void {
    mat4.lookAt(this.position, this.target, [0, 1, 0], this.view);
    const f = 1 / Math.tan(this.fovY / 2);
    const p = this.projection;
    p.fill(0);
    p[0] = f / this.aspect;
    p[5] = f;
    p[11] = -1;
    p[14] = this.near;
    mat4.multiply(this.projection, this.view, this.viewProj);
    mat4.inverse(this.viewProj, this.invViewProj);
    this.extractFrustum();
  }

  private extractFrustum(): void {
    const m = this.viewProj;
    const row = (i: number) => [m[i], m[4 + i], m[8 + i], m[12 + i]];
    const r0 = row(0);
    const r1 = row(1);
    const r2 = row(2);
    const r3 = row(3);
    const planes = [
      r3.map((v, i) => v + r0[i]), // left
      r3.map((v, i) => v - r0[i]), // right
      r3.map((v, i) => v + r1[i]), // bottom
      r3.map((v, i) => v - r1[i]), // top
      r3.map((v, i) => v - r2[i]), // near (reversed-Z: z <= w)
      r2, // far plane at infinity: never culls
    ];
    planes.forEach((pl, i) => {
      const len = Math.hypot(pl[0], pl[1], pl[2]) || 1;
      this.frustum.set([pl[0] / len, pl[1] / len, pl[2] / len, pl[3] / len], i * 4);
    });
  }
}
