import { worldConstants } from "./height";
import { compute, storage, type Compute, type Gpu, type StorageBuffer } from "vgpu";
import paintShader from "../shaders/life-paint.wgsl";

export const LIFE_SIZE = 512;

interface ActiveSplat {
  x: number;
  z: number;
  radius: number;
  age: number;
  duration: number;
  strength: number;
}

/**
 * World-space "restoration" map (1 m per cell, 512 m toroidal window keyed by world cell).
 * Blooming a flower starts a splat that grows outward over a few seconds.
 */
export class LifeMap {
  readonly buffer: StorageBuffer;
  private readonly paint: Compute;
  private readonly active: ActiveSplat[] = [];

  constructor(gpu: Gpu) {
    this.buffer = storage(gpu, LIFE_SIZE * LIFE_SIZE * 8, "read-write");
    this.paint = compute(gpu, paintShader, {
      constants: worldConstants(paintShader),
      label: "life-paint",
      set: { splat: { center: [0, 0], radius: 1, strength: 0 }, life: this.buffer },
    });
  }

  bloom(x: number, z: number, radius: number, duration = 4, strength = 1): void {
    this.active.push({ x, z, radius, age: 0, duration, strength });
  }

  update(dt: number): void {
    for (let i = this.active.length - 1; i >= 0; i--) {
      const s = this.active[i];
      s.age += dt;
      const k = Math.min(s.age / s.duration, 1);
      const ease = 1 - Math.pow(1 - k, 3);
      const radius = Math.max(1.5, s.radius * ease);
      this.paint.set({ splat: { center: [s.x, s.z], radius, strength: s.strength } });
      const span = Math.ceil(radius * 1.8) * 2 + 1;
      const groups = Math.ceil(span / 8);
      this.paint.dispatch(groups, groups);
      if (k >= 1) this.active.splice(i, 1);
    }
  }
}
