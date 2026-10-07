import { draw, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import shader from "../shaders/sky-lanterns.wgsl";
import type { Vec3 } from "../engine/camera";

const MAX = 72;

interface Lantern {
  pos: Vec3;
  rise: number;
  age: number;
  life: number;
  seed: number;
}

/** Paper sky lanterns released when the whole lantern path is lit: they rise and drift away. */
export class SkyLanterns {
  readonly draw: Draw;
  private readonly list: Lantern[] = [];
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(MAX * 4);

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.buffer = storage(gpu, MAX * 16, "read");
    this.draw = draw(gpu, {
      label: "sky-lanterns",
      shader,
      vertices: 6,
      blend: "additive",
      depth: { compare: "greater", write: false },
      set: { G: globals, lanterns: this.buffer },
    });
  }

  /** Releases one sky lantern above each point, a few at a time. */
  release(points: Vec3[]): void {
    points.forEach((p, i) => {
      if (this.list.length >= MAX) return;
      this.list.push({ pos: [p[0], p[1] + 1.4, p[2]], rise: 0.55 + Math.random() * 0.4, age: -i * 0.35 - Math.random() * 0.5, life: 80 + Math.random() * 30, seed: Math.random() });
    });
  }

  update(dt: number, windDir: [number, number]): void {
    let o = 0;
    for (let i = this.list.length - 1; i >= 0; i--) {
      const l = this.list[i];
      l.age += dt;
      if (l.age < 0) continue;
      if (l.age > l.life) {
        this.list.splice(i, 1);
        continue;
      }
      // Rising on warm air, drifting more as it climbs into the breeze, swaying a little.
      const drift = 0.3 + Math.min(1.5, l.age * 0.03);
      l.pos[0] += (windDir[0] * drift + Math.sin(l.age * 0.7 + l.seed * 9) * 0.15) * dt;
      l.pos[1] += l.rise * Math.min(1, l.age * 0.5) * dt;
      l.pos[2] += (windDir[1] * drift + Math.cos(l.age * 0.6 + l.seed * 7) * 0.15) * dt;
      const fade = Math.min(1, l.age * 0.8) * Math.min(1, (l.life - l.age) / 10);
      this.data.set([l.pos[0], l.pos[1], l.pos[2], fade], o);
      o += 4;
    }
    this.count = o / 4;
    if (o) this.buffer.write(this.data.subarray(0, o));
  }

  private count = 0;

  encode(pass: FramePass): void {
    if (this.count) pass.draw(this.draw, { instances: this.count });
  }
}
