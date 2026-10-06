import { draw, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import trailShader from "../shaders/wind-trail.wgsl";
import type { Vec3 } from "../engine/camera";

const SAMPLES = 128;
const STEP = 0.2;
const STRANDS = 7;

/** The wind made visible: curling strands of brighter air along its recent path. */
export class WindTrail {
  readonly draw: Draw;
  private readonly path: Vec3[] = [];
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(SAMPLES * 4);
  private flow = 0;

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.buffer = storage(gpu, SAMPLES * 16, "read");
    this.draw = draw(gpu, {
      label: "wind-trail",
      shader: trailShader,
      vertices: (SAMPLES - 1) * 6,
      blend: "additive",
      colors: [null, { writeMask: [] }],
      cull: "none",
      depth: { compare: "greater", write: false },
      set: { G: globals, path: this.buffer, T: { count: 0, strength: 0, step: STEP, flow: 0 } },
    });
  }

  /** Starts the trail afresh (after a jump). */
  reset(): void {
    this.path.length = 0;
  }

  update(dt: number, leader: Vec3, speed: number, gust: number): void {
    const last = this.path[0];
    if (!last) this.path.unshift([leader[0], leader[1], leader[2]]);
    else {
      const d = Math.hypot(leader[0] - last[0], leader[1] - last[1], leader[2] - last[2]);
      if (d > 30) this.path.length = 0;
      if (d >= STEP) {
        const steps = Math.min(Math.floor(d / STEP), 60);
        for (let s = 1; s <= steps; s++) {
          const k = (s * STEP) / d;
          this.path.unshift([last[0] + (leader[0] - last[0]) * k, last[1] + (leader[1] - last[1]) * k, last[2] + (leader[2] - last[2]) * k]);
        }
        if (this.path.length > SAMPLES) this.path.length = SAMPLES;
      }
    }
    // The flow pattern streams back at the wind's speed (in arc length).
    this.flow -= speed * dt * 0.6;
    this.path.forEach((p, i) => this.data.set([p[0], p[1], p[2], 0], i * 4));
    this.buffer.write(this.data.subarray(0, this.path.length * 4));
    const strength = Math.min(1, Math.max(0, (speed - 4) / 17) * 0.6 + gust * 0.6);
    this.draw.set({ T: { count: this.path.length, strength, flow: this.flow } });
  }

  encode(pass: FramePass): void {
    if (this.path.length > 4) pass.draw(this.draw, { instances: STRANDS });
  }
}
