import { draw, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import petalShader from "../shaders/petals.wgsl";
import type { Vec3 } from "../engine/camera";

const MAX_PETALS = 1024;
const STRIDE = 12;
const PATH_STEP = 0.12;
const PATH_LEN = 1400;

interface Petal {
  pos: Vec3;
  color: [number, number, number];
  seed: number;
  size: number;
  age: number;
  stiffness: number;
}

/**
 * The stream of petals trailing the wind. Each petal is assigned an arc-length slot along the
 * leader's recorded path and swirls around it; newer petals join at the back of the stream.
 */
export class PetalStream {
  private readonly petals: Petal[] = [];
  private readonly path: Vec3[] = [];
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(MAX_PETALS * STRIDE);
  private readonly drawCall: Draw;
  private time = 0;

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.buffer = storage(gpu, MAX_PETALS * STRIDE * 4, "read");
    this.drawCall = draw(gpu, {
      label: "petals",
      shader: petalShader,
      vertices: 72,
      colors: [null, { writeMask: [] }],
      depth: { compare: "greater" },
      multisample: { alphaToCoverage: true },
      set: { G: globals, petals: this.buffer },
    });
  }

  get draw(): Draw {
    return this.drawCall;
  }

  get count(): number {
    return this.petals.length;
  }

  add(from: Vec3, color: [number, number, number]): void {
    if (this.petals.length >= MAX_PETALS) return;
    this.petals.push({
      pos: [from[0], from[1], from[2]],
      color,
      seed: Math.random(),
      size: 0.34 + Math.random() * 0.12,
      age: 0,
      stiffness: 3.2 + Math.random() * 3,
    });
  }

  /** Gathers the stream at a new spot (after free roam). */
  regroup(at: Vec3): void {
    this.path.length = 0;
    for (const p of this.petals) p.pos.splice(0, 3, at[0], at[1], at[2]);
  }

  /** Arc length of the stream behind the leader; it grows slowly with the petal count. */
  length(gust: number): number {
    return (1.2 + 1.5 * Math.sqrt(this.petals.length)) * (1 + gust * 0.5);
  }

  /** Average position of the stream (for the camera / audio). */
  center(): Vec3 {
    const c: Vec3 = [0, 0, 0];
    if (!this.petals.length) return c;
    for (const p of this.petals) for (let i = 0; i < 3; i++) c[i] += p.pos[i];
    return c.map((v) => v / this.petals.length) as Vec3;
  }

  private recordPath(leader: Vec3): void {
    const last = this.path[0];
    if (!last) {
      this.path.unshift([leader[0], leader[1], leader[2]]);
      return;
    }
    const d = Math.hypot(leader[0] - last[0], leader[1] - last[1], leader[2] - last[2]);
    if (d < PATH_STEP) return;
    const steps = Math.min(Math.floor(d / PATH_STEP), 50);
    for (let s = 1; s <= steps; s++) {
      const k = (s * PATH_STEP) / d;
      this.path.unshift([last[0] + (leader[0] - last[0]) * k, last[1] + (leader[1] - last[1]) * k, last[2] + (leader[2] - last[2]) * k]);
    }
    if (this.path.length > PATH_LEN) this.path.length = PATH_LEN;
  }

  private pathAt(dist: number): Vec3 {
    const i = Math.min(this.path.length - 1, Math.floor(dist / PATH_STEP));
    return this.path[Math.max(0, i)];
  }

  update(dt: number, leader: Vec3, forward: Vec3, gust: number): void {
    this.time += dt;
    this.recordPath(leader);
    const n = this.petals.length;
    const d = this.data;
    let o = 0;
    for (let i = 0; i < n; i++) {
      const p = this.petals[i];
      p.age += dt;
      // Slot along the path: the stream lengthens with sqrt(count) and stretches in a gust.
      const k = i / Math.max(1, n - 1);
      const length = this.length(gust);
      const s = 0.4 + k * length;
      const base = this.path.length ? this.pathAt(s) : leader;
      const r = 0.25 + 0.5 * Math.sqrt(k) * Math.min(1, n / 20) + gust * 0.2;
      const w = this.time * (1.6 + p.seed * 1.5) + p.seed * 50;
      // Orbit in a plane roughly perpendicular to travel.
      const side: Vec3 = [-forward[2], 0, forward[0]];
      const ox = Math.cos(w) * r;
      const oy = Math.sin(w * 1.3) * r * 0.6;
      const target: Vec3 = [base[0] + side[0] * ox, base[1] + oy, base[2] + side[2] * ox];
      const a = 1 - Math.exp(-p.stiffness * dt * (p.age < 1.5 ? 0.6 : 1));
      for (let j = 0; j < 3; j++) p.pos[j] += (target[j] - p.pos[j]) * a;

      d[o++] = p.pos[0]; d[o++] = p.pos[1]; d[o++] = p.pos[2]; d[o++] = p.size;
      d[o++] = p.color[0]; d[o++] = p.color[1]; d[o++] = p.color[2]; d[o++] = p.seed;
      d[o++] = 0; d[o++] = 0; d[o++] = 0; d[o++] = Math.max(0, 1 - p.age / 1.5);
    }
    if (o) this.buffer.write(d.subarray(0, o));
  }

  encode(pass: FramePass): void {
    if (this.petals.length) pass.draw(this.drawCall, { instances: this.petals.length });
  }
}
