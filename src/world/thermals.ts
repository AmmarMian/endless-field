import { draw, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import shader from "../shaders/insects.wgsl";
import type { Vec3 } from "../engine/camera";
import { ecology } from "./ecology";
import { riverInfo, terrainHeightM as terrainHeight } from "./height";

const COUNT = 4;
const SPECKS = 160;
/** How high a thermal carries (m above its ground). */
export const THERMAL_TOP = 28;

interface Thermal {
  x: number;
  z: number;
  ground: number;
  radius: number;
  seeds: number[];
}

function rand(a: number, b: number): number {
  return a + Math.random() * (b - a);
}

/**
 * Thermals: columns of warm air rising off sunny open slopes by day. A slow spiral of pale
 * specks (seed fluff, dust) shows each one; circling inside it lifts the wind high without
 * a wingbeat.
 */
export class Thermals {
  readonly draw: Draw;
  private readonly list: Thermal[] = [];
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(COUNT * SPECKS * 4);
  private count = 0;

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.buffer = storage(gpu, this.data.byteLength, "read");
    this.draw = draw(gpu, {
      label: "thermals",
      shader,
      vertices: 6,
      blend: "additive",
      depth: { compare: "greater", write: false },
      constants: { SIZE_K: 2.4, WARMTH: 0.15 },
      set: { G: globals, specks: this.buffer },
    });
  }

  private place(near: Vec3, heading: number): Thermal | null {
    for (let k = 0; k < 14; k++) {
      const a = heading - Math.PI / 2 + rand(-1.6, 1.6);
      const r = rand(45, 170);
      const x = near[0] + Math.cos(a) * r;
      const z = near[2] + Math.sin(a) * r;
      const eco = ecology(x, z);
      const [d, , hw] = riverInfo(x, z);
      // Sun-facing open slopes heat best.
      if (eco.forest > 0.2 || d < hw + 10 || eco.exposure < 0.1 || eco.slope < 0.03) continue;
      return { x, z, ground: terrainHeight(x, z), radius: rand(7, 10), seeds: Array.from({ length: SPECKS }, () => Math.random()) };
    }
    return null;
  }

  /** Where the thermals are, with their tops (for the hawks circling in them). */
  get columns(): { x: number; z: number; top: number; radius: number }[] {
    return this.list.map((th) => ({ x: th.x, z: th.z, top: th.ground + THERMAL_TOP, radius: th.radius }));
  }

  /** The thermal column the point is in (its core), or null. */
  at(p: Vec3, day: number): { x: number; z: number; ground: number; radius: number } | null {
    if (day < 0.3) return null;
    for (const th of this.list) {
      if (Math.hypot(p[0] - th.x, p[2] - th.z) < th.radius * 0.9 && p[1] - th.ground < THERMAL_TOP - 4) return th;
    }
    return null;
  }

  /** Lift at a point: 0 outside .. 1 in the core, fading near the top. */
  lift(p: Vec3, day: number): number {
    let best = 0;
    for (const th of this.list) {
      const d = Math.hypot(p[0] - th.x, p[2] - th.z) / th.radius;
      const h = (p[1] - th.ground) / THERMAL_TOP;
      if (d < 1 && h < 1) best = Math.max(best, (1 - d * d) * (1 - Math.max(0, h - 0.7) / 0.3));
    }
    return best * day;
  }

  update(t: number, near: Vec3, heading: number, day: number): void {
    while (this.list.length < COUNT) {
      const th = this.place(near, heading);
      if (!th) break;
      this.list.push(th);
    }
    let o = 0;
    for (const [i, th] of this.list.entries()) {
      if (Math.hypot(th.x - near[0], th.z - near[2]) > 260) {
        const n = this.place(near, heading);
        if (n) this.list[i] = n;
        continue;
      }
      if (day < 0.05) continue;
      for (const q of th.seeds) {
        // Rising and turning slowly, the spiral tighter toward the top.
        const rise = (t * 0.07 + q) % 1;
        const h = rise * THERMAL_TOP;
        const a = q * 40 + t * (0.35 + q * 0.2);
        const r = th.radius * (0.25 + 0.6 * ((q * 13.7) % 1)) * (1 - rise * 0.35);
        const fade = Math.min(1, rise * 6) * Math.min(1, (1 - rise) * 4);
        this.data.set([th.x + Math.cos(a) * r, th.ground + 0.5 + h, th.z + Math.sin(a) * r, 0.75 * fade * day], o);
        o += 4;
      }
    }
    this.count = o / 4;
    if (o) this.buffer.write(this.data.subarray(0, o));
  }

  encode(pass: FramePass): void {
    if (this.count) pass.draw(this.draw, { instances: this.count });
  }
}
