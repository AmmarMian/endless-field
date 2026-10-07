import { draw, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import shader from "../shaders/insects.wgsl";
import type { Vec3 } from "../engine/camera";
import { ecology } from "./ecology";
import { riverInfo, surfaceHeight } from "./height";

const SWARMS = 9;
const PER = 26;
const MAX = SWARMS * PER;

interface Swarm {
  centre: Vec3;
  radius: number;
  /** Which of its insects are still there. */
  alive: boolean[];
  seeds: number[];
  /** Seconds before an emptied swarm re-forms elsewhere. */
  regrow: number;
}

function rand(a: number, b: number): number {
  return a + Math.random() * (b - a);
}

/**
 * Swarms of small insects hanging in the air over the meadow and the river. A swallow catches
 * them on the wing: flying through a swarm snaps them up one by one (`onCatch`).
 */
export class Insects {
  readonly draw: Draw;
  private readonly swarms: Swarm[] = [];
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(MAX * 4);
  private count = 0;
  /** Caught one: where. */
  onCatch: ((at: Vec3) => void) | null = null;

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.buffer = storage(gpu, MAX * 16, "read");
    this.draw = draw(gpu, {
      label: "insects",
      shader,
      vertices: 6,
      blend: "additive",
      depth: { compare: "greater", write: false },
      set: { G: globals, specks: this.buffer },
    });
  }

  /** Swarm centres (for the birds to lead to, for instance). */
  get centres(): Vec3[] {
    return this.swarms.filter((s) => s.regrow <= 0).map((s) => s.centre);
  }

  private place(near: Vec3, heading: number): Swarm | null {
    for (let k = 0; k < 10; k++) {
      // Mostly ahead, 25-110 m away.
      const a = heading - Math.PI / 2 + rand(-1.3, 1.3);
      const r = rand(25, 110);
      const x = near[0] + Math.cos(a) * r;
      const z = near[2] + Math.sin(a) * r;
      if (ecology(x, z).forest > 0.4) continue;
      const [d, , hw] = riverInfo(x, z);
      const overWater = d < hw;
      // Over water they hang low (to be caught skimming); over the meadow a little higher.
      const y = surfaceHeight(x, z) + (overWater ? rand(0.7, 1.1) : rand(1.2, 3.2));
      return { centre: [x, y, z], radius: rand(0.9, 1.6), alive: new Array(PER).fill(true), seeds: Array.from({ length: PER }, () => Math.random()), regrow: 0 };
    }
    return null;
  }

  update(dt: number, t: number, wind: Vec3, heading: number, catching: boolean): void {
    while (this.swarms.length < SWARMS) {
      const s = this.place(wind, heading);
      if (!s) break;
      this.swarms.push(s);
    }
    let o = 0;
    for (const [si, s] of this.swarms.entries()) {
      const far = Math.hypot(s.centre[0] - wind[0], s.centre[2] - wind[2]) > 180;
      const empty = s.alive.every((a) => !a);
      if (empty && s.regrow <= 0) s.regrow = rand(8, 18);
      if (s.regrow > 0) s.regrow -= dt;
      if (far || (empty && s.regrow <= 0)) {
        const n = this.place(wind, heading);
        if (n) this.swarms[si] = n;
        continue;
      }
      for (let i = 0; i < PER; i++) {
        if (!s.alive[i]) continue;
        const q = s.seeds[i] * 100;
        // Each insect dances on its own little loop; the cloud itself drifts and breathes.
        const r = s.radius * (0.35 + 0.65 * ((q * 7.31) % 1));
        const p: Vec3 = [
          s.centre[0] + Math.sin(t * (1.3 + (q % 1.7)) + q) * r + Math.sin(t * 0.3 + si) * 0.4,
          s.centre[1] + Math.sin(t * (2.1 + (q % 1.1)) + q * 2) * r * 0.45 + Math.sin(t * 0.5 + si * 2) * 0.25,
          s.centre[2] + Math.cos(t * (1.1 + (q % 1.3)) + q * 3) * r,
        ];
        if (catching && Math.hypot(p[0] - wind[0], p[1] - wind[1], p[2] - wind[2]) < 0.95) {
          s.alive[i] = false;
          this.onCatch?.(p);
          continue;
        }
        const twinkle = 0.55 + 0.45 * Math.sin(t * 9 + q * 5);
        this.data.set([p[0], p[1], p[2], twinkle], o);
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
