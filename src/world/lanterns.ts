import { draw, geometry, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import lanternShader from "../shaders/lantern.wgsl";
import { terrainHeightM as terrainHeight, worldConstants } from "./height";
import { lanternAt, lanternCount } from "./lantern-path";
import { loadBin } from "../engine/assets";

interface Manifest {
  vertexBytes: number;
  indexCount: number;
  variants: { height: number; lightY: number; lods: { firstIndex: number; indexCount: number }[] }[];
}

/** Full-detail range (m); beyond it the lighter mesh, and nothing past RANGE. */
const NEAR = 30;
const RANGE = 300;
const FLOATS = 8;
/** How close the wind must pass (m), and how long a chain waits for the next lantern (s). */
const REACH = 2.6;
const CHAIN_TIMEOUT = 14;

/** Stone lanterns along the lantern path (Blender model, tools/blender/model_lantern.py). */
export class Lanterns {
  readonly draws: Draw[];
  private readonly items: { x: number; y: number; z: number; variant: number; data: number[] }[];
  private readonly lists: number[][];

  private constructor(draws: Draw[], private readonly buffers: StorageBuffer[], items: Lanterns["items"]) {
    this.draws = draws;
    this.items = items;
    this.lists = draws.map(() => []);
    this.lit = new Float32Array(items.length);
    this.litTarget = new Uint8Array(items.length);
    this.flare = new Float32Array(items.length);
    this.near = items.map(() => false);
  }

  static async load(gpu: Gpu, globals: SharedUniforms, base = "assets/lantern"): Promise<Lanterns> {
    const [manifest, bin] = await Promise.all([
      fetch(`${base}/lantern.json`).then((r) => r.json() as Promise<Manifest>),
      loadBin(`${base}/lantern.bin`),
    ]);
    const geo = geometry(gpu, {
      label: "lantern",
      buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
    });
    const n = lanternCount();
    const items = Array.from({ length: n }, (_, k) => {
      const l = lanternAt(k);
      // Sink the plinth a little so it sits in the gravel on uneven ground.
      const y = terrainHeight(l.x, l.z) - 0.05;
      return { x: l.x, y, z: l.z, variant: l.variant, data: [l.x, y, l.z, 1, Math.cos(l.yaw), Math.sin(l.yaw), k, 0] };
    });
    const buffers: StorageBuffer[] = [];
    const draws: Draw[] = [];
    for (const v of manifest.variants) {
      for (const lod of v.lods) {
        const buffer = storage(gpu, n * FLOATS * 4, "read");
        buffers.push(buffer);
        draws.push(
          draw(gpu, {
            label: "lantern",
            shader: lanternShader,
            constants: worldConstants(lanternShader),
            geometry: geo.slice({ firstIndex: lod.firstIndex, indexCount: lod.indexCount }),
            depth: { compare: "greater" },
            set: { G: globals, lanterns: buffer },
          }),
        );
      }
    }
    return new Lanterns(draws, buffers, items);
  }

  /** Lantern positions (debug / tests). */
  get positions(): [number, number, number][] {
    return this.items.map((it) => [it.x, it.y, it.z]);
  }

  /** Writes lantern positions (and the lookup header) into the globals' lampPos array. */
  writePositions(out: Float32Array): void {
    out.fill(0);
    this.items.forEach((it, k) => out.set([it.x, it.y + 0.9, it.z], k * 4));
    const n = this.items.length;
    out[3] = n;
    out[7] = this.items[0].x;
    out[11] = n > 1 ? (this.items[this.items.length - 1].x - this.items[0].x) / (this.items.length - 1) : 1;
  }

  /** Lit state, flare (decays after a touch), and the current in-order chain. */
  private readonly lit: Float32Array;
  private readonly litTarget: Uint8Array;
  private readonly flare: Float32Array;
  private chain = 0;
  private lastIndex = -2;
  private lastTouch = -1e9;
  private readonly near: boolean[];

  /** Lanterns lit so far, the length of the current in-order chain, and the total. */
  get progress(): { lit: number; chain: number; total: number } {
    let lit = 0;
    for (const v of this.litTarget) lit += v;
    return { lit, chain: this.chain, total: this.items.length };
  }

  /**
   * The wind lights lanterns it passes within reach of. Lighting the next one in order extends
   * the chain (each a note higher); out of order starts a new chain there; dawdling more than
   * CHAIN_TIMEOUT seconds lets it lapse. Returns what happened this frame (for sound and HUD).
   */
  touch(at: readonly number[], now: number): { index: number; chain: number; complete: boolean } | null {
    let event: { index: number; chain: number; complete: boolean } | null = null;
    this.items.forEach((it, k) => {
      const inside = Math.hypot(it.x - at[0], it.z - at[2]) < REACH && at[1] - it.y < 4;
      if (inside && !this.near[k]) {
        const inOrder = k === this.lastIndex + 1 && now - this.lastTouch < CHAIN_TIMEOUT;
        this.chain = inOrder ? this.chain + 1 : 1;
        this.lastIndex = k;
        this.lastTouch = now;
        this.litTarget[k] = 1;
        this.flare[k] = 1;
        const complete = this.chain === this.items.length;
        if (complete) this.flare.fill(1);
        event = { index: k, chain: this.chain, complete };
      }
      this.near[k] = inside;
    });
    return event;
  }

  /** Eases brightness and writes it (index order) into the globals' lamp array. */
  animate(dt: number, out: Float32Array, now: number): void {
    for (let k = 0; k < this.items.length; k++) {
      this.lit[k] += (this.litTarget[k] - this.lit[k]) * Math.min(1, dt * 3);
      this.flare[k] = Math.max(0, this.flare[k] - dt / 1.6);
      // The finale's light runs down the path as a wave.
      out[k] = this.lit[k] + this.flare[k] * this.flare[k];
    }
    if (now - this.lastTouch > CHAIN_TIMEOUT) this.chain = 0;
  }

  update(cam: readonly number[]): void {
    for (const l of this.lists) l.length = 0;
    for (const it of this.items) {
      const d = Math.hypot(it.x - cam[0], it.y - cam[1], it.z - cam[2]);
      if (d > RANGE) continue;
      this.lists[it.variant * 2 + (d < NEAR ? 0 : 1)].push(...it.data);
    }
    this.lists.forEach((l, i) => {
      if (l.length) this.buffers[i].write(new Float32Array(l));
    });
  }

  encode(pass: FramePass): void {
    this.draws.forEach((d, i) => {
      const count = this.lists[i].length / FLOATS;
      if (count) pass.draw(d, { instances: count });
    });
  }
}
