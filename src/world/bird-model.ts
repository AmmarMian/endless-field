import { draw, geometry, storage, type Draw, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import birdShader from "../shaders/birds.wgsl";

/** One baked animation clip: `frames` intervals from `start` (start + frames is its last sample). */
export interface Clip {
  start: number;
  frames: number;
  fps: number;
  loop: boolean;
}

interface Manifest {
  vertexBytes: number;
  indexCount: number;
  matrixBytes: number;
  bones: string[];
  clips: Record<string, Clip>;
  perchHeight: number;
  length: number;
}

/** Floats per bird instance (see `Bird` in birds.wgsl). */
export const BIRD_STRIDE = 12;

/**
 * A bird species (tools/blender/model_birds.py): its mesh, skin weights and baked animation.
 * Draws instances from a storage buffer of BIRD_STRIDE floats each:
 *   position xyz, heading | pitch, bank, size, glow | frame, faded-from frame, its weight, random
 */
export class BirdSpecies {
  /** The shader drawing this species, and the name it gives the instance buffer. */
  shader: typeof birdShader = birdShader;
  instanceKey = "birds";

  private constructor(
    private readonly gpu: Gpu,
    private readonly geo: ReturnType<typeof geometry>,
    private readonly skins: StorageBuffer,
    private readonly bones: number,
    readonly clips: Record<string, Clip>,
    /** Body origin to the soles when perched (model meters: times the instance size). */
    readonly perchHeight: number,
    /** Anything else the model's manifest carries (sizes, gait strides...). */
    readonly info: Record<string, unknown> = {},
  ) {}

  static async load(gpu: Gpu, name: string, base = "assets/birds"): Promise<BirdSpecies> {
    const [m, bin] = await Promise.all([
      fetch(`${base}/${name}.json`).then((r) => r.json() as Promise<Manifest>),
      fetch(`${base}/${name}.bin`).then((r) => r.arrayBuffer()),
    ]);
    const geo = geometry(gpu, {
      label: name,
      buffers: [{ data: new Uint8Array(bin, 0, m.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4", j: "uint8x4" } }],
      indices: new Uint32Array(bin, m.vertexBytes, m.indexCount),
    });
    const off = m.vertexBytes + m.indexCount * 4;
    const skins = storage(gpu, m.matrixBytes, "read");
    skins.write(new Uint32Array(bin.slice(off, off + m.matrixBytes)));
    return new BirdSpecies(gpu, geo, skins, m.bones.length, m.clips, m.perchHeight, m as unknown as Record<string, unknown>);
  }

  /** A draw of this species' birds from `instances` (and their soft light, with `halo`). */
  draws(
    globals: SharedUniforms,
    instances: StorageBuffer,
    label: string,
    opts: { constants?: Record<string, number>; alphaToCoverage?: boolean } = {},
  ): { body: Draw; halo: Draw } {
    const set = { G: globals, [this.instanceKey]: instances, skins: this.skins };
    const body = draw(this.gpu, {
      label,
      shader: this.shader,
      geometry: this.geo,
      cull: "none",
      depth: { compare: "greater" },
      constants: { BONES: this.bones, ...opts.constants },
      ...(opts.alphaToCoverage ? { multisample: { alphaToCoverage: true } } : {}),
      set,
    });
    const halo = draw(this.gpu, {
      label: `${label}-light`,
      shader: this.shader,
      entry: { vertex: "vs_halo", fragment: "fs_halo" },
      vertices: 6,
      blend: "additive",
      depth: { compare: "greater", write: false },
      constants: { BONES: this.bones, ...opts.constants },
      set,
    });
    return { body, halo };
  }
}

/**
 * Plays a species' clips on one bird: the current clip, and the one it is crossfading from.
 * Loops wrap; one-shot clips hold their last frame (`done` turns true).
 */
export class BirdAnimator {
  clip: string;
  /** Seconds into the current clip (scaled by its speed). */
  time = 0;
  speed = 1;
  private from: string | null = null;
  private fromTime = 0;
  private fromSpeed = 1;
  private fade = 0;
  private fadeLen = 0;

  constructor(
    private readonly clips: Record<string, Clip>,
    clip: string,
    time = 0,
  ) {
    this.clip = clip;
    this.time = time;
  }

  has(name: string): boolean {
    return name in this.clips;
  }

  /** Length of a clip in seconds (at speed 1). */
  duration(name = this.clip): number {
    const c = this.clips[name];
    return c ? c.frames / c.fps : 0;
  }

  /** Crossfades to `name` over `fade` s. Already playing it: only the speed changes (unless `restart`). */
  play(name: string, fade = 0.25, speed = 1, restart = false, at = 0): void {
    if (!(name in this.clips)) return;
    this.speed = speed;
    if (name === this.clip && !restart) return;
    if (fade > 0) {
      this.from = this.clip;
      this.fromTime = this.time;
      this.fromSpeed = this.from === name ? 0 : this.speed;
      this.fade = 0;
      this.fadeLen = fade;
    } else this.from = null;
    this.clip = name;
    this.time = at;
  }

  /** How far through its current clip (0..1; loops wrap). */
  get phase(): number {
    const c = this.clips[this.clip];
    const f = (this.time * c.fps) / c.frames;
    return c.loop ? ((f % 1) + 1) % 1 : Math.min(1, f);
  }

  /** The current one-shot has reached its end. */
  get done(): boolean {
    const c = this.clips[this.clip];
    return !c.loop && this.time * c.fps >= c.frames;
  }

  update(dt: number): void {
    this.time += dt * this.speed;
    if (this.from !== null) {
      this.fromTime += dt * this.fromSpeed;
      this.fade += dt;
      if (this.fade >= this.fadeLen) this.from = null;
    }
  }

  private frame(name: string, time: number): number {
    const c = this.clips[name];
    let f = time * c.fps;
    if (c.loop) f = ((f % c.frames) + c.frames) % c.frames;
    else f = Math.min(Math.max(f, 0), c.frames - 1e-3);
    return c.start + f;
  }

  /** Writes (frame, faded-from frame, its weight) at `out[o..o+3]`. */
  write(out: Float32Array, o: number): void {
    out[o] = this.frame(this.clip, this.time);
    if (this.from !== null) {
      const k = Math.min(1, this.fade / this.fadeLen);
      out[o + 1] = this.frame(this.from, this.fromTime);
      out[o + 2] = 1 - k * k * (3 - 2 * k);
    } else {
      out[o + 1] = out[o];
      out[o + 2] = 0;
    }
  }
}
