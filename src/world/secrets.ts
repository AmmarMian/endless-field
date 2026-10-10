import { draw, geometry, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import shader from "../shaders/secrets.wgsl";
import type { Vec3 } from "../engine/camera";
import { ecology } from "./ecology";
import { riverInfo, terrainHeightM as terrainHeight } from "./height";
import { pathNear } from "./lantern-path";
import { sunflowerNear } from "./sunflower-field";
import { loadBin } from "../engine/assets";

interface Manifest {
  vertexBytes: number;
  indexCount: number;
  variants: { name: string; firstIndex: number; indexCount: number }[];
}

/** A secret: a ring of pinwheels in the meadow, or a frame of wind bells at a wood's edge. */
export interface Secret {
  kind: "pinwheels" | "bells";
  x: number;
  z: number;
  y: number;
  /** Woken by the wind (kept across sessions). */
  awake: boolean;
}

interface Piece {
  variant: number;
  pos: Vec3;
  yaw: number;
  scale: number;
  seed: number;
  spin: number;
  /** Spin speed (rad/s) or swing amplitude (rad), excited by the wind, easing back. */
  energy: number;
  secret: number;
}

function hash(x: number, z: number, k: number): number {
  const s = Math.sin(x * 12.9898 + z * 78.233 + k * 37.719) * 43758.5453;
  return s - Math.floor(s);
}

/** How a hilltop stands out from the ground around it (m). */
function prominence(x: number, z: number): number {
  const h = terrainHeight(x, z);
  let ring = 0;
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    ring += terrainHeight(x + Math.cos(a) * 25, z + Math.sin(a) * 25);
  }
  return h - ring / 8;
}

/**
 * Places the secrets for this world: on a coarse grid, each cell keeps its best candidate (a
 * rise in the open meadow for pinwheels; the edge of a wood for wind bells), so they sit where
 * they can be seen and are spread across the land.
 */
export function placeSecrets(): Secret[] {
  const out: Secret[] = [];
  const CELL = 290;
  for (let cx = -1200; cx < 1800; cx += CELL) {
    for (let cz = -1000; cz < 800; cz += CELL) {
      let best: Secret | null = null;
      let bestScore = 0;
      for (let k = 0; k < 10; k++) {
        const x = cx + hash(cx, cz, k) * CELL;
        const z = cz + hash(cz, cx, k + 7) * CELL;
        if (Math.hypot(x, z - 30) < 40) continue; // not right at the spawn
        const eco = ecology(x, z);
        const [d, , hw] = riverInfo(x, z);
        if (d < hw + 10 || pathNear(x, z, 12) || sunflowerNear(x, z, 10) || eco.slope > 0.35) continue;
        const edge = eco.forest > 0.15 && eco.forest < 0.5;
        const open = eco.forest < 0.1;
        if (!edge && !open) continue;
        const score = (open ? 1 : 0.8) * (1 + Math.max(0, prominence(x, z)) * 0.6) * (0.5 + hash(x, z, 3));
        if (score > bestScore) {
          bestScore = score;
          best = { kind: open ? "pinwheels" : "bells", x, z, y: terrainHeight(x, z), awake: false };
        }
      }
      if (best && hash(cx, cz, 99) < 0.38) out.push(best);
    }
  }
  return out;
}

const STORE = "endless-field-secrets";
/** Extra glows other things can show (where the birds are feeding). */
const EXTRA = 8;

/**
 * The wind's secrets: pinwheels that spin up and wind bells that ring as the wind passes.
 * Each one not yet woken glints from afar; waking it is celebrated by `onWake`.
 */
export class Secrets {
  readonly draws: Draw[];
  readonly glintDraw: Draw;
  readonly list: Secret[];
  private readonly pieces: Piece[] = [];
  private readonly buffer: StorageBuffer;
  private readonly glintBuf: StorageBuffer;
  private readonly data: Float32Array<ArrayBuffer>;
  private readonly glintData: Float32Array<ArrayBuffer>;
  private readonly ranges: { first: number; count: number }[] = [];
  private readonly key: string;
  private visible = new Set<number>();
  /** A secret woken for the first time. */
  onWake: ((s: Secret, index: number) => void) | null = null;
  /** A bell rang / pinwheels whirred (for sound): position and strength 0..1. */
  onBell: ((at: Vec3, strength: number, bell: number) => void) | null = null;
  onWhirr: ((at: Vec3, strength: number) => void) | null = null;

  private constructor(draws: Draw[], glintDraw: Draw, list: Secret[], pieces: Piece[], buffer: StorageBuffer, glintBuf: StorageBuffer, key: string) {
    this.draws = draws;
    this.glintDraw = glintDraw;
    this.list = list;
    this.pieces = pieces;
    this.buffer = buffer;
    this.glintBuf = glintBuf;
    this.key = key;
    this.data = new Float32Array(pieces.length * 8);
    this.glintData = new Float32Array((list.length + EXTRA) * 4);
    // Pieces are sorted by variant, so each draw takes a contiguous range.
    for (let v = 0; v < draws.length; v++) {
      const first = pieces.findIndex((p) => p.variant === v);
      const count = pieces.filter((p) => p.variant === v).length;
      this.ranges.push({ first: Math.max(0, first), count });
    }
  }

  static async load(gpu: Gpu, globals: SharedUniforms, seed: number, windDir: [number, number], base = "assets/secrets"): Promise<Secrets> {
    const [manifest, bin] = await Promise.all([
      fetch(`${base}/secrets.json`).then((r) => r.json() as Promise<Manifest>),
      loadBin(`${base}/secrets.bin`),
    ]);
    const geo = geometry(gpu, {
      label: "secrets",
      buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
    });
    const key = `${STORE}-${seed}`;
    let woken: number[] = [];
    try {
      woken = JSON.parse(localStorage.getItem(key) ?? "[]") as number[];
    } catch {
      // Storage unavailable.
    }
    const list = placeSecrets();
    for (const i of woken) if (list[i]) list[i].awake = true;
    // Face into the prevailing wind (pinwheels turn their sails to it; bells hang across it).
    const facing = Math.atan2(-windDir[0], -windDir[1]);
    const pieces: Piece[] = [];
    list.forEach((s, si) => {
      if (s.kind === "pinwheels") {
        const n = 6 + Math.floor(hash(s.x, s.z, 1) * 4);
        for (let i = 0; i < n; i++) {
          const a = (i / n) * Math.PI * 2 + hash(s.x, s.z, i) * 0.6;
          const r = 1.2 + hash(s.z, s.x, i) * 1.8;
          const x = s.x + Math.cos(a) * r;
          const z = s.z + Math.sin(a) * r;
          pieces.push({
            variant: Math.floor(hash(x, z, 5) * 3),
            pos: [x, terrainHeight(x, z) - 0.03, z],
            yaw: facing + (hash(x, z, 6) - 0.5) * 0.5,
            scale: 2.5 + hash(x, z, 8) * 0.7,
            seed: hash(x, z, 9),
            spin: hash(x, z, 10) * 6,
            energy: s.awake ? 2 : 0.4,
            secret: si,
          });
        }
      } else {
        pieces.push({ variant: 3, pos: [s.x, s.y - 0.05, s.z], yaw: facing + Math.PI / 2, scale: 1, seed: hash(s.x, s.z, 2), spin: 0, energy: 0.08, secret: si });
      }
    });
    pieces.sort((a, b) => a.variant - b.variant);
    const buffer = storage(gpu, Math.max(1, pieces.length) * 32, "read");
    const glintBuf = storage(gpu, (list.length + EXTRA) * 16, "read");
    const draws = manifest.variants.map((v) =>
      draw(gpu, {
        label: `secret-${v.name}`,
        shader,
        geometry: geo.slice({ firstIndex: v.firstIndex, indexCount: v.indexCount }),
        cull: "none",
        depth: { compare: "greater" },
        set: { G: globals, secrets: buffer, glints: glintBuf },
      }),
    );
    const glintDraw = draw(gpu, {
      label: "secret-glint",
      shader,
      entry: { vertex: "vs_glint", fragment: "fs_glint" },
      vertices: 6,
      blend: "additive",
      depth: { compare: "greater", write: false },
      set: { G: globals, secrets: buffer, glints: glintBuf },
    });
    return new Secrets(draws, glintDraw, list, pieces, buffer, glintBuf, key);
  }

  /** Secrets not yet woken (for the birds to lead toward). */
  get hidden(): Secret[] {
    return this.list.filter((s) => !s.awake);
  }

  private save(): void {
    try {
      localStorage.setItem(this.key, JSON.stringify(this.list.flatMap((s, i) => (s.awake ? [i] : []))));
    } catch {
      // Non-essential.
    }
  }

  private bellTimer = 0;

  update(dt: number, wind: Vec3, windSpeed: number, cam: readonly number[]): void {
    // Wake: the wind passing through (low enough to touch them).
    this.list.forEach((s, i) => {
      const d = Math.hypot(s.x - wind[0], s.z - wind[2]);
      const low = wind[1] - s.y < (s.kind === "bells" ? 3.5 : 2.5);
      const reach = s.kind === "pinwheels" ? 4.5 : 3;
      if (d < reach && low) {
        for (const p of this.pieces) {
          if (p.secret !== i) continue;
          if (s.kind === "pinwheels") p.energy = Math.max(p.energy, 18 + windSpeed * 0.8 - Math.hypot(p.pos[0] - wind[0], p.pos[2] - wind[2]) * 2);
          else p.energy = Math.max(p.energy, 0.45);
        }
        if (!s.awake) {
          s.awake = true;
          this.save();
          this.onWake?.(s, i);
        }
      }
    });
    // Ease back toward a gentle stir from the breeze (a woken secret keeps a little more).
    let o = 0;
    let whirr = 0;
    this.bellTimer -= dt;
    for (const p of this.pieces) {
      const s = this.list[p.secret];
      const rest = p.variant === 3 ? (s.awake ? 0.12 : 0.06) : s.awake ? 2.2 : 0.5;
      p.energy = rest + (p.energy - rest) * Math.exp(-dt * (p.variant === 3 ? 0.35 : 0.45));
      if (p.variant !== 3) {
        p.spin += p.energy * dt;
        if (Math.hypot(p.pos[0] - cam[0], p.pos[2] - cam[2]) < 30) whirr = Math.max(whirr, (p.energy - rest) / 20);
      } else if (this.bellTimer <= 0 && p.energy > 0.1 && Math.hypot(p.pos[0] - cam[0], p.pos[2] - cam[2]) < 60) {
        // Bells ring as they swing: more often the harder they swing.
        if (Math.random() < p.energy * 1.4) this.onBell?.(p.pos, Math.min(1, p.energy * 2), Math.floor(Math.random() * 3));
      }
      this.data.set([p.pos[0], p.pos[1], p.pos[2], p.yaw, p.spin, p.variant === 3 ? p.energy : 0, p.scale, p.seed], o);
      o += 8;
    }
    if (this.bellTimer <= 0) this.bellTimer = 0.18 + Math.random() * 0.25;
    if (whirr > 0.02) {
      const near = this.pieces.find((p) => p.variant !== 3 && Math.hypot(p.pos[0] - cam[0], p.pos[2] - cam[2]) < 30);
      if (near) this.onWhirr?.(near.pos, Math.min(1, whirr));
    }
    if (o) this.buffer.write(this.data.subarray(0, o));
    let g = 0;
    for (const s of this.list) {
      this.glintData.set([s.x, s.y + (s.kind === "bells" ? 2.1 : 2.0), s.z, s.awake ? 0 : 1], g);
      g += 4;
    }
    for (const e of this.extra.slice(0, EXTRA)) {
      this.glintData.set(e, g);
      g += 4;
    }
    this.glintCount = g / 4;
    if (g) this.glintBuf.write(this.glintData.subarray(0, g));
  }

  private extra: [number, number, number, number][] = [];
  private glintCount = 0;

  /** Other places to glow this frame: (x, y, z, strength). */
  setExtraGlints(points: [number, number, number, number][]): void {
    this.extra = points;
  }

  encode(pass: FramePass): void {
    this.draws.forEach((d, i) => {
      const r = this.ranges[i];
      if (r && r.count) pass.draw(d, { instances: r.count, firstInstance: r.first });
    });
  }

  encodeGlints(pass: FramePass): void {
    if (this.glintCount) pass.draw(this.glintDraw, { instances: this.glintCount });
  }
}
