import { draw, geometry, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import flowerShader from "../shaders/flower.wgsl";
import glowShader from "../shaders/flower-glow.wgsl";
import { mountainHeight, terrainHeightM as terrainHeight, worldConstants } from "./height";
import { sunflowerNear } from "./sunflower-field";
import { pathNear } from "./lantern-path";
import { biome } from "./biome";

const CELL = 46;
const LOAD_RADIUS = 320;
const MAX_FLOWERS = 4096;
const FLOAT_STRIDE = 12;

export const PALETTES: [number, number, number][] = [
  [0.86, 0.07, 0.05], // poppy red
  [0.93, 0.91, 0.86], // white
  [0.96, 0.62, 0.04], // gold
  [0.36, 0.24, 0.86], // violet
  [0.96, 0.34, 0.56], // pink
  [0.97, 0.36, 0.06], // orange
];

export interface Flower {
  id: string;
  x: number;
  y: number;
  z: number;
  color: [number, number, number];
  seed: number;
  height: number;
  petalLength: number;
  petals: number;
  bloom: number;
  bloomed: boolean;
  cluster: Cluster;
}

export interface Cluster {
  key: string;
  x: number;
  z: number;
  flowers: Flower[];
  complete: boolean;
}

/** Deterministic per-cell random stream (mulberry32 seeded by the cell). */
function cellRandom(cx: number, cz: number): () => number {
  let s = (Math.imul(cx, 73856093) ^ Math.imul(cz, 19349663) ^ 0x5bd1e995) >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Builds the shared template: stem ribbon, two leaves, 8 petal slots and a center fan. */
function buildTemplate(): Float32Array<ArrayBuffer> {
  const out: number[] = [];
  const v = (part: number, a: number, b: number, c: number) => out.push(part, a, b, c);
  const grid = (part: number, idx: number, nu: number, nv: number, u0 = 0, v0 = -1, v1 = 1) => {
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        const ua = u0 + ((1 - u0) * i) / nu;
        const ub = u0 + ((1 - u0) * (i + 1)) / nu;
        const va = v0 + ((v1 - v0) * j) / nv;
        const vb = v0 + ((v1 - v0) * (j + 1)) / nv;
        v(part, ua, va, idx); v(part, ub, va, idx); v(part, ua, vb, idx);
        v(part, ua, vb, idx); v(part, ub, va, idx); v(part, ub, vb, idx);
      }
    }
  };
  grid(0, 0, 8, 1); // stem
  for (let l = 0; l < 2; l++) grid(3, l, 5, 2); // leaves
  for (let p = 0; p < 8; p++) grid(1, p, 5, 2); // petals
  const fan = 10;
  for (let i = 0; i < fan; i++) {
    const a0 = (i / fan) * Math.PI * 2;
    const a1 = ((i + 1) / fan) * Math.PI * 2;
    v(2, 0, 0, 0); v(2, 1, a1, 0); v(2, 1, a0, 0);
  }
  return new Float32Array(out);
}

export class Flowers {
  private readonly clusters = new Map<string, Cluster>();
  private readonly bloomedIds = new Set<string>();
  private readonly instances: StorageBuffer;
  private readonly data = new Float32Array(MAX_FLOWERS * FLOAT_STRIDE);
  private readonly drawCall: Draw;
  private readonly glowCall: Draw;
  private loadedCell = "";
  private active: Flower[] = [];
  bloomedCount = 0;

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.instances = storage(gpu, MAX_FLOWERS * FLOAT_STRIDE * 4, "read");
    this.drawCall = draw(gpu, {
      label: "flowers",
      shader: flowerShader,
      constants: worldConstants(flowerShader),
      geometry: geometry(gpu, { buffers: [{ data: buildTemplate(), attributes: { a: "float32x4" } }] }),
      depth: { compare: "greater" },
      set: { G: globals, flowers: this.instances },
    });
    this.glowCall = draw(gpu, {
      label: "flower-glow",
      shader: glowShader,
      constants: worldConstants(glowShader),
      vertices: 6,
      blend: "additive",
      depth: { compare: "greater", write: false },
      set: { G: globals, flowers: this.instances },
    });
  }

  get draws(): Draw[] {
    return [this.drawCall, this.glowCall];
  }

  get loaded(): readonly Flower[] {
    return this.active;
  }

  private makeCluster(cx: number, cz: number): Cluster {
    const key = `${cx},${cz}`;
    const rnd = cellRandom(cx, cz);
    const cluster: Cluster = { key, x: 0, z: 0, flowers: [], complete: false };
    const roll = rnd();
    // Keep the spawn area populated so the first minute always has something to find.
    const nearSpawn = Math.abs(cx) <= 1 && Math.abs(cz) <= 1;
    const [grove, meadow, forest] = biome((cx + 0.5) * CELL, (cz + 0.5) * CELL);
    if (roll > 0.45 + meadow * 0.25 - grove * 0.15 - forest * 0.2 && !nearSpawn) return cluster;
    cluster.x = (cx + 0.15 + rnd() * 0.7) * CELL;
    cluster.z = (cz + 0.15 + rnd() * 0.7) * CELL;
    if (mountainHeight(cluster.x, cluster.z) > 150) return cluster;
    if (sunflowerNear(cluster.x, cluster.z, 10) || pathNear(cluster.x, cluster.z, 6)) return cluster;
    const count = 5 + Math.floor(rnd() * 10);
    const color = PALETTES[Math.floor(rnd() * PALETTES.length)];
    const spread = 4 + rnd() * 6;
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < count; i++) {
      const r = spread * Math.sqrt((i + 0.5) / count) * (0.75 + rnd() * 0.5);
      const a = i * golden + rnd() * 0.6;
      const x = cluster.x + Math.cos(a) * r;
      const z = cluster.z + Math.sin(a) * r;
      const id = `${key}:${i}`;
      const bloomed = this.bloomedIds.has(id);
      const tint = 0.85 + rnd() * 0.3;
      cluster.flowers.push({
        id,
        x,
        y: terrainHeight(x, z),
        z,
        color: [color[0] * tint, color[1] * tint, color[2] * tint].map((c) => Math.min(c, 1)) as [number, number, number],
        seed: rnd(),
        height: 1.05 + rnd() * 0.45,
        petalLength: 0.2 + rnd() * 0.09,
        petals: 5 + Math.floor(rnd() * 4),
        bloom: bloomed ? 1 : 0,
        bloomed,
        cluster,
      });
    }
    cluster.complete = cluster.flowers.every((f) => f.bloomed);
    return cluster;
  }

  /** Streams clusters in and out around the player. */
  private stream(px: number, pz: number): void {
    const cx = Math.floor(px / CELL);
    const cz = Math.floor(pz / CELL);
    const cellKey = `${cx},${cz}`;
    if (cellKey === this.loadedCell) return;
    this.loadedCell = cellKey;
    const r = Math.ceil(LOAD_RADIUS / CELL);
    const keep = new Set<string>();
    for (let dz = -r; dz <= r; dz++) {
      for (let dx = -r; dx <= r; dx++) {
        const key = `${cx + dx},${cz + dz}`;
        keep.add(key);
        if (!this.clusters.has(key)) this.clusters.set(key, this.makeCluster(cx + dx, cz + dz));
      }
    }
    for (const key of this.clusters.keys()) if (!keep.has(key)) this.clusters.delete(key);
    this.active = [];
    for (const c of this.clusters.values()) for (const f of c.flowers) this.active.push(f);
    if (this.active.length > MAX_FLOWERS) this.active.length = MAX_FLOWERS;
  }

  /**
   * Advances bloom animations and returns flowers newly touched by the stream
   * (within `radius` of the leader).
   */
  update(dt: number, px: number, py: number, pz: number, radius: number): Flower[] {
    this.stream(px, pz);
    const touched: Flower[] = [];
    const r2 = radius * radius;
    const d = this.data;
    let o = 0;
    for (const f of this.active) {
      if (!f.bloomed) {
        const dx = f.x - px;
        const dy = f.y + f.height - py;
        const dz = f.z - pz;
        if (dx * dx + dy * dy * 0.5 + dz * dz < r2) {
          f.bloomed = true;
          this.bloomedIds.add(f.id);
          this.bloomedCount++;
          touched.push(f);
        }
      }
      if (f.bloomed && f.bloom < 1) f.bloom = Math.min(1, f.bloom + dt / 1.3);
      d[o++] = f.x; d[o++] = f.y; d[o++] = f.z; d[o++] = f.bloom;
      d[o++] = f.color[0]; d[o++] = f.color[1]; d[o++] = f.color[2]; d[o++] = f.seed;
      d[o++] = f.height; d[o++] = f.petalLength; d[o++] = 0.9; d[o++] = f.petals;
    }
    if (o > 0) this.instances.write(d.subarray(0, o));
    return touched;
  }

  /** Clusters whose every flower has now bloomed (reported once). */
  completedClusters(touched: Flower[]): Cluster[] {
    const done: Cluster[] = [];
    for (const f of touched) {
      const c = f.cluster;
      if (!c.complete && c.flowers.every((x) => x.bloomed)) {
        c.complete = true;
        done.push(c);
      }
    }
    return done;
  }

  /** Nearest unbloomed flower, used to give attract-mode and hints a direction. */
  nearestClosed(px: number, pz: number): Flower | null {
    let best: Flower | null = null;
    let bd = Infinity;
    for (const f of this.active) {
      if (f.bloomed) continue;
      const d = (f.x - px) ** 2 + (f.z - pz) ** 2;
      if (d < bd) {
        bd = d;
        best = f;
      }
    }
    return best;
  }

  encode(pass: FramePass): void {
    if (this.active.length) pass.draw(this.drawCall, { instances: this.active.length });
  }

  /** Additive halos; encode after all opaque geometry. */
  encodeGlow(pass: FramePass): void {
    if (this.active.length) pass.draw(this.glowCall, { instances: this.active.length });
  }
}
