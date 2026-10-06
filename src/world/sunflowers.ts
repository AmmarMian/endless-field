import { compute, draw, geometry, sampler, storage, type Compute, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import simShader from "../shaders/sunflower-sim.wgsl";
import renderShader from "../shaders/sunflower.wgsl";
import { loadTexture } from "../engine/textures";
import { terrainBase, terrainHeightM as terrainHeight } from "./height";

import { SUNFLOWERS, sunflowerField } from "./sunflower-field";

export { SUNFLOWERS, sunflowerField, sunflowerNear } from "./sunflower-field";

/**
 * Grades the field for the current world: the level is the field's mean natural ground
 * height, so leveling cuts and fills evenly. Call after setWorldSeed, before heights are used.
 */
export function gradeSunflowerField(): void {
  let sum = 0;
  let weight = 0;
  const reach = Math.max(SUNFLOWERS.rx, SUNFLOWERS.rz) * 1.25;
  for (let x = SUNFLOWERS.x - reach; x <= SUNFLOWERS.x + reach; x += 4) {
    for (let z = SUNFLOWERS.z - reach; z <= SUNFLOWERS.z + reach; z += 4) {
      const f = sunflowerField(x, z);
      if (f <= 0) continue;
      sum += terrainBase(x, z) * f;
      weight += f;
    }
  }
  SUNFLOWERS.level = weight ? sum / weight : 0;
}

interface Manifest {
  vertexBytes: number;
  indexCount: number;
  variants: { height: number; pivot: number[]; lods: { firstIndex: number; indexCount: number }[] }[];
}

/** LOD switch distances (m) and the dithered crossfade band beyond each. */
const LOD = [8, 28, 70];
const BAND = 0.2;
const RANGE = 420;
const PLANT_FLOATS = 8;

function rand(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface PlantRec {
  x: number;
  y: number;
  z: number;
  variant: number;
}

/**
 * A planted sunflower field: Blender-modeled plants (tools/blender/model_sunflower.py) in rows,
 * each stem simulated on the GPU as a damped cantilever under aerodynamic drag.
 */
export class Sunflowers {
  readonly draws: Draw[];
  private readonly sim: Compute;
  private readonly visibleData: Uint32Array<ArrayBuffer>;
  private readonly lists: number[][];
  private readonly ranges: { first: number; count: number }[];
  /** Plants bucketed into 12 m chunks so whole chunks are culled at once. */
  private readonly chunks: { x: number; z: number; y: number; ids: number[] }[] = [];

  private constructor(
    private readonly manifest: Manifest,
    private readonly plants: PlantRec[],
    draws: Draw[],
    sim: Compute,
    private readonly visible: StorageBuffer,
  ) {
    this.draws = draws;
    this.sim = sim;
    this.visibleData = new Uint32Array(plants.length * 2 * 2);
    this.lists = draws.map(() => []);
    this.ranges = draws.map(() => ({ first: 0, count: 0 }));
    const byKey = new Map<string, { x: number; z: number; y: number; ids: number[] }>();
    plants.forEach((p, i) => {
      const key = `${Math.floor(p.x / 12)},${Math.floor(p.z / 12)}`;
      let c = byKey.get(key);
      if (!c) byKey.set(key, (c = { x: (Math.floor(p.x / 12) + 0.5) * 12, z: (Math.floor(p.z / 12) + 0.5) * 12, y: p.y, ids: [] }));
      c.ids.push(i);
    });
    this.chunks = [...byKey.values()];
  }

  static async load(gpu: Gpu, globals: SharedUniforms, base = "assets/sunflower"): Promise<Sunflowers> {
    const [manifest, bin, albedoTex] = await Promise.all([
      fetch(`${base}/sunflower.json`).then((r) => r.json() as Promise<Manifest>),
      fetch(`${base}/sunflower.bin`).then((r) => r.arrayBuffer()),
      loadTexture(gpu, `${base}/albedo.png`, { srgb: true }),
    ]);
    const geo = geometry(gpu, {
      label: "sunflower",
      buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
    });

    const plants = Sunflowers.plant();
    const plantData = new Float32Array(plants.length * PLANT_FLOATS);
    const r = rand(1234);
    plants.forEach((p, i) => {
      // Mature heads face east, away from the evening sun; each a little differently.
      const yaw = Math.atan2(-0.36, 0.93) + (r() - 0.5) * 0.7;
      const scale = (p as PlantRec & { scale: number }).scale;
      // Natural frequency of the first bending mode: tall, heavy-headed stems sway slower.
      const freq = (0.62 + r() * 0.22) / Math.sqrt(scale);
      plantData.set([p.x, p.y, p.z, scale, yaw, p.variant, r(), freq], i * PLANT_FLOATS);
    });
    const plantBuf = storage(gpu, plantData.byteLength, "read");
    plantBuf.write(plantData);
    const stateBuf = storage(gpu, plants.length * 32, "read-write");
    stateBuf.write(new Float32Array(plants.length * 8));
    const visible = storage(gpu, plants.length * 2 * 8, "read");
    const variantData = new Float32Array(manifest.variants.flatMap((v) => [v.pivot[0], v.pivot[1], v.pivot[2], v.height]));
    const variantBuf = storage(gpu, variantData.byteLength, "read");
    variantBuf.write(variantData);
    const samp = sampler(gpu, { minFilter: "linear", magFilter: "linear", mipmapFilter: "linear", maxAnisotropy: 4 });

    const sim = compute(gpu, simShader, {
      label: "sunflower-sim",
      set: { G: globals, plants: plantBuf, state: stateBuf, S: { count: plants.length, dt: 1 / 60, pad0: 0, pad1: 0 } },
    });
    const draws: Draw[] = [];
    for (const v of manifest.variants) {
      for (const l of v.lods) {
        draws.push(
          draw(gpu, {
            label: "sunflower",
            shader: renderShader,
            geometry: geo.slice({ firstIndex: l.firstIndex, indexCount: l.indexCount }),
            depth: { compare: "greater" },
            multisample: { alphaToCoverage: true },
            set: { G: globals, plants: plantBuf, state: stateBuf, visible, variants: variantBuf, samp, albedoTex },
          }),
        );
      }
    }
    return new Sunflowers(manifest, plants, draws, sim, visible);
  }

  /** Lays out the rows: real fields are drilled ~0.75 m apart, plants every ~0.4 m. */
  private static plant(): (PlantRec & { scale: number })[] {
    const r = rand(77);
    const out: (PlantRec & { scale: number })[] = [];
    const [ux, uz] = SUNFLOWERS.dir;
    for (let b = -SUNFLOWERS.rz * 1.1; b <= SUNFLOWERS.rz * 1.1; b += SUNFLOWERS.row) {
      // Drill rows wander slightly.
      const rowOff = (r() - 0.5) * 0.08;
      for (let a = -SUNFLOWERS.rx * 1.1; a <= SUNFLOWERS.rx * 1.1; a += 0.36 + r() * 0.12) {
        const bb = b + rowOff + (r() - 0.5) * 0.1 + Math.sin(a * 0.05 + b) * 0.05;
        const x = SUNFLOWERS.x + a * ux - bb * uz;
        const z = SUNFLOWERS.z + a * uz + bb * ux;
        const f = sunflowerField(x, z);
        if (f < 0.5 || r() < 0.05) continue;
        // Edge plants get more light (and more room) but suffer more wind: shorter, varied.
        const vigor = 0.88 + 0.22 * r() - (1 - f) * 0.3 + 0.08 * Math.sin(a * 0.11) * Math.cos(b * 0.17);
        const young = r() < 0.07;
        const variant = young ? 3 : Math.min(2, Math.floor(r() * 3 * (0.6 + 0.4 * vigor)));
        out.push({ x, y: terrainHeight(x, z) - 0.03, z, variant, scale: Math.max(0.7, vigor) });
      }
    }
    return out;
  }

  get count(): number {
    return this.plants.length;
  }

  /** Steps the stem physics and picks detail levels for this frame. */
  update(dt: number, cam: readonly number[], frustum: Float32Array): void {
    this.sim.set({ S: { dt } });
    this.sim.dispatch(Math.ceil(this.plants.length / 64));
    for (const l of this.lists) l.length = 0;
    const lods = this.manifest.variants[0].lods.length;
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    const bits = (w: number) => {
      f32[0] = w;
      return u32[0];
    };
    const ONE = bits(1);
    const visit = (i: number) => {
      const p = this.plants[i];
      const cy = p.y + 1;
      const d = Math.hypot(p.x - cam[0], cy - cam[1], p.z - cam[2]);
      if (d > RANGE) return;
      const base = p.variant * lods;
      const put = (lod: number, w: number) => this.lists[base + lod].push(i, w === 1 ? ONE : bits(w));
      // Dithered crossfade over a band beyond each switch distance.
      let lod = 0;
      while (lod < LOD.length && d >= LOD[lod] * (1 + BAND)) lod++;
      if (lod < LOD.length && d >= LOD[lod]) {
        const k = (d - LOD[lod]) / (LOD[lod] * BAND);
        put(lod, 1 - k);
        put(lod + 1, -(1 - k));
      } else put(lod, 1);
    };
    for (const c of this.chunks) {
      if (Math.hypot(c.x - cam[0], c.z - cam[2]) > RANGE + 10) continue;
      const cy = c.y + 1;
      let inside = true;
      for (let k = 0; k < 6 && inside; k++) {
        const o = k * 4;
        if (frustum[o] * c.x + frustum[o + 1] * cy + frustum[o + 2] * c.z + frustum[o + 3] < -10) inside = false;
      }
      if (inside) for (const i of c.ids) visit(i);
    }
    let first = 0;
    this.lists.forEach((l, i) => {
      this.visibleData.set(l, first * 2);
      this.ranges[i].first = first;
      this.ranges[i].count = l.length / 2;
      first += l.length / 2;
    });
    if (first) this.visible.write(this.visibleData.subarray(0, first * 2));
  }

  /** Debug: skip a LOD level (profiling). */
  skip = [false, false, false, false];

  encode(pass: FramePass): void {
    this.draws.forEach((d, i) => {
      const r = this.ranges[i];
      if (r.count && !this.skip[i % 4]) pass.draw(d, { instances: r.count, firstInstance: r.first });
    });
  }
}
