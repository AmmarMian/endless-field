import { draw, geometry, sampler, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import plantShader from "../shaders/plants.wgsl";
import { loadTexture } from "../engine/textures";
import { ecology } from "./ecology";
import { biome, canopyLight } from "./biome";
import { sunflowerNear } from "./sunflower-field";
import { pathNear } from "./lantern-path";
import { riverBed, riverInfo, riverUpper, terrainHeightM, worldConstants } from "./height";

/** Per-species shading: dry tint before restoration, and night glow of non-leaf texels. */
interface SpeciesDef {
  name: string;
  dryTint: number;
  nightGlow: number;
  /** Max draw distance (m). */
  range: number;
}

const DEFS: SpeciesDef[] = [
  { name: "fern", dryTint: 0.9, nightGlow: 0, range: 110 },
  { name: "mushrooms", dryTint: 0, nightGlow: 1.6, range: 60 },
  { name: "logs", dryTint: 0, nightGlow: 0, range: 220 },
  { name: "rocks", dryTint: 0, nightGlow: 0, range: 260 },
  { name: "lilies", dryTint: 0, nightGlow: 0.9, range: 120 },
  { name: "irises", dryTint: 0.4, nightGlow: 0.6, range: 110 },
  { name: "riverrocks", dryTint: 0, nightGlow: 0, range: 220 },
];
const FERN = 0;
const MUSHROOMS = 1;
const LOGS = 2;
const ROCKS = 3;
const LILIES = 4;
const IRISES = 5;
const RIVERROCKS = 6;

const CELL = 24;
const LOAD_RADIUS = 280;
const MAX = 32768;
const STRIDE = 12;
const PER_FRAME = 10;

interface Variant {
  firstIndex: number;
  indexCount: number;
  radius: number;
}

interface Item {
  shade: number;
  species: number;
  variant: number;
  x: number;
  y: number;
  z: number;
  scale: number;
  yaw: number;
  seed: number;
}

interface Loaded {
  def: SpeciesDef;
  variants: Variant[];
  draws: Draw[];
  buffer: StorageBuffer;
  data: Float32Array<ArrayBuffer>;
  counts: number[];
  firsts: number[];
}

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

/**
 * Undergrowth placed by the ecological model: ferns in moist shade, mushrooms on the forest
 * floor, fallen logs in the forest, boulders on slopes and in the hills.
 */
export class Undergrowth {
  private readonly cells = new Map<string, Item[]>();
  private readonly pending: { key: string; cx: number; cz: number; d: number }[] = [];
  private loadedCell = "";
  private items: Item[] = [];

  private constructor(private readonly species: Loaded[]) {}

  static async load(gpu: Gpu, globals: SharedUniforms, life: StorageBuffer, base = "assets/plants"): Promise<Undergrowth> {
    const samp = sampler(gpu, { minFilter: "linear", magFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat", maxAnisotropy: 4 });
    const species = await Promise.all(
      DEFS.map(async (def): Promise<Loaded> => {
        const dir = `${base}/${def.name}`;
        const [manifest, bin, albedoTex] = await Promise.all([
          fetch(`${dir}/plants.json`).then((r) => r.json()),
          fetch(`${dir}/plants.bin`).then((r) => r.arrayBuffer()),
          loadTexture(gpu, `${dir}/albedo.png`, { srgb: true }),
        ]);
        const geo = geometry(gpu, {
          label: `undergrowth-${def.name}`,
          buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
          indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
        });
        const buffer = storage(gpu, MAX * STRIDE * 4, "read");
        const variants = manifest.variants as Variant[];
        const draws = variants.map((v, i) =>
          draw(gpu, {
            label: `undergrowth-${def.name}-${i}`,
            shader: plantShader,
            constants: worldConstants(plantShader),
            geometry: geo.slice({ firstIndex: v.firstIndex, indexCount: v.indexCount }),
            depth: { compare: "greater" },
            multisample: { alphaToCoverage: true },
            set: { G: globals, plants: buffer, samp, albedoTex, life, P: { dryTint: def.dryTint, nightGlow: def.nightGlow, pad0: 0, pad1: 0 } },
          }),
        );
        return { def, variants, draws, buffer, data: new Float32Array(MAX * STRIDE), counts: variants.map(() => 0), firsts: variants.map(() => 0) };
      }),
    );
    return new Undergrowth(species);
  }

  get draws(): Draw[] {
    return this.species.flatMap((s) => s.draws);
  }

  private variantCount(species: number): number {
    return this.species[species].variants.length;
  }

  private makeCell(cx: number, cz: number): Item[] {
    const r = rand(Math.imul(cx, 0x2c1b3c6d) ^ Math.imul(cz, 0x297a2d39) ^ 0x51ed27);
    const x0 = cx * CELL;
    const z0 = cz * CELL;
    const eco = ecology(x0 + CELL / 2, z0 + CELL / 2);
    const out: Item[] = [];
    const place = (species: number, x: number, z: number, scale: number, sink = 0, variant = -1, y?: number) => {
      if (y === undefined) {
        const [d, , hw] = riverInfo(x, z);
        if (d < hw * 1.3) return;
      }
      if (sunflowerNear(x, z, 2) || pathNear(x, z, 2.2)) return;
      out.push({
        shade: canopyLight(x, biome(x, z)[2]),
        species,
        variant: variant >= 0 ? variant : Math.floor(r() * this.variantCount(species)),
        x,
        y: (y ?? terrainHeightM(x, z)) - sink * scale,
        z,
        scale,
        yaw: r() * Math.PI * 2,
        seed: r(),
      });
    };
    this.riverLife(x0, z0, r, place);
    if (eco.temperature < 0.25) return out;
    const shade = eco.forest + Math.max(0, eco.moisture - 0.55) * 1.2;

    // Ferns: clumps in moist shade; dense on the forest floor.
    const fernClumps = Math.round(Math.min(1, shade) * 5 * (0.6 + r() * 0.8));
    for (let c = 0; c < fernClumps; c++) {
      const fx = x0 + r() * CELL;
      const fz = z0 + r() * CELL;
      const n = 3 + Math.floor(r() * 7);
      for (let i = 0; i < n; i++) {
        const a = r() * Math.PI * 2;
        const rr = Math.sqrt(r()) * 3.2;
        place(FERN, fx + Math.cos(a) * rr, fz + Math.sin(a) * rr, 0.7 + r() * 0.6);
      }
    }
    // Mushrooms: a few clusters on moist forest floor.
    if (eco.forest > 0.3 && eco.moisture > 0.5) {
      const n = Math.floor(r() * 3.2 * eco.forest);
      for (let i = 0; i < n; i++) place(MUSHROOMS, x0 + r() * CELL, z0 + r() * CELL, 0.8 + r() * 0.6);
    }
    // Fallen logs and stumps in the forest.
    if (eco.forest > 0.4 && r() < 0.32 * eco.forest) {
      place(LOGS, x0 + r() * CELL, z0 + r() * CELL, 0.8 + r() * 0.5, 0.05);
    }
    // Boulders: on slopes, high ground and in the forest; half-buried so they sit in the soil.
    const rockChance = 0.08 + eco.slope * 0.9 + Math.min(eco.elevation / 120, 0.4) + eco.forest * 0.12;
    if (r() < rockChance) {
      const n = 1 + Math.floor(r() * 2.5);
      for (let i = 0; i < n; i++) place(ROCKS, x0 + r() * CELL, z0 + r() * CELL, 0.5 + r() * 0.9 + eco.elevation / 200, 0.3);
    }
    return out;
  }

  /** Lilies on calm lowland water, irises along the waterline, boulders in the channel. */
  private riverLife(
    x0: number,
    z0: number,
    r: () => number,
    place: (species: number, x: number, z: number, scale: number, sink?: number, variant?: number, y?: number) => void,
  ): void {
    const [dc, , hwc] = riverInfo(x0 + CELL / 2, z0 + CELL / 2);
    if (dc > hwc * 2 + CELL) return;
    const tries = 26;
    for (let i = 0; i < tries; i++) {
      const x = x0 + r() * CELL;
      const z = z0 + r() * CELL;
      const [d, water, hw, px] = riverInfo(x, z);
      const k = d / hw;
      const upper = riverUpper(px);
      if (k < 0.95 && k > 0.45 && upper < 0.05 && r() < 0.35) {
        // Lily pads float where the current slackens near the banks.
        place(LILIES, x, z, 0.8 + r() * 0.5, 0, -1, water + 0.05);
      } else if (k > 0.85 && k < 1.45 && r() < 0.25) {
        place(IRISES, x, z, 0.8 + r() * 0.5, 0, -1, terrainHeightM(x, z) - 0.05);
      } else if (k < 1.1 && r() < (upper > 0.05 ? 0.2 : 0.05)) {
        // Boulders sit half-buried in the bed, breaking the surface in the mountain stream.
        const bed = riverBed(Math.min(d, hw), water, hw, upper);
        place(RIVERROCKS, x, z, 0.6 + r() * (upper > 0.05 ? 1.1 : 0.6), 0.35, -1, bed + 0.2);
      } else if (k > 1.0 && k < 1.8 && r() < 0.08) {
        place(RIVERROCKS, x, z, 0.4 + r() * 0.5, 0.3, -1, terrainHeightM(x, z));
      }
    }
  }

  private stream(px: number, pz: number): void {
    const cx = Math.floor(px / CELL);
    const cz = Math.floor(pz / CELL);
    const key = `${cx},${cz}`;
    if (key !== this.loadedCell) {
      this.loadedCell = key;
      const rad = Math.ceil(LOAD_RADIUS / CELL);
      const keep = new Set<string>();
      this.pending.length = 0;
      for (let dz = -rad; dz <= rad; dz++) {
        for (let dx = -rad; dx <= rad; dx++) {
          if (dx * dx + dz * dz > rad * rad) continue;
          const k = `${cx + dx},${cz + dz}`;
          keep.add(k);
          if (!this.cells.has(k)) this.pending.push({ key: k, cx: cx + dx, cz: cz + dz, d: dx * dx + dz * dz });
        }
      }
      for (const k of this.cells.keys()) if (!keep.has(k)) this.cells.delete(k);
      this.pending.sort((a, b) => b.d - a.d);
      this.items = [...this.cells.values()].flat();
    }
    let budget = this.cells.size === 0 ? 400 : PER_FRAME;
    let added = false;
    while (budget-- > 0 && this.pending.length) {
      const c = this.pending.pop()!;
      if (!this.cells.has(c.key)) this.cells.set(c.key, this.makeCell(c.cx, c.cz));
      added = true;
    }
    if (added) this.items = [...this.cells.values()].flat();
  }

  update(cam: readonly number[], frustum: Float32Array, distanceScale = 1): void {
    this.stream(cam[0], cam[2]);
    const buckets = this.species.map((s) => s.variants.map((): Item[] => []));
    for (const it of this.items) {
      const range = this.species[it.species].def.range * distanceScale;
      const d = Math.hypot(it.x - cam[0], it.z - cam[2]);
      if (d > range) continue;
      let visible = true;
      for (let i = 0; i < 6 && visible; i++) {
        const o = i * 4;
        if (frustum[o] * it.x + frustum[o + 1] * it.y + frustum[o + 2] * it.z + frustum[o + 3] < -3 * it.scale) visible = false;
      }
      if (visible) buckets[it.species][it.variant].push(it);
    }
    this.species.forEach((s, si) => {
      const range = s.def.range * distanceScale;
      let n = 0;
      buckets[si].forEach((list, vi) => {
        s.firsts[vi] = n;
        for (const it of list) {
          if (n >= MAX) break;
          const d = Math.hypot(it.x - cam[0], it.z - cam[2]);
          const fade = Math.min(1, Math.max(0, (range - d) / (range * 0.2)));
          s.data.set([it.x, it.y, it.z, it.scale, Math.cos(it.yaw), Math.sin(it.yaw), it.seed, fade, 0, it.shade, 0, 0], n * STRIDE);
          n++;
        }
        s.counts[vi] = n - s.firsts[vi];
      });
      if (n) s.buffer.write(s.data.subarray(0, n * STRIDE));
    });
  }

  encode(pass: FramePass): void {
    for (const s of this.species) {
      s.draws.forEach((d, vi) => {
        if (s.counts[vi] > 0) pass.draw(d, { instances: s.counts[vi], firstInstance: s.firsts[vi] });
      });
    }
  }
}
