import { draw, geometry, sampler, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import plantShader from "../shaders/plants.wgsl";
import { loadTexture } from "../engine/textures";
import { BED_CELL, bedInCell, bedInside, type BedInfo } from "./bed-shape";
import { terrainHeightM as terrainHeight } from "./height";

const SPECIES = ["gazania", "ursinia", "empodium", "dandelion", "heliophila"];
const SPACING: Record<string, number> = { gazania: 0.55, ursinia: 0.55, empodium: 0.5, dandelion: 0.5, heliophila: 0.9 };
const LOAD_RADIUS = 170;
const DRAW_RADIUS = 140;
const MAX_PLANTS = 16384;
const STRIDE = 12;

interface Variant {
  firstIndex: number;
  indexCount: number;
  radius: number;
}

interface PlantSpecies {
  name: string;
  variants: Variant[];
  draws: Draw[];
  buffer: StorageBuffer;
  data: Float32Array<ArrayBuffer>;
  /** Per variant: instance count written this frame and its first instance. */
  counts: number[];
  firsts: number[];
}

interface PlantInstance {
  x: number;
  y: number;
  z: number;
  scale: number;
  yaw: number;
  seed: number;
  variant: number;
}

interface LoadedBed {
  info: BedInfo;
  plants: PlantInstance[];
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

/** Wildflower beds: Poly Haven ground-cover plants scattered inside deterministic clearings. */
export class FlowerBeds {
  private readonly beds = new Map<string, LoadedBed>();
  private loadedCell = "";

  private constructor(private readonly species: PlantSpecies[]) {}

  static async load(gpu: Gpu, globals: SharedUniforms, life: StorageBuffer, base = "/assets/plants"): Promise<FlowerBeds> {
    const samp = sampler(gpu, {
      minFilter: "linear",
      magFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
      maxAnisotropy: 4,
    });
    const species = await Promise.all(
      SPECIES.map(async (name): Promise<PlantSpecies> => {
        const dir = `${base}/${name}`;
        const [manifest, bin, albedoTex] = await Promise.all([
          fetch(`${dir}/plants.json`).then((r) => r.json()),
          fetch(`${dir}/plants.bin`).then((r) => r.arrayBuffer()),
          loadTexture(gpu, `${dir}/albedo.png`, { srgb: true }),
        ]);
        const geo = geometry(gpu, {
          label: `plants-${name}`,
          buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
          indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
        });
        const buffer = storage(gpu, MAX_PLANTS * STRIDE * 4, "read");
        const variants = manifest.variants as Variant[];
        const draws = variants.map((v, i) =>
          draw(gpu, {
            label: `plants-${name}-${i}`,
            shader: plantShader,
            geometry: geo.slice({ firstIndex: v.firstIndex, indexCount: v.indexCount }),
            depth: { compare: "greater" },
            multisample: { alphaToCoverage: true },
            set: { G: globals, plants: buffer, samp, albedoTex, life, P: { dryTint: 1, nightGlow: 1, pad0: 0, pad1: 0 } },
          }),
        );
        return { name, variants, draws, buffer, data: new Float32Array(MAX_PLANTS * STRIDE), counts: variants.map(() => 0), firsts: variants.map(() => 0) };
      }),
    );
    return new FlowerBeds(species);
  }

  get draws(): Draw[] {
    return this.species.flatMap((s) => s.draws);
  }

  /** Bed under a point (used to spread restoration and gameplay hints). */
  bedsNear(x: number, z: number, radius: number): BedInfo[] {
    return [...this.beds.values()].map((b) => b.info).filter((b) => Math.hypot(b.x - x, b.z - z) < radius + b.radius);
  }

  private populate(info: BedInfo): LoadedBed {
    const sp = this.species[info.species];
    const spacing = SPACING[sp.name];
    const r = rand(info.cx * 92821 + info.cz * 68917 + 17);
    const plants: PlantInstance[] = [];
    const ext = info.radius * 1.7;
    for (let gz = -ext; gz <= ext; gz += spacing) {
      for (let gx = -ext; gx <= ext; gx += spacing) {
        const x = info.x + gx + (r() - 0.5) * spacing * 0.9;
        const z = info.z + gz + (r() - 0.5) * spacing * 0.9;
        const inside = bedInside(info, x, z);
        // Thin toward the edge, with a few strays beyond it, so the bed feathers into the field.
        const stray = Math.max(0, 1 - Math.hypot(x - info.x, z - info.z) / (info.radius * 1.7)) * 0.08;
        if (r() > inside * 1.1 + stray) continue;
        plants.push({
          x,
          y: terrainHeight(x, z) - 0.02,
          z,
          scale: (0.85 + r() * 0.5) * (0.75 + 0.35 * inside),
          yaw: r() * Math.PI * 2,
          seed: r(),
          variant: Math.floor(r() * sp.variants.length),
        });
      }
    }
    return { info, plants };
  }

  private stream(px: number, pz: number): void {
    const cx = Math.floor(px / BED_CELL);
    const cz = Math.floor(pz / BED_CELL);
    const key = `${cx},${cz}`;
    if (key === this.loadedCell) return;
    this.loadedCell = key;
    const r = Math.ceil(LOAD_RADIUS / BED_CELL);
    const keep = new Set<string>();
    for (let dz = -r; dz <= r; dz++) {
      for (let dx = -r; dx <= r; dx++) {
        const k = `${cx + dx},${cz + dz}`;
        const info = bedInCell(cx + dx, cz + dz);
        if (!info) continue;
        keep.add(k);
        if (!this.beds.has(k)) this.beds.set(k, this.populate(info));
      }
    }
    for (const k of this.beds.keys()) if (!keep.has(k)) this.beds.delete(k);
  }

  update(cam: readonly number[], frustum: Float32Array, drawRadius = DRAW_RADIUS): void {
    this.stream(cam[0], cam[2]);
    // Bucket visible plants by species, then by variant, so each variant draws a contiguous range.
    const buckets = this.species.map((s) => s.variants.map((): PlantInstance[] => []));
    for (const bed of this.beds.values()) {
      const b = bed.info;
      const dist = Math.hypot(b.x - cam[0], b.z - cam[2]);
      if (dist - b.radius * 1.4 > drawRadius) continue;
      const y = terrainHeight(b.x, b.z);
      let visible = true;
      for (let i = 0; i < 6 && visible; i++) {
        const o = i * 4;
        if (frustum[o] * b.x + frustum[o + 1] * y + frustum[o + 2] * b.z + frustum[o + 3] < -b.radius * 1.5) visible = false;
      }
      if (!visible) continue;
      for (const p of bed.plants) buckets[b.species][p.variant].push(p);
    }
    this.species.forEach((s, si) => {
      let n = 0;
      buckets[si].forEach((list, vi) => {
        s.firsts[vi] = n;
        for (const p of list) {
          if (n >= MAX_PLANTS) break;
          const d = Math.hypot(p.x - cam[0], p.z - cam[2]);
          // Shrink plants away near the draw limit instead of popping.
          const fade = 1 - Math.min(1, Math.max(0, (d - (drawRadius - 15)) / 15));
          s.data.set([p.x, p.y, p.z, p.scale, Math.cos(p.yaw), Math.sin(p.yaw), p.seed, fade, 0, 1, 0, 0], n * STRIDE);
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
