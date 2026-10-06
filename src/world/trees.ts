import {
  draw,
  geometry,
  sampler,
  storage,
  type Draw,
  type FramePass,
  type Gpu,
  type SharedUniforms,
  type StorageBuffer,
  type Texture,
} from "vgpu";
import treeShader from "../shaders/tree.wgsl";
import impostorShader from "../shaders/tree-impostor.wgsl";
import { loadTexture } from "../engine/textures";
import { riverInfo, terrainHeightM as terrainHeight } from "./height";
import { ecology, treeDensity, treeSuitability, type Eco } from "./ecology";

interface LodInfo {
  file: string;
  vertexCount: number;
  vertexBytes: number;
  indexCount: number;
  groups: Record<"bark" | "leaves", { firstIndex: number; indexCount: number }>;
}

interface TreeManifest {
  name: string;
  height: number;
  canopy: { center: number[]; radius: number };
  lods: LodInfo[];
  lodDistances: [number, number];
  impostor: { frames: number; cols: number; rows: number; tile: number; size: number; centerY: number };
  textures: Record<string, string>;
}

interface Species {
  manifest: TreeManifest;
  /** Instance buffers for LOD0, LOD1, impostor. */
  buffers: StorageBuffer[];
  data: Float32Array<ArrayBuffer>[];
  counts: number[];
  meshDraws: { bark: Draw; leaves: Draw }[];
  impostor: Draw;
  scale: [number, number];
}

interface TreeInstance {
  species: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  scale: number;
  seed: number;
}

export const SPECIES = [
  { name: "jacaranda", scale: [0.5, 0.78] as [number, number] },
  { name: "island", scale: [2.0, 2.8] as [number, number] },
  { name: "spruce", scale: [0.7, 1.15] as [number, number] },
];

const CELL = 72;
const VIEW = 900;
/** Trees grow in over this band at the edge of the view, inside the fog. */
const FADE = 150;
/** Cells generated per frame; the rest queue so fast flight never hitches. */
const CELLS_PER_FRAME = 6;
const MAX_INSTANCES = 8192;
const STRIDE = 8;

function cellRandom(cx: number, cz: number): () => number {
  let s = (Math.imul(cx, 0x27d4eb2d) ^ Math.imul(cz, 0x165667b1) ^ 0x2545f491) >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sparse lone trees with three LODs: full mesh, reduced mesh, impostor billboard. */
export class Trees {
  private constructor(
    private readonly species: Species[],
    private readonly cells: Map<string, TreeInstance[]>,
  ) {}

  private loadedCell = "";
  private instances: TreeInstance[] = [];

  static async load(gpu: Gpu, globals: SharedUniforms, base = "/assets/trees"): Promise<Trees> {
    const linear = sampler(gpu, {
      minFilter: "linear",
      magFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
      maxAnisotropy: 8,
    });
    const clampLinear = sampler(gpu, {
      minFilter: "linear",
      magFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    const species = await Promise.all(
      SPECIES.map(async (spec): Promise<Species> => {
        const dir = `${base}/${spec.name}`;
        const manifest = (await (await fetch(`${dir}/tree.json`)).json()) as TreeManifest;
        const tex = async (key: string, srgb: boolean): Promise<Texture> => loadTexture(gpu, `${dir}/${manifest.textures[key]}`, { srgb });
        const [trunkDiff, trunkNor, branchesDiff, branchesNor, leaves, impAlbedo, impNormal, ...bins] = await Promise.all([
          tex("trunk_diff", true),
          tex("trunk_nor", false),
          tex("branches_diff", true),
          tex("branches_nor", false),
          tex("leaves", true),
          loadTexture(gpu, `${dir}/impostor_albedo.png`, { srgb: true }),
          loadTexture(gpu, `${dir}/impostor_normal.png`, { srgb: false }),
          ...manifest.lods.map(async (l) => (await fetch(`${dir}/${l.file}`)).arrayBuffer()),
        ]);
        const buffers = [0, 1, 2].map(() => storage(gpu, MAX_INSTANCES * STRIDE * 4, "read"));
        const leafTint = spec.name === "jacaranda" ? 1.0 : 1.05;
        const meshDraws = manifest.lods.map((lod, li) => {
          const bin = bins[li] as ArrayBuffer;
          const geo = geometry(gpu, {
            label: `${spec.name}-lod${li}`,
            buffers: [
              {
                data: new Uint8Array(bin, 0, lod.vertexBytes),
                attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" },
              },
            ],
            indices: new Uint32Array(bin, lod.vertexBytes, lod.indexCount),
          });
          const set = {
            G: globals,
            trees: buffers[li],
            tree: { height: manifest.height, leafTint, pad0: 0, pad1: 0 },
            samp: linear,
            trunkDiff,
            trunkNor,
            branchesDiff,
            branchesNor,
            leaves,
          };
          const g = lod.groups;
          return {
            bark: draw(gpu, {
              label: `${spec.name}-lod${li}-bark`,
              shader: treeShader,
              geometry: geo.slice({ firstIndex: g.bark.firstIndex, indexCount: g.bark.indexCount }),
              entry: { fragment: "fs_bark" },
              depth: { compare: "greater" },
              set,
            }),
            leaves: draw(gpu, {
              label: `${spec.name}-lod${li}-leaves`,
              shader: treeShader,
              geometry: geo.slice({ firstIndex: g.leaves.firstIndex, indexCount: g.leaves.indexCount }),
              entry: { fragment: "fs_leaves" },
              depth: { compare: "greater" },
              multisample: { alphaToCoverage: true },
              set,
            }),
          };
        });
        const im = manifest.impostor;
        const impostor = draw(gpu, {
          label: `${spec.name}-impostor`,
          shader: impostorShader,
          vertices: 6,
          depth: { compare: "greater" },
          multisample: { alphaToCoverage: true },
          set: {
            G: globals,
            trees: buffers[2],
            imp: { frames: im.frames, cols: im.cols, rows: im.rows, size: im.size, centerY: im.centerY, height: manifest.height, leafTint, pad: 0 },
            samp: clampLinear,
            albedoAtlas: impAlbedo,
            normalAtlas: impNormal,
          },
        });
        return {
          manifest,
          buffers,
          data: [0, 1, 2].map(() => new Float32Array(MAX_INSTANCES * STRIDE)),
          counts: [0, 0, 0],
          meshDraws,
          impostor,
          scale: spec.scale,
        };
      }),
    );
    return new Trees(species, new Map());
  }

  get draws(): Draw[] {
    return this.species.flatMap((s) => [...s.meshDraws.flatMap((d) => [d.bark, d.leaves]), s.impostor]);
  }

  /**
   * Trees from an ecological niche model: canopy density follows moisture and temperature,
   * and each tree's species is drawn by how well its niche fits the spot (see ecology.ts).
   */
  private makeCell(cx: number, cz: number): TreeInstance[] {
    const rnd = cellRandom(cx, cz);
    const pick = (eco: Eco): number => {
      const suit = treeSuitability(eco);
      const total = suit.reduce((a, b) => a + b, 0);
      if (total < 0.05) return -1;
      let r = rnd() * total;
      for (let i = 0; i < suit.length; i++) {
        r -= suit[i];
        if (r <= 0) return i;
      }
      return suit.length - 1;
    };
    const tree = (x: number, z: number, eco: Eco): TreeInstance | null => {
      const [d, , hw] = riverInfo(x, z);
      if (d < hw * 1.8) return null;
      const species = pick(eco);
      if (species < 0) return null;
      const [s0, s1] = SPECIES[species].scale;
      // Trees grow taller where conditions suit them best.
      const vigor = 0.85 + 0.15 * Math.min(1, eco.moisture * 1.4);
      return { species, x, y: eco.height - 0.15, z, yaw: rnd() * Math.PI * 2, scale: (s0 + (s1 - s0) * rnd()) * vigor, seed: rnd() };
    };
    const center = ecology((cx + 0.5) * CELL, (cz + 0.5) * CELL);
    const density = treeDensity(center);
    if (density < 0.08) {
      // Open country: the odd lone tree, on the highest of a few spots (Flower's hilltop trees).
      if (rnd() > 0.32) return [];
      let best = { x: 0, z: 0, y: -Infinity };
      for (let i = 0; i < 6; i++) {
        const x = (cx + 0.1 + rnd() * 0.8) * CELL;
        const z = (cz + 0.1 + rnd() * 0.8) * CELL;
        const y = terrainHeight(x, z);
        if (y > best.y) best = { x, z, y };
      }
      const lone = tree(best.x, best.z, ecology(best.x, best.z));
      return lone ? [lone] : [];
    }
    // Woodland: a jittered lattice, finer where the canopy closes; each candidate survives with
    // the local canopy density, so forests thin out naturally toward dry or cold ground.
    // The forest's core (deep east) closes into dense stands.
    const n = density > 0.7 ? ((cx + 0.5) * CELL > 1100 ? 7 : 6) : 4;
    const out: TreeInstance[] = [];
    for (let i = 0; i < n * n; i++) {
      const x = (cx + (i % n + 0.15 + rnd() * 0.7) / n) * CELL;
      const z = (cz + (Math.floor(i / n) + 0.15 + rnd() * 0.7) / n) * CELL;
      const eco = ecology(x, z);
      if (rnd() > treeDensity(eco)) continue;
      const t = tree(x, z, eco);
      if (t) out.push(t);
    }
    return out;
  }

  private readonly pending: { key: string; cx: number; cz: number; d: number }[] = [];

  private stream(px: number, pz: number): void {
    const cx = Math.floor(px / CELL);
    const cz = Math.floor(pz / CELL);
    const key = `${cx},${cz}`;
    if (key !== this.loadedCell) {
      this.loadedCell = key;
      const r = Math.ceil(VIEW / CELL);
      const keep = new Set<string>();
      this.pending.length = 0;
      for (let dz = -r; dz <= r; dz++) {
        for (let dx = -r; dx <= r; dx++) {
          const k = `${cx + dx},${cz + dz}`;
          keep.add(k);
          if (!this.cells.has(k)) this.pending.push({ key: k, cx: cx + dx, cz: cz + dz, d: dx * dx + dz * dz });
        }
      }
      for (const k of this.cells.keys()) if (!keep.has(k)) this.cells.delete(k);
      // Nearest first: what is about to be close is generated before the far horizon.
      this.pending.sort((a, b) => b.d - a.d);
      this.instances = [...this.cells.values()].flat();
    }
    // A bounded amount of generation per frame (everything at once on the first frame).
    let budget = this.cells.size === 0 ? Infinity : CELLS_PER_FRAME;
    let added = false;
    while (budget-- > 0 && this.pending.length) {
      const c = this.pending.pop()!;
      if (!this.cells.has(c.key)) this.cells.set(c.key, this.makeCell(c.cx, c.cz));
      added = true;
    }
    if (added) this.instances = [...this.cells.values()].flat();
  }

  /** Nearby trees (for gameplay collision / avoidance). */
  near(x: number, z: number, radius: number): TreeInstance[] {
    return this.instances.filter((t) => (t.x - x) ** 2 + (t.z - z) ** 2 < radius * radius);
  }

  trunkRadius(t: TreeInstance): number {
    return [0.9, 0.35, 0.4][t.species] * t.scale;
  }

  canopyTop(t: TreeInstance): number {
    return t.y + this.species[t.species].manifest.height * t.scale;
  }

  update(cam: readonly number[], frustum: Float32Array, lodBias = 1): void {
    this.stream(cam[0], cam[2]);
    for (const s of this.species) s.counts.fill(0);
    for (const t of this.instances) {
      const s = this.species[t.species];
      const m = s.manifest;
      const h = m.height * t.scale;
      const cy = t.y + h * 0.5;
      const radius = Math.max(h * 0.6, m.impostor.size * t.scale * 0.5);
      let visible = true;
      for (let i = 0; i < 6 && visible; i++) {
        const o = i * 4;
        if (frustum[o] * t.x + frustum[o + 1] * cy + frustum[o + 2] * t.z + frustum[o + 3] < -radius) visible = false;
      }
      if (!visible) continue;
      const dist = Math.hypot(t.x - cam[0], cy - cam[1], t.z - cam[2]);
      if (dist > VIEW * lodBias) continue;
      // Grow in at the far edge instead of popping.
      const grow = Math.min(1, Math.max(0, (VIEW * lodBias - dist) / FADE));
      const d = dist / (t.scale * lodBias);
      const lod = d < m.lodDistances[0] ? 0 : d < m.lodDistances[1] ? 1 : 2;
      const n = s.counts[lod]++;
      if (n >= MAX_INSTANCES) continue;
      s.data[lod].set([t.x, t.y, t.z, t.scale * (0.2 + 0.8 * grow * grow), Math.cos(t.yaw), Math.sin(t.yaw), t.seed, 0], n * STRIDE);
    }
    for (const s of this.species) {
      for (let lod = 0; lod < 3; lod++) {
        if (s.counts[lod]) s.buffers[lod].write(s.data[lod].subarray(0, Math.min(s.counts[lod], MAX_INSTANCES) * STRIDE));
      }
    }
  }

  encode(pass: FramePass): void {
    for (const s of this.species) {
      for (let lod = 0; lod < 2; lod++) {
        const n = Math.min(s.counts[lod], MAX_INSTANCES);
        if (!n) continue;
        pass.draw(s.meshDraws[lod].bark, { instances: n });
        pass.draw(s.meshDraws[lod].leaves, { instances: n });
      }
      const ni = Math.min(s.counts[2], MAX_INSTANCES);
      if (ni) pass.draw(s.impostor, { instances: ni });
    }
  }
}
