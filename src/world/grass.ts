import { worldConstants } from "./height";
import {
  compute,
  draw,
  storage,
  type Compute,
  type Draw,
  type FramePass,
  type Gpu,
  type SharedUniforms,
  type StorageBuffer,
} from "vgpu";
import cullShader from "../shaders/grass-cull.wgsl";
import grassShader from "../shaders/grass.wgsl";
import type { MountainSet } from "./mountains";

const BLADE_BYTES = 64;

export interface GrassLodConfig {
  label: string;
  segments: number;
  /** Finest grid spacing (blade identity), and this ring's multiple of it. */
  baseSpacing: number;
  k: number;
  /** Multiple of the next (coarser) ring, 0 for the last ring. */
  kNext: number;
  rInner: number;
  rOuter: number;
  fade: number;
  widthScale: number;
  widthNext: number;
  heightScale: number;
}

export type GrassQuality = "low" | "medium" | "high" | "ultra";

/**
 * Nested LOD rings: each ring keeps every 3rd blade (per axis) of the ring inside it, so a
 * blade keeps its place and shape as it moves between rings; only the extra blades of a
 * ring fade out at its outer edge, and the survivors widen to keep the field covered.
 */
export function grassLods(quality: GrassQuality): GrassLodConfig[] {
  const base = { low: 0.15, medium: 0.115, high: 0.09, ultra: 0.075 }[quality];
  const radii = {
    low: [24, 70, 140],
    medium: [32, 85, 160, 380],
    high: [42, 100, 180, 480],
    ultra: [55, 120, 200, 560],
  }[quality];
  // Each ring keeps 1 blade in 9, so widths grow ~3x per ring to keep the field closed.
  const widths = [1, 2.6, 7.6, 22];
  return radii.map((rOuter, i) => {
    const last = i === radii.length - 1;
    return {
      label: ["grass-near", "grass-mid", "grass-far", "grass-horizon"][i],
      segments: i === 0 ? 7 : i === 1 ? 3 : 2,
      baseSpacing: base,
      k: 3 ** i,
      kNext: last ? 0 : 3 ** (i + 1),
      rInner: i === 0 ? 0 : radii[i - 1],
      rOuter,
      fade: last ? rOuter * 0.35 : Math.min(10, (rOuter - (i === 0 ? 0 : radii[i - 1])) * 0.3),
      widthScale: widths[i],
      widthNext: last ? widths[i] : widths[i + 1],
      heightScale: i === 3 ? 1.1 : 1,
    };
  });
}

interface Lod {
  config: GrassLodConfig;
  gridSize: number;
  verts: number;
  cull: Compute;
  render: Draw;
  args: StorageBuffer;
  blades: StorageBuffer;
  reset: Uint32Array<ArrayBuffer>;
}

/** GPU-driven grass: compute places/culls/animates blades, indirect draws render them. */
export class Grass {
  private lods: Lod[] = [];
  quality: GrassQuality;

  constructor(
    private readonly gpu: Gpu,
    private readonly globals: SharedUniforms,
    private readonly life: StorageBuffer,
    private readonly mountains: MountainSet,
    quality: GrassQuality,
  ) {
    this.quality = quality;
    this.build(quality);
  }

  /** Rebuilds the LOD rings for a new density; frees the old blade buffers. */
  setQuality(quality: GrassQuality): void {
    if (quality === this.quality) return;
    for (const lod of this.lods) {
      for (const b of [lod.blades, lod.args]) (b as unknown as { destroy?: () => void }).destroy?.();
    }
    this.quality = quality;
    this.build(quality);
  }

  private build(quality: GrassQuality): void {
    const { gpu, globals, life, mountains } = this;
    this.lods = grassLods(quality).map((config) => {
      const spacing = config.baseSpacing * config.k;
      const gridSize = Math.ceil((config.rOuter * 2) / spacing) + 2;
      const blades = storage(gpu, gridSize * gridSize * BLADE_BYTES, "read-write");
      const args = storage(gpu, 16, { indirect: true });
      const verts = (2 * (config.segments - 1) + 1) * 3;
      const cull = compute(gpu, cullShader, {
        label: `${config.label}-cull`,
        constants: worldConstants(cullShader),
        set: {
          G: globals,
          P: {
            centerCell: [0, 0],
            gridSize,
            spacing,
            rInner: config.rInner,
            rOuter: config.rOuter,
            fade: config.fade,
            widthScale: config.widthScale,
            heightScale: config.heightScale,
            thinStart: 0,
            thinEnd: 1,
            thinMin: 1,
            baseSpacing: config.baseSpacing,
            k: config.k,
            kNext: config.kNext,
            widthNext: config.widthNext,
          },
          blades,
          args,
          life,
          mtnTex: mountains.texture,
          mtnSamp: mountains.sampler,
        },
      });
      const render = draw(gpu, {
        label: config.label,
        shader: grassShader,
        constants: { ...worldConstants(grassShader), NSEG: config.segments },
        depth: { compare: "greater" },
        set: { G: globals, blades },
      });
      return { config, gridSize, verts, cull, render, args, blades, reset: new Uint32Array([verts, 0, 0, 0]) };
    });
  }

  get draws(): Draw[] {
    return this.lods.map((l) => l.render);
  }

  update(camX: number, camZ: number): void {
    for (const lod of this.lods) {
      lod.args.write(lod.reset);
      lod.cull.set({
        P: { centerCell: [Math.round(camX / (lod.config.baseSpacing * lod.config.k)), Math.round(camZ / (lod.config.baseSpacing * lod.config.k))] },
      });
      const groups = Math.ceil(lod.gridSize / 16);
      lod.cull.dispatch(groups, groups);
    }
  }

  encode(pass: FramePass): void {
    for (const lod of this.lods) pass.draw(lod.render, { indirect: lod.args });
  }

  /** Reads back how many blades each LOD drew last frame (diagnostics only). */
  async counts(): Promise<number[]> {
    return Promise.all(this.lods.map(async (l) => new Uint32Array(await l.args.read())[1]));
  }
}
