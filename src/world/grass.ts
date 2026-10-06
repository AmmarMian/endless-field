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
  spacing: number;
  rInner: number;
  rOuter: number;
  fade: number;
  widthScale: number;
  heightScale: number;
  thinStart: number;
  thinEnd: number;
  thinMin: number;
}

export type GrassQuality = "low" | "medium" | "high" | "ultra";

export function grassLods(quality: GrassQuality): GrassLodConfig[] {
  const nearSpacing = { low: 0.15, medium: 0.115, high: 0.09, ultra: 0.075 }[quality];
  const farSpacing = { low: 0.42, medium: 0.33, high: 0.27, ultra: 0.23 }[quality];
  const nearRadius = { low: 20, medium: 26, high: 30, ultra: 36 }[quality];
  const horizonSpacing = { low: 1, medium: 0.95, high: 0.8, ultra: 0.65 }[quality];
  const horizonRadius = { low: 0, medium: 200, high: 260, ultra: 300 }[quality];
  return [
    {
      label: "grass-near",
      segments: 7,
      spacing: nearSpacing,
      rInner: 0,
      rOuter: nearRadius,
      fade: 4,
      widthScale: 1,
      heightScale: 1,
      thinStart: 1e5,
      thinEnd: 2e5,
      thinMin: 1,
    },
    {
      label: "grass-far",
      segments: 2,
      spacing: farSpacing,
      rInner: nearRadius,
      rOuter: 130,
      fade: 4,
      widthScale: farSpacing / nearSpacing * 0.75,
      heightScale: 1,
      thinStart: nearRadius + 10,
      thinEnd: 130,
      thinMin: 0.4,
    },
    // Horizon ring: very sparse, wide blades so the field reaches the fog when flying high.
    ...(quality === "low"
      ? []
      : [
          {
            label: "grass-horizon",
            segments: 2,
            spacing: horizonSpacing,
            rInner: 124,
            rOuter: horizonRadius,
            fade: 6,
            widthScale: (horizonSpacing / nearSpacing) * 0.6,
            heightScale: 1.1,
            thinStart: 130,
            thinEnd: horizonRadius,
            thinMin: 0.45,
          },
        ]),
  ];
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
      const gridSize = Math.ceil((config.rOuter * 2) / config.spacing) + 2;
      const blades = storage(gpu, gridSize * gridSize * BLADE_BYTES, "read-write");
      const args = storage(gpu, 16, { indirect: true });
      const verts = (2 * (config.segments - 1) + 1) * 3;
      const cull = compute(gpu, cullShader, {
        label: `${config.label}-cull`,
        set: {
          G: globals,
          P: {
            centerCell: [0, 0],
            gridSize,
            spacing: config.spacing,
            rInner: config.rInner,
            rOuter: config.rOuter,
            fade: config.fade,
            widthScale: config.widthScale,
            heightScale: config.heightScale,
            thinStart: config.thinStart,
            thinEnd: config.thinEnd,
            thinMin: config.thinMin,
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
        constants: { NSEG: config.segments },
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
        P: { centerCell: [Math.round(camX / lod.config.spacing), Math.round(camZ / lod.config.spacing)] },
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
