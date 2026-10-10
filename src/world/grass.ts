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
  heightNext: number;
}

export type GrassQuality = "low" | "medium" | "high" | "ultra";

/**
 * Nested LOD rings: each ring keeps every 3rd blade (per axis) of the ring inside it, so a
 * blade keeps its place and shape as it moves between rings; only the extra blades of a
 * ring fade out at its outer edge, and the survivors widen to keep the field covered.
 */
export function grassLods(quality: GrassQuality): GrassLodConfig[] {
  const base = { low: 0.15, medium: 0.115, high: 0.09, ultra: 0.075 }[quality];
  // The last ring is very sparse and wide-bladed: it carries the grass's texture and its fuzz
  // on the ridges out toward the horizon, so far hills read as grassland, not bare ground.
  const radii = {
    low: [24, 70, 140, 420],
    medium: [32, 85, 160, 380, 850],
    high: [42, 100, 180, 480, 1100],
    ultra: [55, 120, 200, 560, 1300],
  }[quality];
  // Each ring keeps 1 blade in 9, so widths grow ~3x per ring to keep the field closed.
  const widths = [1, 2.6, 7.6, 22, 62];
  const hs = [1, 1, 1.05, 1.15, 1.3];
  return radii.map((rOuter, i) => {
    const last = i === radii.length - 1;
    return {
      label: ["grass-near", "grass-mid", "grass-far", "grass-horizon", "grass-distant"][i],
      segments: i === 0 ? 7 : i === 1 ? 3 : 2,
      baseSpacing: base,
      k: 3 ** i,
      kNext: last ? 0 : 3 ** (i + 1),
      rInner: i === 0 ? 0 : radii[i - 1],
      rOuter,
      // Continuous LOD: the blades that do not continue thin out across the whole ring (the
      // near ring: its outer 70%), so density falls smoothly with distance and each blade
      // changes slowly as you fly, instead of a band where they all shrink at once.
      fade: last ? rOuter * 0.35 : (rOuter - (i === 0 ? 0 : radii[i - 1])) * (i >= 3 ? 0.35 : 0.4),
      widthScale: widths[i],
      widthNext: last ? widths[i] : widths[i + 1],
      heightScale: hs[i],
      heightNext: last ? hs[i] : hs[i + 1],
    };
  });
}

interface Lod {
  config: GrassLodConfig;
  gridSize: number;
  verts: number;
  cull: Compute;
  render: Draw;
  /** The same blades without compute (direct mode). */
  direct: Draw | null;
  args: StorageBuffer;
  blades: StorageBuffer;
  reset: Uint32Array<ArrayBuffer>;
}

/**
 * GPU-driven grass: compute places/culls/animates blades, indirect draws render them. Some
 * phones draw nothing that way; in direct mode the same blades come from a plain instanced
 * draw (each vertex places its blade itself), on a sparser grid to keep it affordable.
 */
export class Grass {
  private lods: Lod[] = [];
  quality: GrassQuality;
  /** Direct mode (no compute): set by `useDirect`. */
  direct = false;

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

  /** Switches to direct mode (for devices whose compute grass draws nothing). */
  useDirect(): void {
    if (this.direct) return;
    this.direct = true;
    for (const lod of this.lods) {
      for (const b of [lod.blades, lod.args]) (b as unknown as { destroy?: () => void }).destroy?.();
    }
    this.build(this.quality);
  }

  private build(quality: GrassQuality): void {
    const { gpu, globals, life, mountains } = this;
    // Direct mode: a sparser, wider-bladed field (every vertex places its own blade).
    const sparse = this.direct ? 1.7 : 1;
    this.lods = grassLods(quality).map((base) => {
      const config = { ...base, baseSpacing: base.baseSpacing * sparse, widthScale: base.widthScale * sparse, widthNext: base.widthNext * sparse };
      const spacing = config.baseSpacing * config.k;
      const gridSize = Math.ceil((config.rOuter * 2) / spacing) + 2;
      const blades = storage(gpu, this.direct ? BLADE_BYTES : gridSize * gridSize * BLADE_BYTES, "read-write");
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
            heightNext: config.heightNext,
            lodShift: [0, 0],
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
      const direct = this.direct
        ? draw(gpu, {
            label: `${config.label}-direct`,
            shader: cullShader,
            entry: { vertex: "vs_direct", fragment: "fs_direct" },
            vertices: verts,
            constants: { ...worldConstants(cullShader), NSEG: config.segments },
            depth: { compare: "greater" },
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
                heightNext: config.heightNext,
                lodShift: [0, 0],
              },
              life,
              mtnTex: mountains.texture,
              mtnSamp: mountains.sampler,
            },
          })
        : null;
      return { config, gridSize, verts, cull, render, direct, args, blades, reset: new Uint32Array([verts, 0, 0, 0]) };
    });
  }

  get draws(): Draw[] {
    return this.lods.map((l) => l.direct ?? l.render);
  }

  /** Debug: shifts the rings' centre away from the camera (to see LOD changes alone). */
  lodShift: [number, number] = [0, 0];

  update(camX: number, camZ: number): void {
    camX += this.lodShift[0];
    camZ += this.lodShift[1];
    for (const lod of this.lods) {
      if (lod.direct) {
        lod.direct.set({ P: { lodShift: this.lodShift, centerCell: [Math.round(camX / (lod.config.baseSpacing * lod.config.k)), Math.round(camZ / (lod.config.baseSpacing * lod.config.k))] } });
        continue;
      }
      lod.args.write(lod.reset);
      lod.cull.set({
        P: { lodShift: this.lodShift, centerCell: [Math.round(camX / (lod.config.baseSpacing * lod.config.k)), Math.round(camZ / (lod.config.baseSpacing * lod.config.k))] },
      });
      const groups = Math.ceil(lod.gridSize / 16);
      lod.cull.dispatch(groups, groups);
    }
  }

  encode(pass: FramePass): void {
    for (const lod of this.lods) {
      if (lod.direct) pass.draw(lod.direct, { instances: lod.gridSize * lod.gridSize });
      else pass.draw(lod.render, { indirect: lod.args });
    }
  }

  /** Reads back how many blades each LOD drew last frame (diagnostics only). */
  async counts(): Promise<number[]> {
    return Promise.all(this.lods.map(async (l) => new Uint32Array(await l.args.read())[1]));
  }
}
