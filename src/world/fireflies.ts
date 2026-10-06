import { worldConstants } from "./height";
import { draw, type Draw, type FramePass, type Gpu, type SharedUniforms } from "vgpu";
import fireflyShader from "../shaders/fireflies.wgsl";
import type { MountainSet } from "./mountains";

const CELL = 3;
const GRID = 44;

/** Night fireflies: procedural, world-anchored, drawn additively (one instance per cell). */
export class Fireflies {
  readonly draw: Draw;

  constructor(gpu: Gpu, globals: SharedUniforms, mountains: MountainSet) {
    this.draw = draw(gpu, {
      label: "fireflies",
      shader: fireflyShader,
      constants: worldConstants(),
      vertices: 6,
      instances: GRID * GRID,
      blend: "additive",
      colors: [null, { writeMask: [] }],
      depth: { compare: "greater", write: false },
      set: { G: globals, F: { centerCell: [0, 0], gridSize: GRID, cellSize: CELL }, mtnTex: mountains.texture, mtnSamp: mountains.sampler },
    });
  }

  update(camX: number, camZ: number): void {
    this.draw.set({ F: { centerCell: [Math.round(camX / CELL), Math.round(camZ / CELL)] } });
  }

  encode(pass: FramePass, night: number): void {
    if (night > 0.02) pass.draw(this.draw);
  }
}
