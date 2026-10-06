import { draw, type Draw, type FramePass, type Gpu, type SharedUniforms } from "vgpu";
import rainShader from "../shaders/rain.wgsl";

/** Drops in flight at full intensity (rain streaks, or snowflakes in winter). */
const DROPS = 9000;

/** Precipitation around the camera (see rain.wgsl). */
export class Rain {
  readonly draws: Draw[];

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.draws = [
      draw(gpu, {
        label: "rain",
        shader: rainShader,
        vertices: 6,
        depth: { compare: "greater", write: false },
        blend: "additive",
        set: { G: globals },
      }),
    ];
  }

  encode(pass: FramePass, intensity: number): void {
    if (intensity > 0.01) pass.draw(this.draws[0], { instances: Math.ceil(DROPS * intensity) });
  }
}
