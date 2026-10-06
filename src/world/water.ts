import { draw, geometry, sampler, type Draw, type FramePass, type Gpu, type SharedUniforms } from "vgpu";
import waterShader from "../shaders/water.wgsl";
import { loadTexture } from "../engine/textures";
import { riverInfo, worldConstants } from "./height";

const GRID = 192;
const EXTENT = 190;
const SNAP = 2;

/** River surface: a camera-centered grid that only shades inside the river channel. */
export class Water {
  private visible = true;

  private constructor(readonly draw: Draw) {}

  static async load(gpu: Gpu, globals: SharedUniforms): Promise<Water> {
    const pebbles = await loadTexture(gpu, "assets/textures/pebbles.jpg", { srgb: true });
    const verts = new Float32Array((GRID + 1) * (GRID + 1) * 2);
    let o = 0;
    for (let z = 0; z <= GRID; z++) {
      for (let x = 0; x <= GRID; x++) {
        verts[o++] = (x / GRID) * 2 - 1;
        verts[o++] = (z / GRID) * 2 - 1;
      }
    }
    const indices = new Uint32Array(GRID * GRID * 6);
    o = 0;
    for (let z = 0; z < GRID; z++) {
      for (let x = 0; x < GRID; x++) {
        const a = z * (GRID + 1) + x;
        const b = a + GRID + 1;
        indices.set([a, b, a + 1, a + 1, b, b + 1], o);
        o += 6;
      }
    }
    const samp = sampler(gpu, {
      minFilter: "linear",
      magFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
      maxAnisotropy: 8,
    });
    return new Water(
      draw(gpu, {
        label: "water",
        shader: waterShader,
        constants: worldConstants(),
        geometry: geometry(gpu, { buffers: [{ data: verts, attributes: { g: "float32x2" } }], indices }),
        depth: { compare: "greater" },
        set: { G: globals, W: { center: [0, 0], extent: EXTENT, pad: 0 }, samp, pebbles },
      }),
    );
  }

  update(camX: number, camZ: number): void {
    // Skip the draw entirely when the river is out of range.
    this.visible = riverInfo(camX, camZ)[0] < EXTENT * 1.3;
    this.draw.set({ W: { center: [Math.round(camX / SNAP) * SNAP, Math.round(camZ / SNAP) * SNAP] } });
  }

  encode(pass: FramePass): void {
    if (this.visible) pass.draw(this.draw);
  }
}
