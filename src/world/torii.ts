import { draw, geometry, sampler, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import toriiShader from "../shaders/torii.wgsl";
import { loadTexture } from "../engine/textures";
import { terrainHeightM as terrainHeight, worldConstants } from "./height";
import { PATH, pathNormal, pathZ } from "./lantern-path";
import { loadBin } from "../engine/assets";

interface Manifest {
  vertexBytes: number;
  indexCount: number;
}

/** Gate positions: one at each end of the lantern path, straddling it, facing along it. */
export function toriiGates(): { x: number; z: number; cos: number; sin: number }[] {
  return [PATH.x0 + 3, PATH.x1 - 3].map((x) => {
    const [nx, nz] = pathNormal(x);
    // The model's +X (through the gate) runs along the path: (nz, -nx) is the path direction.
    const dx = nz;
    const dz = -nx;
    return { x, z: pathZ(x), cos: dx, sin: -dz };
  });
}

/** The two vermilion torii that open and close the lantern path. */
export class Torii {
  readonly draws: Draw[];

  private constructor(private readonly mesh: Draw, private readonly buffer: StorageBuffer, private readonly data: Float32Array<ArrayBuffer>) {
    this.draws = [mesh];
  }

  static async load(gpu: Gpu, globals: SharedUniforms, base = "assets/torii"): Promise<Torii> {
    const tex = (name: string, srgb: boolean) => loadTexture(gpu, `${base}/tex/${name}_2k.jpg`, { srgb });
    const [manifest, bin, woodDiff, woodNor, rockDiff, rockNor] = await Promise.all([
      fetch(`${base}/torii.json`).then((r) => r.json() as Promise<Manifest>),
      loadBin(`${base}/torii.bin`),
      tex("hinoki_planks_diff", true),
      tex("hinoki_planks_nor_gl", false),
      tex("rock_surface_diff", true),
      tex("rock_surface_nor_gl", false),
    ]);
    const geo = geometry(gpu, {
      label: "torii",
      buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
    });
    const gates = toriiGates();
    const data = new Float32Array(gates.flatMap((g) => [g.x, terrainHeight(g.x, g.z) - 0.05, g.z, 0, g.cos, g.sin, 0, 0]));
    const buffer = storage(gpu, data.byteLength, "read");
    buffer.write(data);
    const samp = sampler(gpu, { minFilter: "linear", magFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat", maxAnisotropy: 8 });
    const mesh = draw(gpu, {
      label: "torii",
      shader: toriiShader,
      constants: worldConstants(toriiShader),
      geometry: geo,
      depth: { compare: "greater" },
      set: { G: globals, gates: buffer, samp, woodDiff, woodNor, rockDiff, rockNor },
    });
    return new Torii(mesh, buffer, data);
  }

  private glow = 0;

  /** Eases the gates' inner glow toward `target` (0..1). */
  setGlow(target: number, dt: number): void {
    const next = this.glow + (target - this.glow) * Math.min(1, dt * 0.5);
    if (Math.abs(next - this.glow) < 1e-5) return;
    this.glow = next;
    this.data[3] = this.data[11] = next;
    this.buffer.write(this.data);
  }

  encode(pass: FramePass): void {
    pass.draw(this.mesh, { instances: 2 });
  }
}
