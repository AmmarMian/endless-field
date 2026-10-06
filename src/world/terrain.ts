import { draw, geometry, sampler, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer, type Texture } from "vgpu";
import terrainShader from "../shaders/terrain.wgsl";
import type { MountainSet } from "./mountains";

const SNAP = 4;

/** Camera-following ground mesh; vertex density falls off as |g|^power toward the rim. */
export class Terrain {
  private readonly drawCall: Draw;

  constructor(gpu: Gpu, globals: SharedUniforms, life: StorageBuffer, pebbles: Texture, mountains: MountainSet, rock: Texture, scree: Texture, forestFloor: Texture, resolution = 384, private readonly radius = 3200) {
    const n = resolution;
    const verts = new Float32Array((n + 1) * (n + 1) * 2);
    let o = 0;
    for (let z = 0; z <= n; z++) {
      for (let x = 0; x <= n; x++) {
        verts[o++] = (x / n) * 2 - 1;
        verts[o++] = (z / n) * 2 - 1;
      }
    }
    const indices = new Uint32Array(n * n * 6);
    o = 0;
    for (let z = 0; z < n; z++) {
      for (let x = 0; x < n; x++) {
        const a = z * (n + 1) + x;
        const b = a + 1;
        const c = a + n + 1;
        const d = c + 1;
        indices.set([a, c, b, b, c, d], o);
        o += 6;
      }
    }
    this.drawCall = draw(gpu, {
      label: "terrain",
      shader: terrainShader,
      geometry: geometry(gpu, { buffers: [{ data: verts, attributes: { g: "float32x2" } }], indices }),
      depth: { compare: "greater" },
      set: {
        G: globals,
        grid: { center: [0, 0], radius: this.radius, power: 2.2, cells: resolution, pad0: 0, pad1: 0, pad2: 0 },
        life,
        samp: sampler(gpu, { minFilter: "linear", magFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat", maxAnisotropy: 8 }),
        pebbles,
        mtnTex: mountains.texture,
        mtnSamp: mountains.sampler,
        rockTex: rock,
        screeTex: scree,
        floorTex: forestFloor,
      },
    });
  }

  get draw(): Draw {
    return this.drawCall;
  }

  update(camX: number, camZ: number): void {
    const cx = Math.round(camX / SNAP) * SNAP;
    const cz = Math.round(camZ / SNAP) * SNAP;
    this.drawCall.set({ grid: { center: [cx, cz] } });
  }

  encode(pass: FramePass): void {
    pass.draw(this.drawCall);
  }
}
