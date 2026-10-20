import { compute, draw, effect, geometry, sampler, storage, target, type Draw, type FramePass, type Gpu, type SharedUniforms } from "vgpu";
import seaShader from "../shaders/sea.wgsl";
import initShader from "../shaders/ocean-init.wgsl";
import updateShader from "../shaders/ocean-update.wgsl";
import rowShader from "../shaders/ocean-fft-row.wgsl";
import colShader from "../shaders/ocean-fft-col.wgsl";
import bakeShader from "../shaders/ocean-bake.wgsl";
import { coastInfo, worldConstants } from "./height";

const N = 256;
const COMPLEX_BYTES = N * N * 2 * 4;
const VEC4_BYTES = N * N * 4 * 4;
const SNAP = 4;

/** The swell: a gentle onshore breeze over a long fetch (calm enough for a beach). */
const SWELL = {
  windSpeed: 9,
  // Toward the shore (+z is south, out to sea): the swell runs north, onto the beach.
  windAngle: -100,
  amplitude: 4,
  patchSize: 140,
  heightScale: 2.2,
  choppyScale: 1.6,
};

/**
 * The sea south of the coast: an FFT swell (ocean-*.wgsl, from the vgpu fft-ocean-surface
 * example) baked each frame into a displacement texture, and the surface drawn over a
 * camera-centered grid reaching the horizon (sea.wgsl), with surf and swash at the beach.
 */
export class Sea {
  private visible = true;
  private simTime = 0;
  private readonly update: ReturnType<typeof compute>;
  private readonly row: ReturnType<typeof compute>;
  private readonly col: ReturnType<typeof compute>;
  private readonly bake: ReturnType<typeof effect>;
  private readonly dispTarget: ReturnType<typeof target>;
  readonly draw: Draw;

  constructor(gpu: Gpu, globals: SharedUniforms, resolution = 320, private readonly radius = 20000) {
    const a = (SWELL.windAngle * Math.PI) / 180;
    const sim = (time: number) => ({ windDir: [Math.cos(a), Math.sin(a)], windSpeed: SWELL.windSpeed, amplitude: SWELL.amplitude, patchSize: SWELL.patchSize, time, _pad: [0, 0] });
    const h0 = storage(gpu, VEC4_BYTES, "read-write");
    const spec = [0, 1, 2].map(() => storage(gpu, COMPLEX_BYTES, "read-write"));
    const tmp = [0, 1, 2].map(() => storage(gpu, COMPLEX_BYTES, "read-write"));
    const displacement = storage(gpu, VEC4_BYTES, "read-write");
    compute(gpu, initShader, { label: "ocean-init", set: { h0, sim: sim(0) } }).dispatch(N / 8, N / 8);
    this.update = compute(gpu, updateShader, { label: "ocean-update", set: { h0, specX: spec[0], specY: spec[1], specZ: spec[2], sim: sim(0) } });
    this.row = compute(gpu, rowShader, { label: "ocean-fft-row", set: { inX: spec[0], inY: spec[1], inZ: spec[2], outX: tmp[0], outY: tmp[1], outZ: tmp[2] } });
    this.col = compute(gpu, colShader, { label: "ocean-fft-col", set: { inX: tmp[0], inY: tmp[1], inZ: tmp[2], disp: displacement } });
    this.dispTarget = target(gpu, { size: [N, N], format: "rgba16float" });
    this.bake = effect(gpu, bakeShader, {
      label: "ocean-bake",
      set: { disp: displacement, B: { patchSize: SWELL.patchSize, heightScale: SWELL.heightScale, choppyScale: SWELL.choppyScale, pad: 0 } },
    });
    this.sim = sim;

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
        const i = z * (n + 1) + x;
        indices.set([i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2], o);
        o += 6;
      }
    }
    this.draw = draw(gpu, {
      label: "sea",
      shader: seaShader,
      constants: worldConstants(seaShader),
      geometry: geometry(gpu, { buffers: [{ data: verts, attributes: { g: "float32x2" } }], indices }),
      depth: { compare: "greater" },
      cull: "none",
      blend: { color: { src: "one", dst: "src-alpha" }, alpha: { src: "zero", dst: "one" } },
      set: {
        G: globals,
        S: { center: [0, 0], radius: this.radius, power: 3, cells: n, patchSize: SWELL.patchSize, glow: 0, pad: 0 },
        disp: this.dispTarget,
        dispSamp: sampler(gpu, { addressModeU: "repeat", addressModeV: "repeat", minFilter: "linear", magFilter: "linear" }),
      },
    });
  }

  private readonly sim: (time: number) => Record<string, unknown>;

  /** Advances the swell and recentres the surface on the camera. */
  step(dt: number, camX: number, camZ: number, night: number): void {
    // The sea is out of sight only far inland (hills hide it well before its fog would).
    this.visible = coastInfo(camX, camZ)[0] < 2600;
    if (!this.visible) return;
    this.simTime += dt;
    this.update.set({ sim: this.sim(this.simTime) });
    this.update.dispatch(N / 8, N / 8);
    this.row.dispatch(N, 1);
    this.col.dispatch(N, 1);
    this.bake.draw(this.dispTarget);
    this.draw.set({ S: { center: [Math.round(camX / SNAP) * SNAP, Math.round(camZ / SNAP) * SNAP], glow: night } });
  }

  encode(pass: FramePass): void {
    if (this.visible) pass.draw(this.draw);
  }
}
