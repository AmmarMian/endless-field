import {
  draw,
  effect,
  sampler,
  surface,
  target,
  type Draw,
  type Effect,
  type Frame,
  type FramePass,
  type Gpu,
  type SharedUniforms,
  type Surface,
  type Target,
  type TimerSpan,
} from "vgpu";
import skyShader from "../shaders/sky.wgsl";
import downShader from "../shaders/post-down.wgsl";
import upShader from "../shaders/post-up.wgsl";
import compositeShader from "../shaders/post-composite.wgsl";

const HDR: GPUTextureFormat = "rgba16float";
const BLOOM_LEVELS = 5;

export interface RendererOptions {
  renderScale: number;
  msaa: boolean;
}

type Size = [number, number];

function scaled(size: readonly [number, number], k: number): Size {
  return [Math.max(1, Math.floor(size[0] * k)), Math.max(1, Math.floor(size[1] * k))];
}

/**
 * Owns the canvas surface, the HDR scene target and the post chain. Scene draws are encoded
 * by the caller inside `render(frame, encodeScene)`.
 */
export class Renderer {
  readonly output: Surface;
  readonly scene: Target;
  readonly sky: Draw;
  private readonly down: Target[] = [];
  private readonly up: Target[] = [];
  private readonly downFx: Effect[] = [];
  private readonly upFx: Effect[] = [];
  private readonly composite: Effect;
  private renderScale: number;
  private bloomOn = true;
  private bloomStrength = 0.1;

  constructor(gpu: Gpu, canvas: HTMLCanvasElement, globals: SharedUniforms, opts: RendererOptions) {
    this.renderScale = opts.renderScale;
    this.output = surface(gpu, canvas, { dpr: [1, 2] });
    const sceneSize = scaled(this.output.size, this.renderScale);
    this.scene = target(gpu, { size: sceneSize, format: HDR, depth: "depth32float", msaa: opts.msaa, label: "scene" });
    // Drawn last with an `equal` test against the cleared far depth (0 in reversed-Z), so the
    // cloud shader only runs on pixels nothing else covered.
    this.sky = draw(gpu, { label: "sky", shader: skyShader, depth: { compare: "equal", write: false }, set: { G: globals } });

    const linear = sampler(gpu, {
      minFilter: "linear",
      magFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      const size = scaled(sceneSize, 1 / 2 ** (i + 1));
      this.down.push(target(gpu, { size, format: HDR, label: `bloom-down-${i}` }));
      const src = i === 0 ? this.scene : this.down[i - 1];
      this.downFx.push(
        effect(gpu, downShader, {
          label: `bloom-down-${i}`,
          set: { src, samp: linear, params: { texel: src.texelSize, threshold: i === 0 ? 1.1 : 0, knee: 0.6 } },
        }),
      );
    }
    for (let i = 0; i < BLOOM_LEVELS - 1; i++) {
      this.up.push(target(gpu, { size: this.down[i].size, format: HDR, label: `bloom-up-${i}` }));
    }
    for (let i = 0; i < BLOOM_LEVELS - 1; i++) {
      const coarse = i === BLOOM_LEVELS - 2 ? this.down[BLOOM_LEVELS - 1] : this.up[i + 1];
      this.upFx.push(
        effect(gpu, upShader, {
          label: `bloom-up-${i}`,
          set: { coarse, fine: this.down[i], samp: linear, params: { texel: coarse.texelSize, radius: 1.0, pad: 0 } },
        }),
      );
    }
    this.composite = effect(gpu, compositeShader, {
      label: "composite",
      set: {
        scene: this.scene,
        bloom: this.up[0],
        samp: linear,
        params: { bloomStrength: 0.1, exposure: 0.8, vignette: 0.6, time: 0, grade: [1.0, 0.6, 0.35, 1.15] },
      },
    });

    this.output.onResize(() => this.resize());
  }

  setRenderScale(k: number): void {
    this.renderScale = k;
    this.resize();
  }

  private resize(): void {
    const sceneSize = scaled(this.output.size, this.renderScale);
    if (this.scene.size[0] === sceneSize[0] && this.scene.size[1] === sceneSize[1]) return;
    this.scene.resize(sceneSize);
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      this.down[i].resize(scaled(sceneSize, 1 / 2 ** (i + 1)));
      const src = i === 0 ? this.scene : this.down[i - 1];
      this.downFx[i].set({ params: { texel: src.texelSize } });
    }
    for (let i = 0; i < BLOOM_LEVELS - 1; i++) this.up[i].resize(this.down[i].size);
    for (let i = 0; i < BLOOM_LEVELS - 1; i++) {
      const coarse = i === BLOOM_LEVELS - 2 ? this.down[BLOOM_LEVELS - 1] : this.up[i + 1];
      this.upFx[i].set({ params: { texel: coarse.texelSize } });
    }
  }

  get aspect(): number {
    return this.output.size[0] / this.output.size[1];
  }

  get viewport(): [number, number] {
    return [this.scene.size[0], this.scene.size[1]];
  }

  setPost(values: { time?: number; exposure?: number; bloomStrength?: number }): void {
    if (values.bloomStrength !== undefined) this.bloomStrength = values.bloomStrength;
    this.composite.set({ params: { ...values, bloomStrength: this.bloomOn ? this.bloomStrength : 0 } });
  }

  setBloom(on: boolean): void {
    this.bloomOn = on;
    this.composite.set({ params: { bloomStrength: on ? this.bloomStrength : 0 } });
  }

  render(frame: Frame, encodeScene: (pass: FramePass) => void, spans?: { scene: TimerSpan; post: TimerSpan }): void {
    frame.pass({ target: this.scene, clear: [0, 0, 0, 1], clearDepth: 0, timer: spans?.scene }, (pass) => {
      encodeScene(pass);
      pass.draw(this.sky);
    });
    if (this.bloomOn) {
      for (let i = 0; i < BLOOM_LEVELS; i++) frame.pass(this.down[i], this.downFx[i]);
      for (let i = BLOOM_LEVELS - 2; i >= 0; i--) frame.pass(this.up[i], this.upFx[i]);
    }
    frame.pass({ target: this.output, timer: spans?.post }, (pass) => pass.draw(this.composite));
  }
}
