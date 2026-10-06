import { uniforms, type Gpu, type SharedUniforms } from "vgpu";
import type { Camera } from "./camera";

export interface Atmosphere {
  sunDir: [number, number, number];
  sunColor: [number, number, number];
  horizonColor: [number, number, number];
  zenithColor: [number, number, number];
  fogDensity: number;
  exposure: number;
}

/** Moonlit night: cool light, deep indigo sky, a faint warm glow left on the horizon. */
export const NIGHT: Atmosphere = {
  sunDir: normalize([-0.25, 0.42, -0.85]),
  sunColor: [0.34, 0.44, 0.7],
  horizonColor: [0.11, 0.1, 0.2],
  zenithColor: [0.015, 0.03, 0.09],
  fogDensity: 0.0032,
  exposure: 1.6,
};

export function mixAtmosphere(a: Atmosphere, b: Atmosphere, t: number): Atmosphere {
  const m = (x: number, y: number) => x + (y - x) * t;
  const v = (x: number[], y: number[]) => x.map((xi, i) => m(xi, y[i])) as [number, number, number];
  return {
    sunDir: normalize(v(a.sunDir, b.sunDir)),
    sunColor: v(a.sunColor, b.sunColor),
    horizonColor: v(a.horizonColor, b.horizonColor),
    zenithColor: v(a.zenithColor, b.zenithColor),
    fogDensity: m(a.fogDensity, b.fogDensity),
    exposure: m(a.exposure, b.exposure),
  };
}

/** Golden-hour palette: low sun so the grass is backlit and glows. */
export const GOLDEN_HOUR: Atmosphere = {
  sunDir: normalize([0.35, 0.2, -0.9]),
  sunColor: [2.6, 1.85, 1.15],
  horizonColor: [0.95, 0.66, 0.45],
  zenithColor: [0.16, 0.34, 0.78],
  fogDensity: 0.0018,
  exposure: 1.0,
};

export function normalize(v: [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** WGSL `array<vec4f, N>` values are nested per element; views avoid per-frame allocation. */
export function vec4Views(flat: Float32Array): Float32Array[] {
  return Array.from({ length: flat.length / 4 }, (_, i) => flat.subarray(i * 4, i * 4 + 4));
}

/** One shared uniform block (`G` in every shader); see src/shaders/lib/globals.wgsl. */
export class Globals {
  readonly uniforms: SharedUniforms;
  readonly trail = new Float32Array(24 * 4);
  sunDir: [number, number, number] = [0, 1, 0];
  private readonly trailViews = vec4Views(this.trail);
  /** Lantern brightness by index (see Lanterns). */
  readonly lamps = new Float32Array(16 * 4);
  private readonly lampViews = vec4Views(this.lamps);
  /** Lantern positions (set once by Lanterns). */
  readonly lampPos = new Float32Array(64 * 4);
  private readonly lampPosViews = vec4Views(this.lampPos);

  constructor(gpu: Gpu, atmosphere: Atmosphere) {
    this.sunDir = atmosphere.sunDir;
    this.uniforms = uniforms(gpu, {
      viewProj: new Float32Array(16),
      invViewProj: new Float32Array(16),
      camPos: [0, 0, 0],
      time: 0,
      sunDir: atmosphere.sunDir,
      windStrength: 0.6,
      sunColor: atmosphere.sunColor,
      fogDensity: atmosphere.fogDensity,
      horizonColor: atmosphere.horizonColor,
      exposure: atmosphere.exposure,
      zenithColor: atmosphere.zenithColor,
      gust: 0,
      playerPos: [0, 0, 0],
      playerSpeed: 0,
      windDir: [0.8, 0.6],
      viewport: [1, 1],
      night: 0,
      explore: 0,
      mist: 0,
      mistBase: 0,
      canopy: 0,
      rain: 0,
      wet: 0,
      season: 0,
      frustum: vec4Views(new Float32Array(24)),
      trail: this.trailViews,
      lamps: this.lampViews,
      lampPos: this.lampPosViews,
    });
  }

  setAtmosphere(a: Atmosphere): void {
    this.sunDir = a.sunDir;
    this.uniforms.set({
      sunDir: a.sunDir,
      sunColor: a.sunColor,
      horizonColor: a.horizonColor,
      zenithColor: a.zenithColor,
      fogDensity: a.fogDensity,
      exposure: a.exposure,
    });
  }

  updateFrame(camera: Camera, time: number, viewport: [number, number], extra: Record<string, unknown> = {}): void {
    this.uniforms.set({
      viewProj: camera.viewProj,
      invViewProj: camera.invViewProj,
      camPos: camera.position,
      frustum: camera.frustumViews,
      time,
      viewport,
      trail: this.trailViews,
      lamps: this.lampViews,
      lampPos: this.lampPosViews,
      ...extra,
    });
  }
}
