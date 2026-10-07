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

/** (summer, autumn, winter, spring) weights for a season value in [0, 4) (mirror of season.wgsl). */
export function seasonWeights(s: number): [number, number, number, number] {
  return [0, 1, 2, 3].map((i) => {
    const d = Math.abs((((s - i) / 4 + 0.5) % 1 + 1) % 1 * 4 - 2);
    return Math.max(0, 1 - d);
  }) as [number, number, number, number];
}

/**
 * Season and weather on top of the time of day: autumn warms the light, winter cools and
 * pales it, spring clears it; rain closes the sky into an overcast grey and thickens the haze.
 */
export function weatherAtmosphere(a: Atmosphere, season: number, rain: number, night = 0): Atmosphere {
  const [su, au, wi, sp] = seasonWeights(season);
  const tint = (c: number[], warm: number[], cold: number[], fresh: number[]) =>
    c.map((v, i) => v * (su + au * warm[i] + wi * cold[i] + sp * fresh[i])) as [number, number, number];
  let sunColor = tint(a.sunColor, [1.06, 0.93, 0.82], [0.88, 0.93, 1.05], [1.0, 1.02, 0.98]);
  let horizonColor = tint(a.horizonColor, [1.08, 0.94, 0.85], [1.0, 1.05, 1.15], [0.98, 1.03, 1.02]);
  let zenithColor = a.zenithColor;
  // Overcast: soft, cool neutral greys (greying the warm evening colors would turn them brown),
  // as bright as the sky they replace; the sun is only a diffuse glow.
  // By night the rain is a deep blue-indigo dark, not grey: the warm lights glow against it.
  const toward = (c: number[], target: number[], dim = 1) => {
    const lum = (c[0] + c[1] + c[2]) / 3;
    return c.map((v, i) => v + (target[i] * lum * dim - v) * rain) as [number, number, number];
  };
  const mixT = (day: number[], dark: number[]) => day.map((v, i) => v + (dark[i] - v) * night);
  sunColor = toward(sunColor, mixT([0.95, 1.0, 1.08], [0.7, 0.85, 1.3])).map((v) => v * (1 - 0.7 * rain)) as [number, number, number];
  horizonColor = toward(horizonColor, mixT([0.97, 1.0, 1.04], [0.55, 0.7, 1.45]), 1 - 0.35 * night);
  zenithColor = toward(zenithColor, mixT([0.9, 0.98, 1.12], [0.45, 0.6, 1.6]), 1 - 0.3 * night);
  return {
    ...a,
    sunColor,
    horizonColor,
    zenithColor,
    fogDensity: a.fogDensity * (1 + (1.6 - 0.9 * night) * rain + 0.3 * wi),
    exposure: a.exposure * (1 + 0.25 * rain * (1 - night)),
  };
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

/** First light: low sun in the east, rose horizon, a cool sky still waking. */
export const DAWN: Atmosphere = {
  sunDir: normalize([-0.35, 0.12, 0.9]),
  sunColor: [2.1, 1.35, 1.05],
  horizonColor: [0.92, 0.6, 0.58],
  zenithColor: [0.2, 0.28, 0.58],
  fogDensity: 0.0024,
  exposure: 1.05,
};

/** Midday: high warm-white sun, clear blue sky. */
export const DAY: Atmosphere = {
  sunDir: normalize([0.2, 0.85, -0.3]),
  sunColor: [2.5, 2.25, 1.95],
  horizonColor: [0.78, 0.82, 0.9],
  zenithColor: [0.17, 0.38, 0.84],
  fogDensity: 0.0014,
  exposure: 0.82,
};

/** After sunset: the light gone violet, the first stars. */
export const DUSK: Atmosphere = {
  sunDir: normalize([0.36, 0.02, -0.93]),
  sunColor: [1.1, 0.5, 0.42],
  horizonColor: [0.58, 0.3, 0.36],
  zenithColor: [0.07, 0.1, 0.3],
  fogDensity: 0.0025,
  exposure: 1.3,
};

/**
 * The day as keyframes over its phase in [0, 1): dawn, a long morning into day, a long golden
 * hour (the field's best light), dusk, then night until the next dawn.
 */
function dayKeys(): [number, Atmosphere][] {
  return [
    [0.0, DAWN],
    [0.1, DAY],
    [0.32, DAY],
    [0.44, GOLDEN_HOUR],
    [0.56, GOLDEN_HOUR],
    [0.62, DUSK],
    [0.68, NIGHT],
    [0.94, NIGHT],
    [1.0, DAWN],
  ];
}

/** Sun's progress along its arc (0 rising .. pi setting) by phase: a long, low golden hour. */
const SUN_ARC: [number, number][] = [
  [-0.03, 0],
  [0.2, 0.5],
  [0.44, 0.86],
  [0.56, 0.94],
  [0.63, 1.02],
];

function sunTheta(p: number): number {
  const q = p > 0.95 ? p - 1 : p;
  for (let i = 0; i < SUN_ARC.length - 1; i++) {
    const [a, ta] = SUN_ARC[i];
    const [b, tb] = SUN_ARC[i + 1];
    if (q >= a && q <= b) return Math.PI * (ta + ((tb - ta) * (q - a)) / (b - a));
  }
  return q < SUN_ARC[0][0] ? -0.1 : Math.PI * 1.02;
}

/**
 * The atmosphere at a time of day (`phase` in [0, 1)), plus how much it is night (0..1). By
 * day the sun follows an arc from its rising point to its setting point (the golden-hour sun);
 * at night the moon takes its place.
 */
export function dayAtmosphere(phase: number): { atm: Atmosphere; night: number } {
  const p = ((phase % 1) + 1) % 1;
  let i = 0;
  const keys = dayKeys();
  while (i < keys.length - 2 && p >= keys[i + 1][0]) i++;
  const [p0, a0] = keys[i];
  const [p1, a1] = keys[i + 1];
  const k = p1 > p0 ? (p - p0) / (p1 - p0) : 0;
  const atm = mixAtmosphere(a0, a1, k * k * (3 - 2 * k));
  // Night weight: dusk -> night, night -> dawn.
  const ramp = (a: number, b: number) => Math.min(1, Math.max(0, (p - a) / (b - a)));
  let night = ramp(0.6, 0.68) * (1 - ramp(0.94, 1.0));
  night = night * night * (3 - 2 * night);
  // Sun arc: rising opposite its setting point, peaking high toward the south, setting where
  // the golden-hour sun sits.
  const set = GOLDEN_HOUR.sunDir;
  const sh = normalize([set[0], 0, set[2]]);
  const side = normalize([-sh[2], 0, sh[0]]);
  const up = normalize([side[0] * 0.55, 0.85, side[2] * 0.55]);
  const theta = sunTheta(p);
  const sun = normalize([-sh[0] * Math.cos(theta) + up[0] * Math.sin(theta), up[1] * Math.sin(theta) + 0.03, -sh[2] * Math.cos(theta) + up[2] * Math.sin(theta)]);
  const dir = mixAtmosphere({ ...atm, sunDir: sun }, { ...atm, sunDir: NIGHT.sunDir }, night).sunDir;
  return { atm: { ...atm, sunDir: dir }, night };
}

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
      underwater: 0,
      waterY: -1000,
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
