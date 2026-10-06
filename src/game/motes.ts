import { draw, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import moteShader from "../shaders/motes.wgsl";
import type { Vec3 } from "../engine/camera";
import { terrainHeightM as terrainHeight } from "../world/height";

const MAX = 384;
const STRIDE = 12;
/** Most things the wind holds at once. */
const MAX_CARRIED = 70;
const PATH_STEP = 0.12;
const PATH_LEN = 600;

export const enum Kind {
  Leaf = 0,
  Fluff = 1,
  Chaff = 2,
  Snow = 3,
  Spray = 4,
  Pollen = 5,
  Firefly = 6,
}

/** What the land under the wind is like, sampled each frame. */
export interface Surroundings {
  altitude: number;
  /** 0 open meadow .. 1 under trees. */
  forest: number;
  /** 1 over / beside the river. */
  river: number;
  /** 1 inside the sunflower field. */
  sunflowers: number;
  /** summer, autumn, winter, spring weights. */
  season: [number, number, number, number];
  night: number;
  rain: number;
}

interface Mote {
  pos: Vec3;
  vel: Vec3;
  color: [number, number, number];
  kind: Kind;
  seed: number;
  size: number;
  age: number;
  /** Seconds held by the wind before it lets go (0 = loose from the start). */
  hold: number;
  carried: boolean;
  /** Seconds left once settled or fading. */
  life: number;
  slot: number;
  landed: boolean;
}

const SIZES: Record<Kind, [number, number]> = {
  [Kind.Leaf]: [0.1, 0.05],
  [Kind.Fluff]: [0.08, 0.04],
  [Kind.Chaff]: [0.08, 0.04],
  [Kind.Snow]: [0.04, 0.02],
  [Kind.Spray]: [0.03, 0.015],
  [Kind.Pollen]: [0.022, 0.01],
  [Kind.Firefly]: [0.045, 0.015],
};

function pick<T>(items: [T, number][]): T {
  let total = 0;
  for (const [, w] of items) total += w;
  let r = Math.random() * total;
  for (const [it, w] of items) {
    r -= w;
    if (r <= 0) return it;
  }
  return items[0][0];
}

/**
 * What the wind carries: it lifts whatever the land offers (seeds and chaff from the meadow,
 * leaves in the forest or in autumn, snow in winter, spray over the river or in the rain,
 * pollen in the sunflowers, fireflies at night), holds it a while in its wake, then lets it
 * go. Seeds that land take root: `onSeed` is called where they come down.
 */
export class Motes {
  readonly draw: Draw;
  private readonly motes: Mote[] = [];
  private readonly path: Vec3[] = [];
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(MAX * STRIDE);
  private spawnDebt = 0;
  private time = 0;
  onSeed: ((x: number, z: number) => void) | null = null;

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.buffer = storage(gpu, MAX * STRIDE * 4, "read");
    this.draw = draw(gpu, {
      label: "motes",
      shader: moteShader,
      vertices: 6,
      cull: "none",
      depth: { compare: "greater" },
      multisample: { alphaToCoverage: true },
      set: { G: globals, motes: this.buffer },
    });
  }

  /** How many things the wind is holding. */
  get carried(): number {
    let n = 0;
    for (const m of this.motes) if (m.carried) n++;
    return n;
  }

  /** Drops everything (after a jump). */
  reset(): void {
    this.motes.length = 0;
    this.path.length = 0;
  }

  /** A puff of pollen from a flower the wind brushed. */
  puff(at: Vec3, color: [number, number, number]): void {
    for (let i = 0; i < 6; i++) {
      this.spawn(Kind.Pollen, [at[0], at[1], at[2]], [color[0] * 1.2 + 0.2, color[1] * 1.2 + 0.2, color[2] * 1.2 + 0.1], 2 + Math.random() * 2);
    }
  }

  private spawn(kind: Kind, pos: Vec3, color: [number, number, number], hold: number): void {
    if (this.motes.length >= MAX) {
      // Recycle the oldest loose one.
      const i = this.motes.findIndex((m) => !m.carried);
      if (i < 0) return;
      this.motes.splice(i, 1);
    }
    const [s0, s1] = SIZES[kind];
    this.motes.push({
      pos,
      vel: [(Math.random() - 0.5) * 2, 1 + Math.random(), (Math.random() - 0.5) * 2],
      color,
      kind,
      seed: Math.random(),
      size: s0 + Math.random() * s1,
      age: 0,
      hold,
      carried: hold > 0,
      life: 4,
      slot: Math.random(),
      landed: false,
    });
  }

  /** What the land offers here, with its colour. */
  private offer(env: Surroundings): [Kind, [number, number, number]] {
    const [su, au, wi, sp] = env.season;
    const kind = pick<Kind>([
      [Kind.Fluff, (su + sp * 1.4) * (1 - env.forest) * (1 - env.sunflowers) + 0.05],
      [Kind.Chaff, (su + au * 0.6) * (1 - env.forest) * 0.8],
      [Kind.Leaf, env.forest * (0.4 + au * 1.6) + au * 0.7 * (1 - env.forest)],
      [Kind.Snow, wi * 2.5],
      [Kind.Spray, env.river * 2.5 + env.rain * 1.5],
      [Kind.Pollen, env.sunflowers * (1 - wi) * 2.5],
      [Kind.Firefly, env.night > 0.5 ? (1 - wi) * 0.6 : 0],
    ]);
    const r = Math.random();
    switch (kind) {
      case Kind.Leaf: {
        const autumn: [number, number, number][] = [[0.75, 0.25, 0.06], [0.85, 0.5, 0.08], [0.6, 0.14, 0.08], [0.8, 0.62, 0.2]];
        const green: [number, number, number] = [0.25 + r * 0.1, 0.4 + r * 0.1, 0.1];
        return [kind, Math.random() < au + 0.15 ? autumn[Math.floor(r * 4)] : green];
      }
      case Kind.Fluff:
        return [kind, [0.95, 0.94, 0.9]];
      case Kind.Chaff:
        return [kind, [0.75 + r * 0.15, 0.62 + r * 0.1, 0.32]];
      case Kind.Snow:
        return [kind, [0.95, 0.97, 1.0]];
      case Kind.Spray:
        return [kind, [0.75, 0.85, 0.95]];
      case Kind.Pollen:
        return [kind, [1.0, 0.78, 0.2]];
      default:
        return [kind, [0.75, 1.0, 0.35]];
    }
  }

  private recordPath(leader: Vec3): void {
    const last = this.path[0];
    if (!last) {
      this.path.unshift([leader[0], leader[1], leader[2]]);
      return;
    }
    const d = Math.hypot(leader[0] - last[0], leader[1] - last[1], leader[2] - last[2]);
    if (d < PATH_STEP) return;
    const steps = Math.min(Math.floor(d / PATH_STEP), 50);
    for (let s = 1; s <= steps; s++) {
      const k = (s * PATH_STEP) / d;
      this.path.unshift([last[0] + (leader[0] - last[0]) * k, last[1] + (leader[1] - last[1]) * k, last[2] + (leader[2] - last[2]) * k]);
    }
    if (this.path.length > PATH_LEN) this.path.length = PATH_LEN;
  }

  update(dt: number, leader: Vec3, forward: Vec3, speed: number, gust: number, env: Surroundings, windDir: [number, number]): void {
    this.time += dt;
    this.recordPath(leader);

    // Skimming low and fast, the wind lifts things up from the land beneath it.
    const low = 1 - Math.min(1, Math.max(0, (env.altitude - 2.5) / 3.5));
    const rate = low * (0.6 + speed * 0.18 + gust * 4);
    this.spawnDebt += rate * dt;
    while (this.spawnDebt >= 1) {
      this.spawnDebt -= 1;
      if (this.carried >= MAX_CARRIED) continue;
      const [kind, color] = this.offer(env);
      const ground = terrainHeight(leader[0], leader[2]);
      const pos: Vec3 = [leader[0] + (Math.random() - 0.5) * 1.5, ground + 0.1 + Math.random() * 0.4, leader[2] + (Math.random() - 0.5) * 1.5];
      this.spawn(kind, pos, color, kind === Kind.Spray ? 0.6 + Math.random() : 4 + Math.random() * 8);
    }

    const side: Vec3 = [-forward[2], 0, forward[0]];
    const length = 2 + 5 * Math.sqrt(this.carried / MAX_CARRIED) * (1 + gust * 0.5);
    const d = this.data;
    let o = 0;
    for (let i = this.motes.length - 1; i >= 0; i--) {
      const m = this.motes[i];
      m.age += dt;
      if (m.carried && m.age > m.hold) m.carried = false;
      if (m.carried) {
        // Held in the wake: an orbit around a slot on the wind's path.
        const s = 0.5 + m.slot * length;
        const base = this.path[Math.min(this.path.length - 1, Math.floor(s / PATH_STEP))] ?? leader;
        const r = 0.3 + 0.6 * m.slot + gust * 0.25;
        const w = this.time * (1.4 + m.seed * 1.8) + m.seed * 50;
        const target: Vec3 = [base[0] + side[0] * Math.cos(w) * r, base[1] + Math.sin(w * 1.3) * r * 0.6, base[2] + side[2] * Math.cos(w) * r];
        const a = 1 - Math.exp(-(2.5 + m.seed * 3) * dt * Math.min(1, m.age * 1.5));
        for (let j = 0; j < 3; j++) {
          const next = m.pos[j] + (target[j] - m.pos[j]) * a;
          m.vel[j] = (next - m.pos[j]) / Math.max(dt, 1e-4);
          m.pos[j] = next;
        }
      } else if (!m.landed) {
        // Let go: each thing falls in its own way, drifting with the breeze.
        const fall = [1.1, 0.25, 0.7, 0.5, 9.8, 0.08, 0][m.kind];
        const drag = [1.6, 2.5, 1.8, 2.2, 0.4, 3, 1.5][m.kind];
        const breeze = [windDir[0] * 1.2, windDir[1] * 1.2];
        m.vel[0] += (breeze[0] - m.vel[0]) * drag * dt;
        m.vel[2] += (breeze[1] - m.vel[2]) * drag * dt;
        if (m.kind === Kind.Spray) m.vel[1] -= fall * dt;
        else m.vel[1] += (-fall - m.vel[1]) * drag * dt;
        if (m.kind === Kind.Leaf) {
          // Leaves see-saw down.
          const sw = Math.sin(this.time * 2.6 + m.seed * 20);
          m.vel[0] += side[0] * sw * 1.5 * dt * 4;
          m.vel[2] += side[2] * sw * 1.5 * dt * 4;
        }
        if (m.kind === Kind.Firefly) {
          m.vel[1] += Math.sin(this.time * 1.3 + m.seed * 30) * dt * 0.8;
          m.life -= dt * 0.25;
        }
        if (m.kind === Kind.Pollen) m.life -= dt * 0.8;
        for (let j = 0; j < 3; j++) m.pos[j] += m.vel[j] * dt;
        const ground = terrainHeight(m.pos[0], m.pos[2]);
        if (m.pos[1] < ground + 0.03) {
          m.pos[1] = ground + 0.03;
          m.landed = true;
          m.life = m.kind === Kind.Leaf ? 6 : m.kind === Kind.Chaff ? 3 : 0.4;
          // A seed that comes down takes root.
          if (m.kind === Kind.Fluff && Math.random() < 0.5) this.onSeed?.(m.pos[0], m.pos[2]);
        }
      } else m.life -= dt;
      if (m.life <= 0 || Math.hypot(m.pos[0] - leader[0], m.pos[2] - leader[2]) > 120) {
        this.motes.splice(i, 1);
        continue;
      }
      const fade = Math.min(0.999, m.life / 1.2, m.age * 3);
      d[o++] = m.pos[0]; d[o++] = m.pos[1]; d[o++] = m.pos[2]; d[o++] = m.size;
      d[o++] = m.color[0]; d[o++] = m.color[1]; d[o++] = m.color[2]; d[o++] = m.seed;
      d[o++] = m.vel[0]; d[o++] = m.vel[1]; d[o++] = m.vel[2]; d[o++] = m.kind + Math.max(0, fade);
    }
    this.count = o / STRIDE;
    if (o) this.buffer.write(d.subarray(0, o));
  }

  private count = 0;

  encode(pass: FramePass): void {
    if (this.count) pass.draw(this.draw, { instances: this.count });
  }
}
