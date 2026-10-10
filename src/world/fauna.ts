import { storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import animalShader from "../shaders/animals.wgsl";
import type { Vec3 } from "../engine/camera";
import { BIRD_STRIDE as STRIDE, BirdAnimator, BirdSpecies } from "./bird-model";
import { ecology } from "./ecology";
import { riverCenter, riverHalfWidth, riverInfo, riverUpper, riverWater, terrainHeightM as terrainHeight } from "./height";
import { pathNear } from "./lantern-path";
import { sunflowerNear } from "./sunflower-field";
import { BED_CELL, bedInCell } from "./bed-shape";

/** What the animals react to, each frame. */
export interface FaunaContext {
  dt: number;
  /** The wind (or the swallow): where it is, its velocity and height above the ground. */
  wind: Vec3;
  windVel: Vec3;
  windAlt: number;
  heading: number;
  night: number;
  rain: number;
  /** Seconds of world time. */
  t: number;
}

function rand(a: number, b: number): number {
  return a + Math.random() * (b - a);
}

function angleTo(from: number, to: number): number {
  return Math.atan2(Math.sin(to - from), Math.cos(to - from));
}

/** Open, gentle ground clear of the river, the lantern path and the sunflower field. */
export function openGround(x: number, z: number, maxForest = 0.25, maxSlope = 0.28): boolean {
  const [d, , hw] = riverInfo(x, z);
  if (d < hw + 4) return false;
  if (pathNear(x, z, 3) || sunflowerNear(x, z, 3)) return false;
  const eco = ecology(x, z);
  return eco.forest <= maxForest && eco.slope <= maxSlope;
}

/** One species' instances: a skinned draw and its per-frame instance data. */
class Herd {
  readonly body: Draw;
  readonly halo: Draw;
  readonly data: Float32Array<ArrayBuffer>;
  count = 0;
  constructor(
    readonly species: BirdSpecies,
    private readonly buffer: StorageBuffer,
    globals: SharedUniforms,
    label: string,
    kind: number,
    readonly max: number,
    alpha = false,
  ) {
    species.shader = animalShader;
    species.instanceKey = "animals";
    const d = species.draws(globals, buffer, label, { constants: { KIND: kind }, alphaToCoverage: alpha });
    this.body = d.body;
    this.halo = d.halo;
    this.data = new Float32Array(max * STRIDE);
  }

  static async load(gpu: Gpu, globals: SharedUniforms, name: string, kind: number, max: number, alpha = false): Promise<Herd> {
    const species = await BirdSpecies.load(gpu, name, "assets/animals");
    return new Herd(species, storage(gpu, max * STRIDE * 4, "read"), globals, name, kind, max, alpha);
  }

  begin(): void {
    this.count = 0;
  }

  /** Adds an instance (position, heading, pitch, roll, size, glow, its animation, random). */
  push(pos: Vec3, yaw: number, pitch: number, roll: number, size: number, glow: number, anim: BirdAnimator, seed: number): void {
    if (this.count >= this.max) return;
    const o = this.count * STRIDE;
    const d = this.data;
    d[o] = pos[0];
    d[o + 1] = pos[1];
    d[o + 2] = pos[2];
    d[o + 3] = yaw;
    d[o + 4] = pitch;
    d[o + 5] = roll;
    d[o + 6] = size;
    d[o + 7] = glow;
    anim.write(d, o + 8);
    d[o + 11] = seed;
    this.count++;
  }

  end(): void {
    if (this.count) this.buffer.write(this.data.subarray(0, this.count * STRIDE));
  }

  encode(pass: FramePass, halo = false): void {
    if (!this.count) return;
    pass.draw(this.body, { instances: this.count });
    if (halo) pass.draw(this.halo, { instances: this.count });
  }
}

/** Ground pitch along a heading (positive dips the head, as the shader wants). */
function groundPitch(x: number, z: number, yaw: number, reach: number): number {
  const fx = Math.sin(yaw) * reach;
  const fz = Math.cos(yaw) * reach;
  return -Math.atan2(terrainHeight(x + fx, z + fz) - terrainHeight(x - fx, z - fz), 2 * reach);
}

// =======================================================================================
// Brown hares: in ones and twos on open meadow. They graze and look about, lope a few meters
// now and then, sit up when the wind comes near, and bolt in long bounds when it passes close.

const HARE_SIZE = 2.2;
const HARES = 7;

type HareState = "graze" | "idle" | "alert" | "hop" | "run";

interface Hare {
  pos: Vec3;
  yaw: number;
  wantYaw: number;
  speed: number;
  state: HareState;
  timer: number;
  anim: BirdAnimator;
  seed: number;
}

class Hares {
  readonly list: Hare[] = [];
  private readonly strides: Record<string, number>;
  private retry = 0;

  constructor(readonly herd: Herd) {
    this.strides = (herd.species.info.stride as Record<string, number>) ?? { hop: 0.45, run: 1.4 };
  }

  private spawn(c: FaunaContext): void {
    for (let tries = 0; tries < 6; tries++) {
      const a = c.heading + rand(-1.4, 1.4);
      const r = rand(45, 150);
      const x = c.wind[0] + Math.sin(a) * r;
      const z = c.wind[2] - Math.cos(a) * r;
      if (!openGround(x, z)) continue;
      const n = Math.random() < 0.4 ? 2 : 1;
      for (let i = 0; i < n && this.list.length < HARES; i++) {
        const px = x + rand(-4, 4);
        const pz = z + rand(-4, 4);
        const yaw = rand(0, Math.PI * 2);
        this.list.push({
          pos: [px, terrainHeight(px, pz), pz],
          yaw,
          wantYaw: yaw,
          speed: 0,
          state: "graze",
          timer: rand(2, 8),
          anim: new BirdAnimator(this.herd.species.clips, "graze", Math.random() * 3),
          seed: Math.random(),
        });
      }
      return;
    }
    this.retry = 1.5;
  }

  private set(h: Hare, state: HareState, timer: number): void {
    h.state = state;
    h.timer = timer;
    const fade = state === "run" ? 0.15 : 0.35;
    h.anim.play(state, fade, 1, false, Math.random());
  }

  update(c: FaunaContext): void {
    const dt = c.dt;
    this.retry -= dt;
    if (this.list.length < HARES && this.retry <= 0) this.spawn(c);
    const herd = this.herd;
    herd.begin();
    for (let i = this.list.length - 1; i >= 0; i--) {
      const h = this.list[i];
      const dx = h.pos[0] - c.wind[0];
      const dz = h.pos[2] - c.wind[2];
      const d = Math.hypot(dx, dz);
      if (d > 230) {
        this.list.splice(i, 1);
        continue;
      }
      const low = c.windAlt < 7;
      const away = Math.atan2(dx, dz);
      h.timer -= dt;
      if (low && d < 14 && h.state !== "run") {
        // Bolting: away from the wind, jinking a little.
        this.set(h, "run", rand(2.5, 4.5));
        h.wantYaw = away + rand(-0.5, 0.5);
      } else if (low && d < 28 && (h.state === "graze" || h.state === "idle")) {
        // Something is coming: sit up and watch it.
        this.set(h, "alert", rand(2, 4));
        h.wantYaw = away + Math.PI + rand(-0.6, 0.6);
      } else if (h.timer <= 0) {
        if (h.state === "run") this.set(h, "hop", rand(1.5, 3));
        else if (h.state === "hop") this.set(h, Math.random() < 0.5 ? "alert" : "graze", rand(2, 5));
        else {
          const r = Math.random();
          if (r < 0.45) this.set(h, "graze", rand(4, 10));
          else if (r < 0.75) this.set(h, "idle", rand(3, 6));
          else {
            this.set(h, "hop", rand(1.2, 3));
            h.wantYaw = h.yaw + rand(-1.5, 1.5);
          }
        }
      }
      if (h.state === "run" && Math.random() < dt * 0.8) h.wantYaw += rand(-0.6, 0.6);
      // Never run into the river: turn along the bank.
      const ax = h.pos[0] + Math.sin(h.wantYaw) * 3;
      const az = h.pos[2] + Math.cos(h.wantYaw) * 3;
      const [rd, , rhw] = riverInfo(ax, az);
      if (rd < rhw + 1.5) h.wantYaw += Math.PI * 0.5;
      h.yaw += angleTo(h.yaw, h.wantYaw) * Math.min(1, dt * (h.state === "run" ? 5 : 2.5));
      const want = h.state === "run" ? 10 : h.state === "hop" ? 1.8 : 0;
      h.speed += (want - h.speed) * Math.min(1, dt * (want > h.speed ? 4 : 3));
      h.pos[0] += Math.sin(h.yaw) * h.speed * dt;
      h.pos[2] += Math.cos(h.yaw) * h.speed * dt;
      h.pos[1] = terrainHeight(h.pos[0], h.pos[2]);
      // The gait's playback rate keeps the feet on the ground.
      const stride = this.strides[h.state];
      h.anim.speed = stride ? Math.max(0.3, h.speed / (stride * HARE_SIZE)) : 1;
      h.anim.update(dt);
      if (c.night < 0.9) herd.push(h.pos, h.yaw, groundPitch(h.pos[0], h.pos[2], h.yaw, 0.6), 0, HARE_SIZE, 0, h.anim, h.seed);
    }
    herd.end();
  }
}

// =======================================================================================
// Butterflies: dancing over the flower beds and here and there over the meadow, settling on
// the flowers to bask (wings opening and closing), swept up and tumbled by a passing gust.

const FLY_SIZE = 3;
const BUTTERFLIES = 36;

interface Butterfly {
  pos: Vec3;
  vel: Vec3;
  yaw: number;
  home: Vec3;
  range: number;
  target: Vec3;
  basking: number;
  phase: number;
  anim: BirdAnimator;
  seed: number;
  /** Over a flower bed (it settles there). */
  bed: boolean;
}

class Butterflies {
  readonly list: Butterfly[] = [];
  private scanAt: Vec3 | null = null;

  constructor(readonly herd: Herd) {}

  private add(home: Vec3, range: number, bed: boolean, species: number | null): void {
    const p: Vec3 = [home[0] + rand(-range, range), home[1] + rand(0.4, 1.4), home[2] + rand(-range, range)];
    this.list.push({
      pos: p,
      vel: [0, 0, 0],
      yaw: rand(0, 6.28),
      home,
      range,
      target: [...p],
      basking: 0,
      phase: Math.random() * 10,
      anim: new BirdAnimator(this.herd.species.clips, "flap", Math.random()),
      // The shader picks the species from the random (6 of them): a bed keeps to a couple.
      seed: species === null ? Math.random() : ((species + (Math.random() < 0.3 ? 1 : 0)) % 6) / 6 + 0.08,
      bed,
    });
  }

  /** Fills in butterflies around the flower beds near the wind (and a few over the meadow). */
  private scan(c: FaunaContext): void {
    this.scanAt = [...c.wind];
    const cx0 = Math.floor(c.wind[0] / BED_CELL);
    const cz0 = Math.floor(c.wind[2] / BED_CELL);
    const beds: Vec3[] = [];
    for (let dz = -2; dz <= 2; dz++) {
      for (let dx = -2; dx <= 2; dx++) {
        const b = bedInCell(cx0 + dx, cz0 + dz);
        if (!b) continue;
        const near = this.list.some((f) => f.bed && Math.hypot(f.home[0] - b.x, f.home[2] - b.z) < 1);
        if (!near) beds.push([b.x, terrainHeight(b.x, b.z), b.z, b.radius, b.species] as unknown as Vec3);
      }
    }
    for (const b of beds) {
      const bb = b as unknown as number[];
      const n = 3 + Math.floor(Math.random() * 3);
      for (let i = 0; i < n && this.list.length < BUTTERFLIES; i++) this.add([bb[0], bb[1], bb[2]], bb[3] * 0.8, true, bb[4]);
    }
    // A few wanderers over open meadow.
    const free = this.list.filter((f) => !f.bed).length;
    for (let i = free; i < 6 && this.list.length < BUTTERFLIES; i++) {
      const a = c.heading + rand(-1.6, 1.6);
      const r = rand(15, 70);
      const x = c.wind[0] + Math.sin(a) * r;
      const z = c.wind[2] - Math.cos(a) * r;
      if (openGround(x, z, 0.4, 0.4)) this.add([x, terrainHeight(x, z), z], 8, false, null);
    }
  }

  update(c: FaunaContext): void {
    const dt = c.dt;
    // Gone with the night and the rain (they shelter in the grass).
    const hidden = c.night > 0.6 || c.rain > 0.5;
    if (!this.scanAt || Math.hypot(c.wind[0] - this.scanAt[0], c.wind[2] - this.scanAt[2]) > 25) this.scan(c);
    const herd = this.herd;
    herd.begin();
    const windSpeed = Math.hypot(c.windVel[0], c.windVel[2]);
    for (let i = this.list.length - 1; i >= 0; i--) {
      const f = this.list[i];
      if (Math.hypot(f.home[0] - c.wind[0], f.home[2] - c.wind[2]) > 170) {
        this.list.splice(i, 1);
        continue;
      }
      const g = terrainHeight(f.pos[0], f.pos[2]);
      const toWind = Math.hypot(f.pos[0] - c.wind[0], f.pos[1] - c.wind[1], f.pos[2] - c.wind[2]);
      const gusted = toWind < 4.5 && windSpeed > 3;
      if (f.basking > 0) {
        // Settled on a flower: bask, then lift off (at once if the wind brushes it).
        f.basking -= dt;
        if (gusted) f.basking = 0;
        if (f.basking <= 0) f.anim.play("flap", 0.1, 7, true);
      } else {
        // Flight: toward a wandering target, with the butterfly's bobbing, jinking flutter.
        f.phase += dt;
        const tx = f.target[0] - f.pos[0];
        const ty = f.target[1] - f.pos[1];
        const tz = f.target[2] - f.pos[2];
        const td = Math.hypot(tx, ty, tz);
        if (td < 0.4 || Math.random() < dt * 0.15) {
          const settle = f.bed && Math.random() < 0.45 && td < 0.4;
          if (settle) {
            // Land on a flower head and bask for a while.
            f.basking = rand(3, 9);
            f.vel = [0, 0, 0];
            f.pos[1] = g + 0.55;
            f.anim.play("bask", 0.3, rand(0.8, 1.2), true, rand(0, 3));
          } else {
            f.target = [f.home[0] + rand(-f.range, f.range), 0, f.home[2] + rand(-f.range, f.range)];
            const tg = terrainHeight(f.target[0], f.target[2]);
            f.target[1] = tg + (f.bed && Math.random() < 0.5 ? 0.58 : rand(0.7, 2.2));
          }
        }
        if (f.basking <= 0) {
          const k = 1.6 / Math.max(td, 0.3);
          const bob = Math.sin(f.phase * 9 + f.seed * 20) * 1.4;
          const jink = Math.sin(f.phase * 5.3 + f.seed * 7) * 1.1;
          const want: Vec3 = [tx * k + Math.cos(f.yaw) * jink, ty * k * 1.4 + bob, tz * k - Math.sin(f.yaw) * jink];
          for (let j = 0; j < 3; j++) f.vel[j] += (want[j] - f.vel[j]) * Math.min(1, dt * 3);
          if (gusted) {
            // Swept along and tumbled by the wind.
            for (let j = 0; j < 3; j++) f.vel[j] += c.windVel[j] * dt * 2.5;
            f.vel[1] += dt * 4;
          }
          for (let j = 0; j < 3; j++) f.pos[j] += f.vel[j] * dt;
          f.pos[1] = Math.max(f.pos[1], g + 0.35);
          const hs = Math.hypot(f.vel[0], f.vel[2]);
          if (hs > 0.15) f.yaw += angleTo(f.yaw, Math.atan2(f.vel[0], f.vel[2])) * Math.min(1, dt * 6);
          // Bursts of flapping with short glides.
          const glide = Math.sin(f.phase * 2.1 + f.seed * 9) > 0.75 && f.vel[1] < 0.5;
          f.anim.play(glide ? "glide" : "flap", 0.12, glide ? 1 : 6.5 + f.seed * 2);
        }
      }
      f.anim.update(dt);
      if (!hidden) herd.push(f.pos, f.yaw, f.basking > 0 ? 0 : -f.vel[1] * 0.08, 0, FLY_SIZE, 0, f.anim, f.seed);
    }
    herd.end();
  }
}

// =======================================================================================
// Roe deer: small groups at the edges of the woods and in quiet meadows. They graze, lift their
// heads to look about, walk on a few steps; as the wind comes near they stare, then bound away.

const DEER_SIZE = 1.4;
const DEER = 6;

type DeerState = "graze" | "idle" | "alert" | "walk" | "run";

interface Deer {
  pos: Vec3;
  yaw: number;
  wantYaw: number;
  speed: number;
  state: DeerState;
  timer: number;
  anim: BirdAnimator;
  seed: number;
}

/** Shared by the ground animals: steering, ground following, gait rate. */
function moveOnGround(a: { pos: Vec3; yaw: number; wantYaw: number; speed: number }, want: number, turn: number, dt: number): void {
  // Keep out of the river: turn along the bank.
  const ax = a.pos[0] + Math.sin(a.wantYaw) * 4;
  const az = a.pos[2] + Math.cos(a.wantYaw) * 4;
  const [rd, , rhw] = riverInfo(ax, az);
  if (rd < rhw + 1.5) a.wantYaw += Math.PI * 0.5;
  a.yaw += angleTo(a.yaw, a.wantYaw) * Math.min(1, dt * turn);
  a.speed += (want - a.speed) * Math.min(1, dt * (want > a.speed ? 3 : 2.5));
  a.pos[0] += Math.sin(a.yaw) * a.speed * dt;
  a.pos[2] += Math.cos(a.yaw) * a.speed * dt;
  a.pos[1] = terrainHeight(a.pos[0], a.pos[2]);
}

class DeerHerds {
  readonly list: Deer[] = [];
  private readonly strides: Record<string, number>;
  private retry = 0;

  constructor(readonly herd: Herd) {
    this.strides = (herd.species.info.stride as Record<string, number>) ?? { walk: 0.9, run: 2.6 };
  }

  private spawn(c: FaunaContext): void {
    for (let tries = 0; tries < 6; tries++) {
      const a = c.heading + rand(-1.6, 1.6);
      const r = rand(70, 180);
      const x = c.wind[0] + Math.sin(a) * r;
      const z = c.wind[2] - Math.cos(a) * r;
      const [d, , hw] = riverInfo(x, z);
      if (d < hw + 6 || pathNear(x, z, 4) || sunflowerNear(x, z, 4)) continue;
      const eco = ecology(x, z);
      // Edges of the woods and quiet hollows, not deep forest nor steep slopes.
      if (eco.slope > 0.3 || eco.forest > 0.7 || eco.temperature < 0.35) continue;
      const n = 2 + Math.floor(Math.random() * 3);
      for (let i = 0; i < n && this.list.length < DEER; i++) {
        const px = x + rand(-6, 6);
        const pz = z + rand(-6, 6);
        const yaw = rand(0, Math.PI * 2);
        this.list.push({ pos: [px, terrainHeight(px, pz), pz], yaw, wantYaw: yaw, speed: 0, state: "graze", timer: rand(2, 9), anim: new BirdAnimator(this.herd.species.clips, "graze", Math.random() * 4), seed: Math.random() });
      }
      return;
    }
    this.retry = 2;
  }

  private set(d: Deer, state: DeerState, timer: number): void {
    d.state = state;
    d.timer = timer;
    d.anim.play(state, state === "run" ? 0.2 : 0.5, 1, false, Math.random() * 2);
  }

  update(c: FaunaContext): void {
    const dt = c.dt;
    this.retry -= dt;
    if (this.list.length < DEER - 1 && this.retry <= 0) this.spawn(c);
    const herd = this.herd;
    herd.begin();
    for (let i = this.list.length - 1; i >= 0; i--) {
      const d = this.list[i];
      const dx = d.pos[0] - c.wind[0];
      const dz = d.pos[2] - c.wind[2];
      const dist = Math.hypot(dx, dz);
      if (dist > 260) {
        this.list.splice(i, 1);
        continue;
      }
      const near = c.windAlt < 9;
      const away = Math.atan2(dx, dz);
      d.timer -= dt;
      if (near && dist < 22 && d.state !== "run") {
        this.set(d, "run", rand(3.5, 6));
        d.wantYaw = away + rand(-0.4, 0.4);
      } else if (near && dist < 45 && (d.state === "graze" || d.state === "idle" || d.state === "walk")) {
        this.set(d, "alert", rand(3, 6));
        d.wantYaw = away + Math.PI + rand(-0.3, 0.3);
      } else if (d.timer <= 0) {
        if (d.state === "run") this.set(d, "walk", rand(2, 4));
        else {
          const r = Math.random();
          if (r < 0.5) this.set(d, "graze", rand(5, 12));
          else if (r < 0.75) this.set(d, "idle", rand(4, 8));
          else {
            this.set(d, "walk", rand(2, 5));
            d.wantYaw = d.yaw + rand(-1.2, 1.2);
          }
        }
      }
      const want = d.state === "run" ? 9 : d.state === "walk" ? 1.1 : 0;
      moveOnGround(d, want, d.state === "run" ? 3 : 1.5, dt);
      const stride = this.strides[d.state];
      d.anim.speed = stride ? Math.max(0.3, d.speed / (stride * DEER_SIZE)) : 1;
      d.anim.update(dt);
      if (c.night < 0.9) herd.push(d.pos, d.yaw, groundPitch(d.pos[0], d.pos[2], d.yaw, 1.0), 0, DEER_SIZE, 0, d.anim, d.seed);
    }
    herd.end();
  }
}

// =======================================================================================
// A red fox: hunting voles in the meadow. It trots, stops to listen with its head cocked, and
// springs high to dive nose first into the grass; it slips away at a run from the wind.

const FOX_SIZE = 1.8;
const FOXES = 2;

type FoxState = "trot" | "idle" | "sniff" | "pounce" | "run";

interface Fox {
  pos: Vec3;
  yaw: number;
  wantYaw: number;
  speed: number;
  state: FoxState;
  timer: number;
  anim: BirdAnimator;
  seed: number;
}

class Foxes {
  readonly list: Fox[] = [];
  private readonly strides: Record<string, number>;
  private retry = 0;
  /** A pounce landed (for a rustle). */
  onPounce: ((at: Vec3) => void) | null = null;

  constructor(readonly herd: Herd) {
    this.strides = (herd.species.info.stride as Record<string, number>) ?? { trot: 0.55, run: 1.5 };
  }

  private set(f: Fox, state: FoxState, timer: number): void {
    f.state = state;
    f.timer = timer;
    f.anim.play(state, state === "pounce" ? 0.2 : 0.35, 1, state === "pounce");
  }

  update(c: FaunaContext): void {
    const dt = c.dt;
    this.retry -= dt;
    if (this.list.length < FOXES && this.retry <= 0) {
      this.retry = 3;
      const a = c.heading + rand(-1.3, 1.3);
      const r = rand(50, 140);
      const x = c.wind[0] + Math.sin(a) * r;
      const z = c.wind[2] - Math.cos(a) * r;
      if (openGround(x, z, 0.5, 0.3)) {
        const yaw = rand(0, 6.28);
        this.list.push({ pos: [x, terrainHeight(x, z), z], yaw, wantYaw: yaw, speed: 0, state: "trot", timer: rand(3, 6), anim: new BirdAnimator(this.herd.species.clips, "trot"), seed: Math.random() });
      }
    }
    const herd = this.herd;
    herd.begin();
    for (let i = this.list.length - 1; i >= 0; i--) {
      const f = this.list[i];
      const dx = f.pos[0] - c.wind[0];
      const dz = f.pos[2] - c.wind[2];
      const dist = Math.hypot(dx, dz);
      if (dist > 240) {
        this.list.splice(i, 1);
        continue;
      }
      f.timer -= dt;
      if (c.windAlt < 7 && dist < 16 && f.state !== "run") {
        this.set(f, "run", rand(3, 5));
        f.wantYaw = Math.atan2(dx, dz) + rand(-0.5, 0.5);
      } else if (f.state === "pounce") {
        if (f.anim.done) {
          this.set(f, "sniff", rand(1.5, 3));
          this.onPounce?.(f.pos);
        }
      } else if (f.timer <= 0) {
        const r = Math.random();
        if (f.state === "idle" && r < 0.5) this.set(f, "pounce", 4);
        else if (r < 0.45) {
          this.set(f, "trot", rand(3, 8));
          f.wantYaw = f.yaw + rand(-1, 1);
        } else if (r < 0.75) this.set(f, "idle", rand(2, 4));
        else this.set(f, "sniff", rand(2, 5));
      }
      // The pounce carries it forward through the leap (the clip lifts it).
      const t = f.anim.time;
      const want = f.state === "run" ? 8.5 : f.state === "trot" ? 1.9 : f.state === "sniff" ? 0.5 : f.state === "pounce" && t > 1.45 && t < 2.15 ? 1.8 : 0;
      moveOnGround(f, want, f.state === "run" ? 3 : 1.6, dt);
      const stride = this.strides[f.state];
      f.anim.speed = stride ? Math.max(0.3, f.speed / (stride * FOX_SIZE)) : 1;
      f.anim.update(dt);
      if (c.night < 0.95) herd.push(f.pos, f.yaw, groundPitch(f.pos[0], f.pos[2], f.yaw, 0.6), 0, FOX_SIZE, 0, f.anim, f.seed);
    }
    herd.end();
  }
}

// =======================================================================================
// Frogs on the river banks: they sit and breathe, call at dusk and through the night (their
// throats swelling), and leap into the water with a plop when the wind comes close.

const FROG_SIZE = 3.5;
const FROGS = 10;

interface Frog {
  pos: Vec3;
  yaw: number;
  state: "sit" | "croak" | "hop" | "gone";
  timer: number;
  anim: BirdAnimator;
  seed: number;
  /** The leap into the water: from, to, progress (s). */
  leap: { from: Vec3; to: Vec3; t: number } | null;
}

class Frogs {
  readonly list: Frog[] = [];
  private retry = 0;
  onSplash: ((at: Vec3) => void) | null = null;
  onCroak: ((at: Vec3) => void) | null = null;

  constructor(readonly herd: Herd) {}

  private spawn(c: FaunaContext): void {
    const x = c.wind[0] + rand(-45, 45);
    const hw = riverHalfWidth(x);
    // Lowland water only (no frogs in the mountain stream).
    if (riverUpper(x) > 0.05) return;
    const side = Math.random() < 0.5 ? -1 : 1;
    // At the water's edge, in the shallows (the reeds behind it).
    const z = riverCenter(x) + side * (hw * 0.93 + rand(-0.3, 0.4));
    const water = riverWater(x);
    const y = Math.max(terrainHeight(x, z), water - 0.02);
    // Facing the water, mostly.
    const yaw = Math.atan2(0, -side) + rand(-0.8, 0.8);
    this.list.push({ pos: [x, y, z], yaw, state: "sit", timer: rand(2, 10), anim: new BirdAnimator(this.herd.species.clips, "sit", Math.random() * 3), seed: Math.random(), leap: null });
  }

  update(c: FaunaContext): void {
    const dt = c.dt;
    const [rd, , rhw] = riverInfo(c.wind[0], c.wind[2]);
    this.retry -= dt;
    if (rd < rhw + 80 && this.list.length < FROGS && this.retry <= 0) {
      this.retry = 0.5;
      this.spawn(c);
    }
    const dusk = c.night > 0.25;
    const herd = this.herd;
    herd.begin();
    for (let i = this.list.length - 1; i >= 0; i--) {
      const f = this.list[i];
      const dist = Math.hypot(f.pos[0] - c.wind[0], f.pos[2] - c.wind[2]);
      if (dist > 90 || f.state === "gone") {
        this.list.splice(i, 1);
        continue;
      }
      f.timer -= dt;
      if (f.state !== "hop" && dist < 7 && c.windAlt < 5) {
        // Into the water: a leap toward the middle of the river.
        const cz = riverCenter(f.pos[0]);
        const to: Vec3 = [f.pos[0] + rand(-0.5, 0.5), riverWater(f.pos[0]) - 0.05, f.pos[2] + Math.sign(cz - f.pos[2]) * 1.3];
        f.leap = { from: [...f.pos], to, t: 0 };
        f.yaw = Math.atan2(to[0] - f.pos[0], to[2] - f.pos[2]);
        f.state = "hop";
        f.anim.play("hop", 0.05, 1, true);
      } else if (f.state === "sit" && f.timer <= 0) {
        if (dusk || Math.random() < 0.15) {
          f.state = "croak";
          f.timer = rand(1.6, 3.2);
          f.anim.play("croak", 0.2, 1, true);
          this.onCroak?.(f.pos);
        } else f.timer = rand(3, 9);
      } else if (f.state === "croak" && f.timer <= 0) {
        f.state = "sit";
        f.timer = rand(dusk ? 2 : 6, dusk ? 7 : 16);
        f.anim.play("sit", 0.3);
      }
      if (f.leap) {
        f.leap.t += dt;
        const u = Math.min(1, f.leap.t / 0.5);
        for (let j = 0; j < 3; j++) f.pos[j] = f.leap.from[j] + (f.leap.to[j] - f.leap.from[j]) * u;
        f.pos[1] += Math.sin(Math.PI * u) * 0.5;
        if (u >= 1) {
          this.onSplash?.(f.pos);
          f.state = "gone";
        }
      }
      f.anim.update(dt);
      herd.push(f.pos, f.yaw, 0, 0, FROG_SIZE, 0, f.anim, f.seed);
    }
    herd.end();
  }
}

// =======================================================================================
// Dragonflies over the river: they hang in the air, dart to a new spot, and patrol the reach,
// wings a blur; by day only.

const DRAGON_SIZE = 3.2;
const DRAGONFLIES = 7;

interface Dragonfly {
  pos: Vec3;
  yaw: number;
  target: Vec3;
  hover: number;
  anim: BirdAnimator;
  seed: number;
}

class Dragonflies {
  readonly list: Dragonfly[] = [];

  constructor(readonly herd: Herd) {}

  private pick(x: number): Vec3 {
    const hw = riverHalfWidth(x);
    const z = riverCenter(x) + rand(-hw * 1.1, hw * 1.1);
    return [x, riverWater(x) + rand(0.35, 1.4), z];
  }

  update(c: FaunaContext): void {
    const dt = c.dt;
    const [rd, , rhw] = riverInfo(c.wind[0], c.wind[2]);
    if (rd < rhw + 70 && this.list.length < DRAGONFLIES && riverUpper(c.wind[0]) < 0.3) {
      const p = this.pick(c.wind[0] + rand(-35, 35));
      this.list.push({ pos: p, yaw: rand(0, 6.28), target: [...p], hover: rand(0.5, 2), anim: new BirdAnimator(this.herd.species.clips, "buzz", Math.random()), seed: Math.random() });
    }
    const herd = this.herd;
    herd.begin();
    for (let i = this.list.length - 1; i >= 0; i--) {
      const d = this.list[i];
      if (Math.hypot(d.pos[0] - c.wind[0], d.pos[2] - c.wind[2]) > 80) {
        this.list.splice(i, 1);
        continue;
      }
      const tx = d.target[0] - d.pos[0];
      const ty = d.target[1] - d.pos[1];
      const tz = d.target[2] - d.pos[2];
      const td = Math.hypot(tx, ty, tz);
      if (td < 0.05) {
        // Hanging in the air, then a dart to somewhere new (away from the wind if it is near).
        d.hover -= dt;
        if (d.hover <= 0) {
          d.hover = rand(0.6, 2.5);
          const flee = Math.hypot(d.pos[0] - c.wind[0], d.pos[2] - c.wind[2]) < 5;
          d.target = this.pick(d.pos[0] + rand(-6, 6) + (flee ? Math.sign(d.pos[0] - c.wind[0]) * 8 : 0));
        }
      } else {
        // Darting: fast, straight, easing in.
        const step = Math.min(td, Math.max(1.5, td * 4) * dt);
        d.pos[0] += (tx / td) * step;
        d.pos[1] += (ty / td) * step;
        d.pos[2] += (tz / td) * step;
        d.yaw += angleTo(d.yaw, Math.atan2(tx, tz)) * Math.min(1, dt * 10);
      }
      // A faint hover drift.
      const hb = Math.sin(c.t * 3 + d.seed * 30) * 0.003;
      d.pos[1] += hb;
      d.anim.speed = 22 + d.seed * 6;
      d.anim.update(dt);
      if (c.night < 0.5 && c.rain < 0.5) herd.push(d.pos, d.yaw, 0, 0, DRAGON_SIZE, 0, d.anim, d.seed);
    }
    herd.end();
  }
}

// =======================================================================================
// Mallards on the calm lowland river: pairs paddling slowly, up-ending to feed, preening; they
// paddle off when the wind skims too close.

const DUCK_SIZE = 1.5;
const DUCKS = 6;

interface Duck {
  pos: Vec3;
  yaw: number;
  wantYaw: number;
  speed: number;
  state: "swim" | "idle" | "dabble" | "preen" | "flee";
  timer: number;
  anim: BirdAnimator;
  seed: number;
}

class Mallards {
  readonly list: Duck[] = [];
  private retry = 0;

  constructor(readonly herd: Herd) {}

  private set(d: Duck, state: Duck["state"], timer: number): void {
    d.state = state;
    d.timer = timer;
    d.anim.play(state === "flee" ? "swim" : state, 0.4, state === "flee" ? 2.2 : 1, state === "dabble" || state === "preen");
  }

  update(c: FaunaContext): void {
    const dt = c.dt;
    const [rd, , rhw] = riverInfo(c.wind[0], c.wind[2]);
    this.retry -= dt;
    if (rd < rhw + 120 && this.list.length < DUCKS - 1 && this.retry <= 0) {
      this.retry = 2;
      const x = c.wind[0] + rand(-90, 90);
      if (riverUpper(x) < 0.02 && Math.abs(x - c.wind[0]) > 25) {
        const z = riverCenter(x) + rand(-0.3, 0.3) * riverHalfWidth(x);
        for (let k = 0; k < 2; k++) {
          const yaw = rand(0, 6.28);
          this.list.push({ pos: [x + k * 1.2, riverWater(x), z + k * 0.6], yaw, wantYaw: yaw, speed: 0, state: "swim", timer: rand(3, 8), anim: new BirdAnimator(this.herd.species.clips, "swim", Math.random()), seed: k === 0 ? 0.75 : 0.2 });
        }
      }
    }
    const herd = this.herd;
    herd.begin();
    for (let i = this.list.length - 1; i >= 0; i--) {
      const d = this.list[i];
      const dx = d.pos[0] - c.wind[0];
      const dz = d.pos[2] - c.wind[2];
      const dist = Math.hypot(dx, dz);
      if (dist > 200) {
        this.list.splice(i, 1);
        continue;
      }
      d.timer -= dt;
      if (dist < 9 && c.windAlt < 6 && d.state !== "flee") {
        this.set(d, "flee", rand(3, 5));
        // Along the river, away from the wind.
        d.wantYaw = dx > 0 ? Math.PI / 2 : -Math.PI / 2;
      } else if ((d.state === "dabble" || d.state === "preen") && d.anim.done) this.set(d, "swim", rand(3, 8));
      else if (d.timer <= 0 && d.state !== "dabble" && d.state !== "preen") {
        const r = Math.random();
        if (r < 0.35) {
          this.set(d, "swim", rand(4, 10));
          d.wantYaw = d.yaw + rand(-1.5, 1.5);
        } else if (r < 0.6) this.set(d, "dabble", 10);
        else if (r < 0.8) this.set(d, "preen", 10);
        else this.set(d, "idle", rand(3, 7));
      }
      // Stay on the water, away from the banks.
      const hw = riverHalfWidth(d.pos[0]);
      const off = d.pos[2] - riverCenter(d.pos[0]);
      const ahead = off + Math.cos(d.wantYaw) * 2;
      if (Math.abs(ahead) > hw * 0.7) d.wantYaw = Math.atan2(Math.sin(d.wantYaw), -Math.sign(off) * Math.abs(Math.cos(d.wantYaw)));
      d.yaw += angleTo(d.yaw, d.wantYaw) * Math.min(1, dt * 1.2);
      const want = d.state === "flee" ? 2.2 : d.state === "swim" ? 0.45 : 0;
      d.speed += (want - d.speed) * Math.min(1, dt * 1.5);
      d.pos[0] += Math.sin(d.yaw) * d.speed * dt;
      d.pos[2] += Math.cos(d.yaw) * d.speed * dt;
      d.pos[1] = riverWater(d.pos[0]) + Math.sin(c.t * 1.3 + d.seed * 9) * 0.01;
      d.anim.update(dt);
      herd.push(d.pos, d.yaw, 0, Math.sin(c.t * 0.9 + d.seed * 5) * 0.02, DUCK_SIZE, 0, d.anim, d.seed);
    }
    herd.end();
  }
}

/** All the field's animals. */
export class Fauna {
  private constructor(
    readonly hares: Hares,
    readonly butterflies: Butterflies,
    readonly deer: DeerHerds,
    readonly foxes: Foxes,
    readonly frogs: Frogs,
    readonly dragonflies: Dragonflies,
    readonly ducks: Mallards,
  ) {}

  static async load(gpu: Gpu, globals: SharedUniforms): Promise<Fauna> {
    const [hare, butterfly, deer, fox, frog, dragonfly, mallard] = await Promise.all([
      Herd.load(gpu, globals, "hare", 0, HARES),
      Herd.load(gpu, globals, "butterfly", 1, BUTTERFLIES),
      Herd.load(gpu, globals, "deer", 0, DEER),
      Herd.load(gpu, globals, "fox", 0, FOXES),
      Herd.load(gpu, globals, "frog", 3, FROGS),
      Herd.load(gpu, globals, "dragonfly", 2, DRAGONFLIES, true),
      Herd.load(gpu, globals, "mallard", 4, DUCKS),
    ]);
    return new Fauna(new Hares(hare), new Butterflies(butterfly), new DeerHerds(deer), new Foxes(fox), new Frogs(frog), new Dragonflies(dragonfly), new Mallards(mallard));
  }

  private get all(): { herd: Herd }[] {
    return [this.hares, this.butterflies, this.deer, this.foxes, this.frogs, this.dragonflies, this.ducks];
  }

  get draws(): Draw[] {
    return this.all.map((g) => g.herd.body);
  }

  /** `critters` (globals): the ground animals nearest the wind, for the grass to part around. */
  update(c: FaunaContext, critters: Float32Array): void {
    this.hares.update(c);
    this.butterflies.update(c);
    this.deer.update(c);
    this.foxes.update(c);
    this.frogs.update(c);
    this.dragonflies.update(c);
    this.ducks.update(c);
    const near: [number, number, number, number, number][] = [];
    for (const g of this.grounded()) near.push([Math.hypot(g[0] - c.wind[0], g[2] - c.wind[2]), ...g]);
    near.sort((a, b) => a[0] - b[0]);
    critters.fill(0);
    near.slice(0, critters.length / 4).forEach((n, i) => critters.set([n[1], n[2], n[3], n[0] < 110 ? n[4] : 0], i * 4));
  }

  /** Animals on the ground: x, y, z, how wide the grass parts around them (m). */
  private *grounded(): Generator<[number, number, number, number]> {
    for (const h of this.hares.list) yield [h.pos[0], h.pos[1], h.pos[2], 0.42 * (HARE_SIZE / 2.2)];
    for (const d of this.deer.list) yield [d.pos[0], d.pos[1], d.pos[2], 0.5];
    for (const f of this.foxes.list) yield [f.pos[0], f.pos[1], f.pos[2], 0.45];
    for (const f of this.frogs.list) yield [f.pos[0], f.pos[1], f.pos[2], 0.6];
  }

  encode(pass: FramePass): void {
    for (const g of this.all) g.herd.encode(pass);
  }
}
