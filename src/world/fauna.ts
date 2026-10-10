import { storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import animalShader from "../shaders/animals.wgsl";
import type { Vec3 } from "../engine/camera";
import { BIRD_STRIDE as STRIDE, BirdAnimator, BirdSpecies } from "./bird-model";
import { ecology } from "./ecology";
import { riverInfo, terrainHeightM as terrainHeight } from "./height";
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

/** All the field's animals. */
export class Fauna {
  private constructor(
    readonly hares: Hares,
    readonly butterflies: Butterflies,
  ) {}

  static async load(gpu: Gpu, globals: SharedUniforms): Promise<Fauna> {
    const [hare, butterfly] = await Promise.all([Herd.load(gpu, globals, "hare", 0, HARES), Herd.load(gpu, globals, "butterfly", 1, BUTTERFLIES)]);
    return new Fauna(new Hares(hare), new Butterflies(butterfly));
  }

  get draws(): Draw[] {
    return [this.hares.herd.body, this.butterflies.herd.body];
  }

  /** `critters` (globals): the ground animals nearest the wind, for the grass to part around. */
  update(c: FaunaContext, critters: Float32Array): void {
    this.hares.update(c);
    this.butterflies.update(c);
    const near: [number, number, number, number, number][] = [];
    for (const g of this.grounded()) near.push([Math.hypot(g[0] - c.wind[0], g[2] - c.wind[2]), ...g]);
    near.sort((a, b) => a[0] - b[0]);
    critters.fill(0);
    near.slice(0, critters.length / 4).forEach((n, i) => critters.set([n[1], n[2], n[3], n[0] < 110 ? n[4] : 0], i * 4));
  }

  /** Animals on the ground: x, y, z, how wide the grass parts around them (m). */
  private *grounded(): Generator<[number, number, number, number]> {
    for (const h of this.hares.list) yield [h.pos[0], h.pos[1], h.pos[2], 0.42 * (HARE_SIZE / 2.2)];
  }

  encode(pass: FramePass): void {
    this.hares.herd.encode(pass);
    this.butterflies.herd.encode(pass);
  }
}
