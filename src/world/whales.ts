import { draw, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import whaleShader from "../shaders/whales.wgsl";
import spoutShader from "../shaders/spouts.wgsl";
import { BIRD_STRIDE, BirdAnimator, BirdSpecies } from "./bird-model";
import { SEA_Y, coastInfo, terrainHeightM } from "./height";
import { swellHeight } from "./swell";
import type { Vec3 } from "../engine/camera";

const MAX = 4;
const PUFFS = 512;

type State = "rest" | "swim" | "dive" | "under" | "surface";

interface Whale {
  x: number;
  z: number;
  y: number;
  heading: number;
  pitch: number;
  bank: number;
  size: number;
  seed: number;
  state: State;
  timer: number;
  /** Seconds to the next breath while at the surface. */
  breath: number;
  /** A bird resting on its back: it lies still at the surface until the bird has gone. */
  held: boolean;
  anim: BirdAnimator;
}

interface Puff {
  p: Vec3;
  v: Vec3;
  r: number;
  grow: number;
  age: number;
  life: number;
  seed: number;
}

/** Open sea deep enough for a whale (well beyond the surf). */
function deepWater(x: number, z: number): boolean {
  const [s] = coastInfo(x, z);
  if (s > -170) return false;
  return s < -500 || SEA_Y - terrainHeightM(x, z) > 7;
}

/**
 * Humpback whales offshore, living their own slow lives: logging at the surface and breathing
 * (a tall misty blow every so often), cruising, then the terminal dive (flukes lifted clear
 * of the water), a long time down, and up again somewhere else. A resting whale lets a bird
 * rest on its back; it waits until the bird has flown before diving again.
 */
export class Whales {
  private readonly whales: Whale[] = [];
  private readonly data = new Float32Array(MAX * BIRD_STRIDE);
  private readonly puffs: Puff[] = [];
  private readonly puffData = new Float32Array(PUFFS * 8);
  private count = 0;
  private puffCount = 0;
  private songIn = 30;

  /** Footprints: the smooth slick left where a whale dived (x, z, strength). */
  private readonly prints: { x: number; z: number; k: number }[] = [];

  /** For the sea's surface: each whale's wash and wake, and the footprints. */
  seaInfo(): { whales: number[][]; prints: number[][] } {
    const whales: number[][] = [];
    const prints: number[][] = [];
    for (let i = 0; i < 4; i++) {
      const w = this.whales[i];
      const fp = this.prints[i];
      let wash = 0;
      if (w) {
        const k = w.anim.clip === "surface" ? Math.min(1, w.anim.time / 5) : w.anim.clip === "dive" ? Math.max(0, 1 - w.anim.time / 7) : 1;
        wash = w.state === "under" ? 0 : k;
      }
      whales.push(w ? [w.x, w.z, w.heading, wash] : [0, 0, 0, 0]);
      prints.push([fp ? fp.x : 0, fp ? fp.z : 0, fp ? fp.k : 0, w && w.state === "swim" ? 1 : 0]);
    }
    return { whales, prints };
  }

  /** A whale blew (for the sound): where. */
  onBlow?: (at: Vec3) => void;
  /** Distant song (pan -1..1). */
  onSong?: (pan: number) => void;

  private constructor(
    private readonly species: BirdSpecies,
    private readonly buffer: StorageBuffer,
    readonly draw: Draw,
    private readonly puffBuffer: StorageBuffer,
    readonly spoutDraw: Draw,
  ) {}

  static async load(gpu: Gpu, globals: SharedUniforms): Promise<Whales> {
    const species = await BirdSpecies.load(gpu, "whale", "assets/whales");
    species.shader = whaleShader;
    species.instanceKey = "whales";
    const buffer = storage(gpu, MAX * BIRD_STRIDE * 4, "read");
    const { body } = species.draws(globals, buffer, "whales");
    const puffBuffer = storage(gpu, PUFFS * 32, "read");
    const spoutDraw = draw(gpu, {
      label: "whale-spouts",
      shader: spoutShader,
      vertices: 6,
      blend: "premultiplied",
      depth: { compare: "greater", write: false },
      set: { G: globals, puffs: puffBuffer },
    });
    return new Whales(species, buffer, body, puffBuffer, spoutDraw);
  }

  get draws(): Draw[] {
    return [this.draw, this.spoutDraw];
  }

  private spawn(cam: Vec3): void {
    for (let k = 0; k < 20; k++) {
      const a = Math.random() * Math.PI * 2;
      const r = 220 + Math.random() * 500;
      const x = cam[0] + Math.cos(a) * r;
      const z = cam[2] + Math.sin(a) * r;
      if (!deepWater(x, z)) continue;
      // Each whale starts down, and comes up in its own time.
      this.whales.push({
        x,
        z,
        y: SEA_Y,
        heading: Math.random() * Math.PI * 2,
        pitch: 0,
        bank: 0,
        size: 0.85 + Math.random() * 0.3,
        seed: Math.random(),
        state: "under",
        timer: 2 + Math.random() * 25,
        breath: 2,
        held: false,
        anim: new BirdAnimator(this.species.clips, "rest"),
      });
      return;
    }
  }

  /** A whale lying at the surface near `at` (within `radius`), if any. */
  restingNear(at: Vec3, radius: number): Whale | null {
    let best: Whale | null = null;
    let bd = radius;
    for (const w of this.whales) {
      if (w.state !== "rest" && w.state !== "swim") continue;
      const d = Math.hypot(w.x - at[0], w.z - at[2]);
      if (d < bd) {
        bd = d;
        best = w;
      }
    }
    return best;
  }

  /** Brings a whale up to rest at about (x, z) (a bird over the sea needs somewhere to land). */
  callUp(x: number, z: number, heading: number): Whale | null {
    if (!deepWater(x, z)) return null;
    let w = this.whales.find((v) => v.state === "under") ?? null;
    if (!w && this.whales.length < MAX) {
      this.spawn([x, 0, z]);
      w = this.whales[this.whales.length - 1] ?? null;
    }
    if (!w || w.state !== "under") return null;
    w.x = x;
    w.z = z;
    w.heading = heading;
    w.timer = 0;
    return w;
  }

  /** The spot on a whale's back a bird rests on (just behind the blowhole). */
  backPoint(w: Whale): Vec3 {
    const lx = 0;
    const lz = 1.2 * w.size;
    return [w.x + Math.sin(w.heading) * lz + Math.cos(w.heading) * lx, w.y + 0.52 * w.size, w.z + Math.cos(w.heading) * lz - Math.sin(w.heading) * lx];
  }

  hold(w: Whale | null): void {
    for (const v of this.whales) v.held = v === w;
    if (w && w.state === "swim") {
      w.state = "rest";
      w.anim.play("rest", 1.5, 1);
    }
  }

  update(dt: number, t: number, cam: Vec3, wind: [number, number]): void {
    const [sc] = coastInfo(cam[0], cam[2]);
    const atSea = sc < 400;
    if (atSea && this.whales.length < (sc < -100 ? MAX : 2)) this.spawn(cam);
    for (let i = this.whales.length - 1; i >= 0; i--) {
      const w = this.whales[i];
      if (!w.held && (Math.hypot(w.x - cam[0], w.z - cam[2]) > 1600 || !atSea)) {
        this.whales.splice(i, 1);
        continue;
      }
      this.think(w, dt, cam);
      w.anim.update(dt);
      // Floating on the swell: heave with it, pitch and roll with its slope.
      const sw = swellHeight(w.x, w.z, t);
      const fx = Math.sin(w.heading);
      const fz = Math.cos(w.heading);
      const ahead = swellHeight(w.x + fx * 5, w.z + fz * 5, t) - swellHeight(w.x - fx * 5, w.z - fz * 5, t);
      const side = swellHeight(w.x + fz * 3, w.z - fx * 3, t) - swellHeight(w.x - fz * 3, w.z + fx * 3, t);
      // (Riding a little low: only the back and the dorsal hump break the surface.)
      w.y += (SEA_Y - 0.25 + sw * 0.85 - w.y) * Math.min(1, dt * 1.5);
      w.pitch += (-Math.atan2(ahead, 10) * 0.6 - w.pitch) * Math.min(1, dt);
      w.bank += (Math.atan2(side, 6) * 0.5 - w.bank) * Math.min(1, dt);
    }
    // Song now and then, when out at sea.
    this.songIn -= dt;
    if (this.songIn <= 0) {
      this.songIn = 50 + Math.random() * 90;
      if (sc < -60 && this.whales.length) this.onSong?.(Math.random() * 2 - 1);
    }
    this.simulatePuffs(dt, wind);
    // Footprints form as the flukes go down, then fade over a minute.
    for (const f of this.prints) f.k = f.k >= 0 ? Math.min(1, f.k + dt / 8) : f.k;
    for (let i = this.prints.length - 1; i >= 0; i--) {
      const f = this.prints[i] as { x: number; z: number; k: number; age?: number };
      f.age = (f.age ?? 0) + dt;
      if (f.age > 8) f.k = Math.max(0, 1 - (f.age - 8) / 60);
      if (f.age > 70) this.prints.splice(i, 1);
    }
    this.count = 0;
    for (const w of this.whales) {
      if (w.state === "under") continue;
      const o = this.count++ * BIRD_STRIDE;
      this.data.set([w.x, w.y, w.z, w.heading, w.pitch, w.bank, w.size, 0], o);
      w.anim.write(this.data, o + 8);
      this.data[o + 11] = w.seed;
    }
    if (this.count) this.buffer.write(this.data.subarray(0, this.count * BIRD_STRIDE));
  }

  private think(w: Whale, dt: number, cam: Vec3): void {
    w.timer -= dt;
    switch (w.state) {
      case "under":
        if (w.timer <= 0) {
          w.state = "surface";
          w.anim.play("surface", 0, 1, true);
          w.y = SEA_Y;
        }
        return;
      case "surface":
        if (w.anim.done) {
          w.state = "rest";
          w.timer = 25 + Math.random() * 35;
          w.anim.play("rest", 0.8, 1);
          this.blow(w, 1.15);
          w.breath = 8 + Math.random() * 8;
        }
        return;
      case "dive":
        if (w.anim.done) {
          // Down for a long while; it comes up elsewhere.
          w.state = "under";
          w.timer = 25 + Math.random() * 45;
          for (let k = 0; k < 10; k++) {
            const a = w.heading + (Math.random() - 0.5) * 1.6;
            const d = 80 + Math.random() * 160;
            const nx = w.x + Math.sin(a) * d;
            const nz = w.z + Math.cos(a) * d;
            if (deepWater(nx, nz)) {
              w.x = nx;
              w.z = nz;
              w.heading = a + (Math.random() - 0.5);
              break;
            }
          }
        }
        return;
    }
    // At the surface: breathe, and now and then cruise a little; then dive (unless a bird rests on it).
    w.breath -= dt;
    if (w.breath <= 0) {
      this.blow(w, 1);
      w.breath = 10 + Math.random() * 12;
    }
    if (w.state === "swim") {
      const v = 1.4 * w.size;
      const nx = w.x + Math.sin(w.heading) * v * dt;
      const nz = w.z + Math.cos(w.heading) * v * dt;
      if (deepWater(nx, nz)) {
        w.x = nx;
        w.z = nz;
      } else w.heading += dt * 0.4;
      w.heading += Math.sin(w.seed * 30 + w.timer * 0.1) * 0.02 * dt;
    }
    // Now and then, lying at rest, it rolls and waves a long white flipper, slapping it down.
    if (w.state === "rest" && !w.held) {
      if (w.anim.clip === "rest" && Math.random() < dt * 0.025) w.anim.play("finwave", 1.2, 1, true);
      else if (w.anim.clip === "finwave" && w.anim.done) w.anim.play("rest", 1.2, 1);
    }
    if (w.held) {
      if (w.anim.clip === "finwave") w.anim.play("rest", 1.5, 1);
      return;
    }
    if (w.timer <= 0 && w.anim.clip !== "finwave") {
      if (w.state === "rest" && Math.random() < 0.5) {
        w.state = "swim";
        w.timer = 15 + Math.random() * 25;
        w.anim.play("swim", 2, 1);
      } else {
        // Fluke up: a last breath, then down (leaving its footprint on the water).
        w.state = "dive";
        if (this.prints.length >= 4) this.prints.shift();
        this.prints.push({ x: w.x - Math.sin(w.heading) * 4, z: w.z - Math.cos(w.heading) * 4, k: 0 });
        w.anim.play("dive", 1.2, 1, true);
        if (w.breath > 3) this.blow(w, 0.9);
      }
    }
  }

  /** A breath: the blow rising from the blowhole. */
  private blow(w: Whale, strength: number): void {
    const hole: Vec3 = [w.x + Math.sin(w.heading) * 3.6 * w.size, w.y + 0.1 * w.size, w.z + Math.cos(w.heading) * 3.6 * w.size];
    for (let i = 0; i < 46 && this.puffs.length < PUFFS; i++) {
      const up = (6 + Math.random() * 4) * strength;
      // Humpbacks blow a bushy column, spreading as it rises.
      const spread = 0.5 + Math.random() * 0.9;
      const a = Math.random() * Math.PI * 2;
      this.puffs.push({
        p: [hole[0], hole[1], hole[2]],
        v: [Math.cos(a) * spread, up, Math.sin(a) * spread],
        r: 0.25 + Math.random() * 0.25,
        grow: 0.7 + Math.random() * 0.6,
        age: -i * 0.012,
        life: 2.6 + Math.random() * 1.6,
        seed: Math.random(),
      });
    }
    this.onBlow?.(hole);
  }

  private simulatePuffs(dt: number, wind: [number, number]): void {
    let n = 0;
    for (let i = this.puffs.length - 1; i >= 0; i--) {
      const p = this.puffs[i];
      p.age += dt;
      if (p.age > p.life) {
        this.puffs.splice(i, 1);
        continue;
      }
      if (p.age < 0) continue;
      // Drag slows the jet; the breeze carries the mist off; the droplets settle.
      const drag = Math.exp(-dt * 1.6);
      p.v[0] = p.v[0] * drag + wind[0] * 2.2 * dt;
      p.v[2] = p.v[2] * drag + wind[1] * 2.2 * dt;
      p.v[1] = p.v[1] * drag - 1.2 * dt;
      for (let k = 0; k < 3; k++) p.p[k] += p.v[k] * dt;
      p.r += p.grow * dt;
      if (n < PUFFS) {
        const k = p.age / p.life;
        const o = n * 8;
        this.puffData.set([p.p[0], p.p[1], p.p[2], p.r, 0.32 * Math.min(1, p.age * 6) * (1 - k) * (1 - k), p.seed, k, 0], o);
        n++;
      }
    }
    this.puffCount = n;
    if (n) this.puffBuffer.write(this.puffData.subarray(0, n * 8));
  }

  /** Spray reaching (x, y, z) from a blow right now (0..1): a bird flying through gets wet. */
  sprayAt(at: Vec3): number {
    let s = 0;
    for (const p of this.puffs) {
      if (p.age < 0) continue;
      const d = Math.hypot(p.p[0] - at[0], p.p[1] - at[1], p.p[2] - at[2]);
      if (d < p.r * 1.3) s = Math.max(s, 1 - p.age / p.life);
    }
    return s;
  }

  encode(pass: FramePass): void {
    if (this.count) pass.draw(this.draw, { instances: this.count });
  }

  encodeSpouts(pass: FramePass): void {
    if (this.puffCount) pass.draw(this.spoutDraw, { instances: this.puffCount });
  }
}
