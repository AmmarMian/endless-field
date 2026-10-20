import { storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import { BIRD_STRIDE, BirdAnimator, BirdSpecies } from "./bird-model";
import { SEA_Y, coastInfo, terrainHeightM } from "./height";
import { swellHeight } from "./swell";
import type { Vec3 } from "../engine/camera";

const MAX = 40;
const SOARERS = 7;
const GROUND = 14;

type State = "soar" | "stand" | "walk" | "peck" | "call" | "float" | "takeoff" | "fly" | "approach" | "land";

interface Gull {
  pos: Vec3;
  heading: number;
  pitch: number;
  bank: number;
  size: number;
  seed: number;
  state: State;
  timer: number;
  /** Soaring: the centre it circles, its radius, height and turning direction. */
  centre: [number, number];
  radius: number;
  height: number;
  turn: number;
  angle: number;
  /** Landing: where it will touch down. */
  target: Vec3 | null;
  callIn: number;
  anim: BirdAnimator;
}

function wrapAngle(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

/** A point on the damp sand near the waterline, near (x, z) along the shore. */
function strand(x: number, z: number): Vec3 | null {
  for (let k = 0; k < 30; k++) {
    const px = x + (Math.random() - 0.5) * 30;
    const pz = z + (Math.random() - 0.5) * 30;
    const [s] = coastInfo(px, pz);
    if (s < 0 || s > 30) continue;
    const h = terrainHeightM(px, pz);
    if (h - SEA_Y > 0.08 && h - SEA_Y < 1.2) return [px, h, pz];
  }
  return null;
}

/** The nearest point of the waterline from (x, z) (walking down the coast's gradient). */
function waterline(x: number, z: number): [number, number] {
  let px = x;
  let pz = z;
  for (let i = 0; i < 6; i++) {
    const [s] = coastInfo(px, pz);
    const gx = coastInfo(px + 1, pz)[0] - s;
    const gz = coastInfo(px, pz + 1)[0] - s;
    const g = Math.hypot(gx, gz) || 1;
    px -= (gx / g) * s;
    pz -= (gz / g) * s;
  }
  return [px, pz];
}

/**
 * Herring gulls on the coast. Some hang on the sea breeze above the beach, circling and calling;
 * some stand about on the damp sand at the tide line, walking, pecking at the wrack and
 * throwing their heads back to call. Rush at them and they lift off with alarm calls, wheel
 * about overhead, and later glide back down. They never follow anyone.
 */
export class Gulls {
  private readonly gulls: Gull[] = [];
  private readonly data = new Float32Array(MAX * BIRD_STRIDE);
  private count = 0;
  private refill = 0;
  /** Calls are shared out sparsely: the whole colony waits this long between calls. */
  private quiet = 12;
  private alarmQuiet = 0;

  private call(kind: "long" | "mew" | "alarm", at: Vec3): void {
    if (kind === "alarm") {
      // One alarm for a flushed group, not one per bird.
      if (this.alarmQuiet > 0) return;
      this.alarmQuiet = 8;
    } else {
      if (this.quiet > 0) return;
      this.quiet = 20 + Math.random() * 25;
    }
    this.onCall?.(kind, at);
  }

  /** A gull called: kind, position. */
  onCall?: (kind: "long" | "mew" | "alarm", at: Vec3) => void;

  private constructor(
    private readonly species: BirdSpecies,
    private readonly buffer: StorageBuffer,
    readonly draw: Draw,
  ) {}

  static async load(gpu: Gpu, globals: SharedUniforms): Promise<Gulls> {
    const species = await BirdSpecies.load(gpu, "gull");
    const buffer = storage(gpu, MAX * BIRD_STRIDE * 4, "read");
    const { body } = species.draws(globals, buffer, "gulls");
    return new Gulls(species, buffer, body);
  }

  private make(state: State, pos: Vec3): Gull {
    const g: Gull = {
      pos,
      heading: Math.random() * Math.PI * 2,
      pitch: 0,
      bank: 0,
      size: 0.9 + Math.random() * 0.2,
      seed: Math.random(),
      state,
      timer: Math.random() * 4,
      centre: [pos[0], pos[2]],
      radius: 18 + Math.random() * 30,
      // Mostly low over the waves and the beach (where the wind flies), some higher.
      height: 4 + Math.random() * Math.random() * 12,
      turn: Math.random() < 0.5 ? 1 : -1,
      angle: Math.random() * Math.PI * 2,
      target: null,
      callIn: 20 + Math.random() * 60,
      anim: new BirdAnimator(this.species.clips, state === "soar" ? "glide" : "perch", Math.random() * 3),
    };
    this.gulls.push(g);
    return g;
  }

  private time = 0;
  private flyby = 15;

  update(dt: number, cam: Vec3, threat: Vec3, threatSpeed: number, night: number, t: number): void {
    this.quiet -= dt;
    this.alarmQuiet -= dt;
    this.time = t;
    const [sc] = coastInfo(cam[0], cam[2]);
    const nearCoast = sc < 450;
    const atSea = sc < -20;
    // Now and then a gull wheels past close by (on its own business: it circles on).
    this.flyby -= dt;
    if (nearCoast && this.flyby <= 0 && night < 0.6) {
      this.flyby = 20 + Math.random() * 30;
      const g = this.gulls.find((v) => v.state === "soar" && Math.hypot(v.pos[0] - cam[0], v.pos[2] - cam[2]) > 60);
      if (g) {
        const a = Math.random() * Math.PI * 2;
        g.centre = [threat[0] + Math.cos(a) * 25, threat[2] + Math.sin(a) * 25];
        g.radius = 14 + Math.random() * 12;
        g.height = 3 + Math.random() * 5;
      }
    }
    // Keep the coast populated around the camera (none far inland; fewer about at night).
    this.refill -= dt;
    if (nearCoast && this.refill <= 0) {
      this.refill = 0.5;
      const [wx, wz] = waterline(cam[0], cam[2]);
      const soaring = this.gulls.filter((g) => g.state === "soar" || g.state === "fly").length;
      const standing = this.gulls.length - soaring;
      if (soaring < SOARERS * (1 - night * 0.8)) {
        // Around the camera (over the sea as much as the beach), out of the way.
        const a = Math.random() * Math.PI * 2;
        const r = 90 + Math.random() * 140;
        const x = (atSea ? cam[0] : wx) + Math.cos(a) * r;
        const z = (atSea ? cam[2] : wz) + Math.sin(a) * r * (atSea ? 1 : 0.4);
        const g = this.make("soar", [x, SEA_Y + 20, z]);
        g.centre = [x, z];
        g.pos[1] = Math.max(terrainHeightM(x, z), SEA_Y) + g.height;
      } else if (atSea && standing < GROUND) {
        // A raft of gulls sitting on the swell, somewhere ahead.
        const a = Math.random() * Math.PI * 2;
        const r = 90 + Math.random() * 90;
        const cx = cam[0] + Math.cos(a) * r;
        const cz = cam[2] + Math.sin(a) * r;
        const n = 4 + Math.floor(Math.random() * 6);
        for (let i = 0; i < n && standing + i < GROUND; i++) {
          const x = cx + (Math.random() - 0.5) * 12;
          const z = cz + (Math.random() - 0.5) * 12;
          if (coastInfo(x, z)[0] > -15) continue;
          const g = this.make("float", [x, SEA_Y, z]);
          g.anim.play("perch", 0, 1, true, Math.random() * 5);
        }
      } else if (standing < GROUND) {
        // A small group lands out of sight along the beach, ahead or behind.
        const side = Math.random() < 0.5 ? -1 : 1;
        const [ax, az] = waterline(wx + side * (70 + Math.random() * 60) * Math.cos(0.2), wz);
        const n = 3 + Math.floor(Math.random() * 4);
        for (let i = 0; i < n && standing + i < GROUND; i++) {
          const p = strand(ax, az);
          if (p && Math.hypot(p[0] - cam[0], p[2] - cam[2]) > 60) this.make("stand", p);
        }
      }
    }
    for (let i = this.gulls.length - 1; i >= 0; i--) {
      const g = this.gulls[i];
      const d = Math.hypot(g.pos[0] - cam[0], g.pos[2] - cam[2]);
      if (d > 380 || !nearCoast) {
        this.gulls.splice(i, 1);
        continue;
      }
      this.think(g, dt, threat, threatSpeed, night);
      g.anim.update(dt);
    }
    this.count = 0;
    const perch = this.species.perchHeight;
    for (const g of this.gulls) {
      if (this.count >= MAX) break;
      const o = this.count++ * BIRD_STRIDE;
      const grounded = g.state === "stand" || g.state === "walk" || g.state === "peck" || g.state === "call";
      this.data[o] = g.pos[0];
      // (Floating, the body sits low in the water.)
      this.data[o + 1] = g.pos[1] + (grounded ? perch * g.size : g.state === "float" ? perch * g.size * 0.25 : 0);
      this.data[o + 2] = g.pos[2];
      this.data[o + 3] = g.heading;
      this.data[o + 4] = g.pitch;
      this.data[o + 5] = g.bank;
      this.data[o + 6] = g.size;
      this.data[o + 7] = 0;
      g.anim.write(this.data, o + 8);
      this.data[o + 11] = g.seed;
    }
    if (this.count) this.buffer.write(this.data.subarray(0, this.count * BIRD_STRIDE));
  }

  private think(g: Gull, dt: number, threat: Vec3, threatSpeed: number, night: number): void {
    g.timer -= dt;
    g.callIn -= dt;
    const dx = g.pos[0] - threat[0];
    const dz = g.pos[2] - threat[2];
    const near = Math.hypot(dx, dz);
    if (g.state === "float") {
      // Riding the swell, drifting, turning slowly into the wind.
      g.pos[1] = SEA_Y + swellHeight(g.pos[0], g.pos[2], this.time);
      g.heading += Math.sin(g.seed * 40 + this.time * 0.2) * 0.1 * dt;
      g.pos[0] += Math.sin(g.heading) * 0.15 * dt;
      g.pos[2] += Math.cos(g.heading) * 0.15 * dt;
      if (near < 8 || (near < 22 && threatSpeed > 4)) {
        g.state = "takeoff";
        g.heading = Math.atan2(dx, dz) + (Math.random() - 0.5) * 0.8;
        g.anim.play("takeoff", 0.08, 1, true);
        this.call("alarm", g.pos);
      }
      return;
    }
    const grounded = g.state === "stand" || g.state === "walk" || g.state === "peck" || g.state === "call";
    if (grounded) {
      // Flushed: something rushing in (or simply too close).
      if (near < 6 || (near < 16 && threatSpeed > 4)) {
        g.state = "takeoff";
        g.heading = Math.atan2(dx, dz) + (Math.random() - 0.5) * 0.8;
        g.anim.play("takeoff", 0.08, 1, true);
        this.call("alarm", g.pos);
        return;
      }
      if (g.timer <= 0) {
        const r = Math.random();
        if (r < 0.35) {
          g.state = "walk";
          g.timer = 1.5 + Math.random() * 3;
          g.heading += (Math.random() - 0.5) * 2;
          g.anim.play("walk", 0.25, 1);
        } else if (r < 0.65) {
          g.state = "peck";
          g.timer = this.species.clips.peck ? g.anim.duration("peck") : 1;
          g.anim.play("peck", 0.15, 1, true);
        } else if (r < 0.69 && night < 0.5 && this.quiet <= 0) {
          g.state = "call";
          g.timer = g.anim.duration("call");
          g.anim.play("call", 0.15, 1, true);
          this.call("long", g.pos);
        } else {
          g.state = "stand";
          g.timer = 2 + Math.random() * 6;
          g.anim.play(Math.random() < 0.2 ? "preen" : "perch", 0.3, 1, true);
        }
      }
      if (g.state === "walk") {
        const v = 0.45 * g.size;
        const nx = g.pos[0] + Math.sin(g.heading) * v * dt;
        const nz = g.pos[2] + Math.cos(g.heading) * v * dt;
        const [s] = coastInfo(nx, nz);
        if (s < 0.5 || s > 45) g.heading += Math.PI * 0.6 * dt * 3;
        else {
          g.pos[0] = nx;
          g.pos[2] = nz;
          g.pos[1] = terrainHeightM(nx, nz);
        }
        g.anim.speed = v / (0.2 * g.size);
      }
      return;
    }
    if (g.state === "takeoff") {
      // Spring up and away, beating hard.
      const t = g.anim.time;
      g.pos[0] += Math.sin(g.heading) * 3.5 * Math.min(1, t * 3) * dt;
      g.pos[2] += Math.cos(g.heading) * 3.5 * Math.min(1, t * 3) * dt;
      g.pos[1] += 2.2 * Math.min(1, t * 4) * dt;
      g.pitch = -0.25;
      if (g.anim.done) {
        g.state = "fly";
        g.timer = 15 + Math.random() * 25;
        g.anim.play("flap", 0.1, 3.2);
        g.centre = [g.pos[0] + Math.sin(g.heading) * 25, g.pos[2] + Math.cos(g.heading) * 25];
        g.height = 5 + Math.random() * 7;
      }
      return;
    }
    if (g.state === "land") {
      // Gliding down onto the chosen spot, timed with the clip's touchdown (0.9 s).
      const tgt = g.target!;
      const k = Math.min(1, g.anim.time / 0.9);
      const rem = Math.max(0.05, 0.9 - g.anim.time);
      for (let a = 0; a < 3; a++) g.pos[a] += ((tgt[a] - g.pos[a]) * Math.min(1, dt / rem)) * (k < 1 ? 1 : 0);
      g.pitch *= 0.9;
      g.bank *= 0.9;
      if (g.anim.done) {
        g.pos = [tgt[0], tgt[1], tgt[2]];
        g.state = "stand";
        g.timer = 1 + Math.random() * 3;
        g.anim.play("perch", 0.3, 1);
      }
      return;
    }
    // In the air: circle the centre on the breeze, banking into the turn; the gliders drift
    // slowly along the shore, flapping now and then.
    const speed = 7.5 * g.size;
    const wantR = g.radius;
    const toC = Math.hypot(g.pos[0] - g.centre[0], g.pos[2] - g.centre[1]);
    // Heading of the tangent to the circle, corrected toward the circle's rim.
    const radial = Math.atan2(g.pos[0] - g.centre[0], g.pos[2] - g.centre[1]);
    let tangent = radial + (g.turn * Math.PI) / 2 + g.turn * Math.max(-0.6, Math.min(0.6, (toC - wantR) / wantR));
    // Approaching a landing: straight for the spot, sinking along a shallow glide path.
    const tgt = g.target;
    const toT = tgt ? Math.hypot(tgt[0] - g.pos[0], tgt[2] - g.pos[2]) : 0;
    if (g.state === "approach" && tgt) tangent = Math.atan2(tgt[0] - g.pos[0], tgt[2] - g.pos[2]);
    const dh = wrapAngle(tangent - g.heading);
    const rate = Math.max(-0.9, Math.min(0.9, dh * 1.8));
    g.heading += rate * dt;
    g.bank += (-rate * 0.9 - g.bank) * Math.min(1, dt * 2);
    g.pos[0] += Math.sin(g.heading) * speed * dt;
    g.pos[2] += Math.cos(g.heading) * speed * dt;
    const ground = Math.max(terrainHeightM(g.pos[0], g.pos[2]), SEA_Y);
    const wantY = g.state === "approach" && tgt ? tgt[1] + 0.8 + toT * 0.3 : ground + g.height + Math.sin(g.angle + g.seed * 9) * 2;
    if (g.state === "approach" && tgt && toT < 9 && Math.abs(dh) < 0.5) {
      g.state = "land";
      g.anim.play("land", 0.3, 1, true);
      return;
    }
    g.angle += dt * 0.3;
    g.pos[1] += (wantY - g.pos[1]) * Math.min(1, dt * (g.state === "approach" ? 1.5 : 0.6));
    g.pitch = Math.max(-0.3, Math.min(0.3, (g.pos[1] - wantY) * 0.05));
    g.centre[0] += Math.sin(g.seed * 20) * 0.6 * dt;
    if (g.state === "soar") {
      if (g.anim.clip === "glide" && Math.random() < dt * 0.06) g.anim.play("flap", 0.2, 3.0);
      else if (g.anim.clip === "flap" && Math.random() < dt * 0.6) g.anim.play("glide", 0.4, 1);
    }
    if (g.callIn <= 0) {
      g.callIn = 40 + Math.random() * 80;
      if (night < 0.6 && Math.hypot(g.pos[0] - threat[0], g.pos[2] - threat[2]) < 200) this.call(Math.random() < 0.7 ? "mew" : "long", g.pos);
    }
    if (g.state === "fly") {
      if (g.anim.clip === "flap" && g.timer < 10) g.anim.play("glide", 0.5, 1);
      if (g.timer <= 0) {
        // Back down to the strand, away from whatever flushed it.
        const p = strand(g.centre[0], g.centre[1]);
        if (p && Math.hypot(p[0] - threat[0], p[2] - threat[2]) > 25) {
          g.target = p;
          g.state = "approach";
          g.anim.play("glide", 0.4, 1);
        } else g.timer = 3;
      }
    }
  }

  encode(pass: FramePass): void {
    if (this.count) pass.draw(this.draw, { instances: this.count });
  }
}
