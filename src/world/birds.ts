import { draw, geometry, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import birdShader from "../shaders/birds.wgsl";
import type { Vec3 } from "../engine/camera";
import { ecology } from "./ecology";
import { riverInfo, terrainHeightM as terrainHeight } from "./height";
import { pathNear } from "./lantern-path";
import { sunflowerNear } from "./sunflower-field";

const FLOCKS = 6;
const PER_FLOCK = 9;
const MAX = FLOCKS * PER_FLOCK;
/** Room for the hawks circling in thermals (two per thermal). */
const SOARERS = 12;
/** How close the wind must pass to startle a flock (m). */
const STARTLE = 8;

interface Bird {
  pos: Vec3;
  vel: Vec3;
  yaw: number;
  pitch: number;
  flap: number;
  phase: number;
  spread: number;
  /** 0 on the ground, 1 flying, 2 coming in to land. */
  state: number;
  /** Seconds until the next hop / peck / takeoff. */
  timer: number;
  seed: number;
  /** Where it is going to land. */
  goal: Vec3;
  /** Offset within the flock. */
  offset: [number, number];
}

interface Flock {
  birds: Bird[];
  /** Ground spot the flock settles on. */
  home: Vec3;
  airborne: boolean;
  /** Seconds left flying with the wind in a murmuration (0 when not). */
  following?: number;
  /** Seconds aloft. */
  air: number;
  /** Seconds to the next chirp. */
  chirp?: number;
}

function rand(a: number, b: number): number {
  return a + Math.random() * (b - a);
}

/** Open meadow (no forest, river, path or field) where birds like to settle. */
function goodGround(x: number, z: number): boolean {
  const eco = ecology(x, z);
  if (eco.forest > 0.25 || eco.slope > 0.3) return false;
  const [d, , hw] = riverInfo(x, z);
  if (d < hw + 6) return false;
  return !pathNear(x, z, 4) && !sunflowerNear(x, z, 4);
}

/**
 * Flocks of small birds feeding in the meadow. They hop and peck until the wind passes through
 * them, then burst up together, wheel around and settle somewhere further off.
 */
interface Manifest {
  vertexBytes: number;
  indexCount: number;
  shoulder: number;
  variants: { name: string; firstIndex: number; indexCount: number }[];
}

export class Birds {
  /** Perched and flying poses. */
  readonly draws: Draw[];
  private readonly flocks: Flock[] = [];
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array((MAX + SOARERS) * 12);
  /** Instances: perched birds first, then flying ones. */
  private perched = 0;
  private flying = 0;
  /** Called with the flock's position when it takes off. */
  onTakeoff: ((at: Vec3, n: number) => void) | null = null;
  /**
   * Where a startled flock should head: given its spot and the direction it flees (radians,
   * atan2(dx, dz)), a place worth showing the wind, or null to just settle further off.
   */
  guide: ((from: Vec3, away: number) => [number, number] | null) | null = null;
  /** A feeding flock chirping (so it can be found by ear). */
  onChirp: ((at: Vec3) => void) | null = null;

  private constructor(draws: Draw[], buffer: StorageBuffer) {
    this.draws = draws;
    this.buffer = buffer;
  }

  static async load(gpu: Gpu, globals: SharedUniforms, base = "assets/bird"): Promise<Birds> {
    const [manifest, bin] = await Promise.all([
      fetch(`${base}/bird.json`).then((r) => r.json() as Promise<Manifest>),
      fetch(`${base}/bird.bin`).then((r) => r.arrayBuffer()),
    ]);
    const geo = geometry(gpu, {
      label: "bird",
      buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
    });
    const buffer = storage(gpu, (MAX + SOARERS) * 48, "read");
    const draws = manifest.variants.map((v) =>
      draw(gpu, {
        label: `bird-${v.name}`,
        shader: birdShader,
        geometry: geo.slice({ firstIndex: v.firstIndex, indexCount: v.indexCount }),
        cull: "none",
        depth: { compare: "greater" },
        constants: { SHOULDER: manifest.shoulder },
        set: { G: globals, birds: buffer },
      }),
    );
    return new Birds(draws, buffer);
  }

  /**
   * A fresh feeding spot `near..far` meters from (x, z), away from `avoid`; with `ahead` (a
   * heading), within the half circle in front.
   */
  private findSpot(x: number, z: number, near: number, far: number, avoid?: Vec3, ahead?: number): Vec3 | null {
    for (let tries = 0; tries < 12; tries++) {
      const a = ahead === undefined ? Math.random() * Math.PI * 2 : ahead - Math.PI / 2 + (Math.random() - 0.5) * 2.2;
      const r = rand(near, far);
      const px = x + Math.cos(a) * r;
      const pz = z + Math.sin(a) * r;
      if (avoid && Math.hypot(px - avoid[0], pz - avoid[2]) < 25) continue;
      if (goodGround(px, pz)) return [px, terrainHeight(px, pz), pz];
    }
    return null;
  }

  private settle(flock: Flock, home: Vec3): void {
    flock.home = home;
    flock.airborne = false;
    for (const b of flock.birds) {
      const x = home[0] + b.offset[0];
      const z = home[2] + b.offset[1];
      b.pos.splice(0, 3, x, terrainHeight(x, z), z);
      b.vel.splice(0, 3, 0, 0, 0);
      b.state = 0;
      b.spread = 0;
      b.flap = 0;
      b.pitch = 0;
      b.timer = rand(0.2, 2);
    }
  }

  private spawnFlock(home: Vec3): Flock {
    const birds: Bird[] = [];
    for (let i = 0; i < PER_FLOCK; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.random()) * 3.5;
      birds.push({
        pos: [0, 0, 0],
        vel: [0, 0, 0],
        yaw: Math.random() * Math.PI * 2,
        pitch: 0,
        flap: 0,
        phase: Math.random() * 10,
        spread: 0,
        state: 0,
        timer: 0,
        seed: Math.random(),
        goal: [0, 0, 0],
        offset: [Math.cos(a) * r, Math.sin(a) * r],
      });
    }
    const flock: Flock = { birds, home, airborne: false, air: 0 };
    this.settle(flock, home);
    return flock;
  }

  /** Where flocks are feeding (for a far-off glow that draws the wind to them). */
  feedingSpots(): Vec3[] {
    return this.flocks.filter((f) => !f.airborne).map((f) => f.home);
  }

  /**
   * A barrel roll near a flock invites it along: flocks within `radius` rise and fly with the
   * wind for a while, flowing around it in shifting shapes (a small murmuration).
   */
  invite(at: Vec3, radius: number): number {
    let n = 0;
    for (const f of this.flocks) {
      if ((f.following ?? 0) > 0) continue;
      const near = f.birds.some((b) => Math.hypot(b.pos[0] - at[0], b.pos[2] - at[2]) < radius);
      if (!near) continue;
      f.following = 25;
      f.airborne = true;
      f.air = 0;
      for (const b of f.birds) {
        b.state = 1;
        b.timer = 0;
        if (b.pos[1] - terrainHeight(b.pos[0], b.pos[2]) < 0.2) b.vel.splice(0, 3, 0, 3.5, 0);
      }
      n++;
    }
    return n;
  }

  private leaderPrev: Vec3 | null = null;
  private soar: { x: number; z: number; top: number; radius: number }[] = [];

  /** Thermals to circle in: a pair of large birds wheels near each one's top. */
  soarAt(columns: { x: number; z: number; top: number; radius: number }[]): void {
    this.soar = columns;
  }

  update(dt: number, wind: Vec3, windAlt: number, night: number, heading = 0): void {
    // The wind's velocity (for the murmuration to keep pace).
    const lv: Vec3 = this.leaderPrev && dt > 0 ? [(wind[0] - this.leaderPrev[0]) / dt, (wind[1] - this.leaderPrev[1]) / dt, (wind[2] - this.leaderPrev[2]) / dt] : [0, 0, 0];
    this.leaderPrev = [wind[0], wind[1], wind[2]];
    const lspeed = Math.hypot(lv[0], lv[2]);
    const fx = lspeed > 0.1 ? lv[0] / lspeed : Math.sin(heading);
    const fz = lspeed > 0.1 ? lv[2] / lspeed : -Math.cos(heading);
    const tNow = performance.now() / 1000;
    // Keep a few flocks feeding around the wind; ones left far behind move on ahead.
    while (this.flocks.length < FLOCKS) {
      // The first flock lands close ahead, so it is met early; the rest further out.
      const spot = this.findSpot(wind[0], wind[2], this.flocks.length ? 35 : 25, this.flocks.length ? 140 : 55, undefined, heading);
      if (!spot) break;
      this.flocks.push(this.spawnFlock(spot));
    }
    for (const f of this.flocks) {
      if (!f.airborne && Math.hypot(f.home[0] - wind[0], f.home[2] - wind[2]) > 220) {
        const spot = this.findSpot(wind[0], wind[2], 60, 150, undefined, heading);
        if (spot) this.settle(f, spot);
      }
    }

    const perched: number[] = [];
    const flying: number[] = [];
    for (const f of this.flocks) {
      const near = Math.hypot(f.home[0] - wind[0], f.home[2] - wind[2]);
      if (!f.airborne && near < STARTLE + 4 && windAlt < 6) {
        // Startled: the whole flock goes up, away from the wind, a few at a time.
        const close = f.birds.some((b) => Math.hypot(b.pos[0] - wind[0], b.pos[2] - wind[2]) < STARTLE);
        if (close) {
          f.airborne = true;
          f.air = 0;
          const away = Math.atan2(f.home[0] - wind[0], f.home[2] - wind[2]);
          // Birds lead the way: toward somewhere the wind has not been yet, if there is one.
          const lead = this.guide?.(f.home, away);
          const goal =
            (lead && this.findSpot(lead[0], lead[1], 6, 35)) ??
            this.findSpot(f.home[0] + Math.sin(away) * 60, f.home[2] + Math.cos(away) * 60, 0, 40, wind) ??
            f.home;
          for (const b of f.birds) {
            b.state = 1;
            b.timer = rand(0, 0.35);
            b.goal = goal;
            const a = away + rand(-0.7, 0.7);
            b.vel.splice(0, 3, Math.sin(a) * rand(3, 5), rand(3.5, 5.5), Math.cos(a) * rand(3, 5));
          }
          this.onTakeoff?.(f.home, f.birds.length);
        }
      }
      if (f.airborne) f.air += dt;
      if ((f.following ?? 0) > 0) {
        f.following! -= dt;
        // Murmuration: each bird holds a moving place in a loose, breathing cloud behind and
        // around the wind (offsets turn and swell, so the shape keeps flowing).
        f.birds.forEach((b, i) => {
          const th = i * 2.39996 + tNow * 0.7;
          const rho = 2.2 + 1.6 * Math.sin(tNow * 0.45 + i * 0.9);
          const back = 3.5 + (i % 5) * 0.7 + Math.sin(tNow * 0.8 + i) * 1.2;
          const side = Math.cos(th) * rho;
          const up = 0.8 + Math.sin(th * 1.3) * rho * 0.5;
          const target: Vec3 = [wind[0] - fx * back - fz * side, wind[1] + up, wind[2] - fz * back + fx * side];
          for (let j = 0; j < 3; j++) {
            const want = lv[j] + (target[j] - b.pos[j]) * 2.2;
            b.vel[j] += (want - b.vel[j]) * Math.min(1, dt * 3);
            b.pos[j] += b.vel[j] * dt;
          }
          const g = terrainHeight(b.pos[0], b.pos[2]);
          if (b.pos[1] < g + 0.4) b.pos[1] = g + 0.4;
          b.state = 1;
          b.spread = 1;
          b.yaw = Math.atan2(b.vel[0], b.vel[2]);
          b.pitch = -Math.atan2(b.vel[1], Math.hypot(b.vel[0], b.vel[2]) + 1e-3) * 0.6;
          b.phase += dt * 24;
          b.flap = Math.sin(tNow * 1.7 + i) > 0.2 ? Math.sin(b.phase) * 0.85 + 0.1 : 0.12;
          if (night < 0.95) flying.push(b.pos[0], b.pos[1] + 0.06, b.pos[2], b.yaw, b.flap, b.spread, b.pitch, b.seed, 0, 0, 0, 0);
        });
        if (f.following! <= 0) {
          // Time to go: peel off and settle somewhere ahead.
          f.following = 0;
          f.air = 2;
          const goal = this.findSpot(wind[0] + fx * 60, wind[2] + fz * 60, 0, 50) ?? f.home;
          for (const b of f.birds) b.goal = goal;
        }
        continue;
      }
      // Feeding flocks chirp now and then: the way to find them in the long grass.
      f.chirp = (f.chirp ?? Math.random() * 3) - dt;
      if (!f.airborne && f.chirp <= 0) {
        f.chirp = 1.2 + Math.random() * 2.8;
        if (near < 80 && night < 0.5) this.onChirp?.(f.home);
      }
      let landed = 0;
      for (const [bi, b] of f.birds.entries()) {
        b.timer -= dt;
        // Two lookouts circle high over the feeding flock: a sign, from afar, of birds below.
        const lookout = bi < 2 && !f.airborne;
        if (lookout) {
          b.phase += dt * 20;
          const a = (performance.now() / 1000) * 0.45 + bi * Math.PI + b.seed;
          const r = 9 + bi * 3;
          const ty = f.home[1] + 11 + bi * 2 + Math.sin(a * 0.7) * 1.5;
          const want: Vec3 = [f.home[0] + Math.cos(a) * r, ty, f.home[2] + Math.sin(a) * r];
          const tan: Vec3 = [-Math.sin(a) * r * 0.45, 0, Math.cos(a) * r * 0.45];
          for (let j = 0; j < 3; j++) {
            const dv = tan[j] + (want[j] - b.pos[j]) * 0.8 - b.vel[j];
            b.vel[j] += dv * Math.min(1, dt * 2);
            b.pos[j] += b.vel[j] * dt;
          }
          b.state = 1;
          b.yaw = Math.atan2(b.vel[0], b.vel[2]);
          b.pitch = -Math.atan2(b.vel[1], Math.hypot(b.vel[0], b.vel[2])) * 0.7;
          b.spread = 1;
          // Mostly gliding, banking round; a few wingbeats now and then.
          b.flap = Math.sin(a * 2.3 + b.seed * 7) > 0.6 ? Math.sin(b.phase) * 0.8 + 0.1 : 0.12;
          if (night < 0.95) {
            flying.push(b.pos[0], b.pos[1], b.pos[2], b.yaw, b.flap, b.spread, b.pitch, b.seed, 0, 0, 0, 0);
          }
          continue;
        }
        if (b.state === 0) {
          // Feeding: peck, turn, and the odd little hop.
          if (b.timer <= 0) {
            b.timer = rand(0.4, 2.5);
            if (Math.random() < 0.35) {
              // Mostly small hops; now and then a lookout flutters up above the grass.
              const lookout = Math.random() < 0.3;
              b.vel[1] = lookout ? 4.2 : 1.6;
              const a = b.yaw + rand(-1, 1);
              b.vel[0] = Math.sin(a) * (lookout ? 1.2 : 0.6);
              b.vel[2] = Math.cos(a) * (lookout ? 1.2 : 0.6);
            } else b.yaw += rand(-1.2, 1.2);
          }
          b.vel[1] -= 9.8 * dt;
          for (let j = 0; j < 3; j++) b.pos[j] += b.vel[j] * dt;
          const g = terrainHeight(b.pos[0], b.pos[2]);
          if (b.pos[1] <= g) {
            b.pos[1] = g;
            b.vel.splice(0, 3, 0, 0, 0);
          }
          // Peck: a quick dip of the head.
          b.pitch = Math.max(0, Math.sin(b.timer * 9)) * 0.5;
          b.spread = Math.max(0, b.spread - dt * 4);
          b.flap = 0;
        } else if (b.timer <= 0) {
          const g = terrainHeight(b.pos[0], b.pos[2]);
          const tx = b.goal[0] + b.offset[0];
          const tz = b.goal[2] + b.offset[1];
          const dx = tx - b.pos[0];
          const dz = tz - b.pos[2];
          const dist = Math.hypot(dx, dz);
          // Climb out, wheel round toward the goal, then glide down to it.
          const cruise = f.air < 1.2 ? 9 : Math.min(9, 4 + dist * 0.3);
          const wantY = dist > 14 ? g + 6 + b.seed * 3 + Math.sin(f.air * 0.9 + b.seed * 6) : terrainHeight(tx, tz);
          const dirX = dist > 0.01 ? dx / dist : 0;
          const dirZ = dist > 0.01 ? dz / dist : 0;
          const steer = f.air < 0.8 ? 0.6 : 2.2;
          b.vel[0] += (dirX * cruise - b.vel[0]) * steer * dt;
          b.vel[2] += (dirZ * cruise - b.vel[2]) * steer * dt;
          b.vel[1] += ((wantY - b.pos[1]) * 1.2 - b.vel[1]) * 2 * dt;
          for (let j = 0; j < 3; j++) b.pos[j] += b.vel[j] * dt;
          if (b.pos[1] < g) b.pos[1] = g;
          b.yaw = Math.atan2(b.vel[0], b.vel[2]);
          b.pitch = -Math.atan2(b.vel[1], Math.hypot(b.vel[0], b.vel[2])) * 0.7;
          b.spread = Math.min(1, b.spread + dt * 6);
          // Bursts of wingbeats with glides between; hard flapping on the climb.
          const climbing = b.vel[1] > 0.5 || f.air < 1.5;
          b.phase += dt * (climbing ? 26 : 20);
          const glide = !climbing && Math.sin(f.air * 1.7 + b.seed * 9) > 0.3;
          b.flap = glide ? 0.12 : Math.sin(b.phase) * 0.85 + 0.1;
          if (dist < 0.6 && b.pos[1] - g < 0.4 && f.air > 2) {
            b.state = 0;
            b.timer = rand(0.5, 2);
            b.pos[1] = g;
            b.vel.splice(0, 3, 0, 0, 0);
          }
        }
        if (b.state === 0) landed++;
        // Birds roost at night (fade out by shrinking into the ground).
        const show = 1 - night;
        if (show > 0.05) {
          // Perched models stand on their feet; flying ones are centred on the body.
          const hopping = b.state === 0 && b.pos[1] - terrainHeight(b.pos[0], b.pos[2]) > 0.15;
          if (hopping) {
            b.phase += dt * 28;
            flying.push(b.pos[0], b.pos[1] + 0.06, b.pos[2], b.yaw, Math.sin(b.phase) * 0.9 + 0.1, 1, -0.2, b.seed, 0, 0, 0, 0);
          } else if (b.state === 0) perched.push(b.pos[0], b.pos[1], b.pos[2], b.yaw, 0, 0, b.pitch, b.seed, 0, 0, 0, 0);
          else flying.push(b.pos[0], b.pos[1] + 0.06, b.pos[2], b.yaw, b.flap, b.spread, b.pitch, b.seed, 0, 0, 0, 0);
        }
      }
      if (f.airborne && landed >= f.birds.length - 2 && f.air > 2) {
        f.airborne = false;
        f.home = f.birds[0].goal;
      }
      // Give up circling after a while and settle wherever the goal is.
      if (f.airborne && f.air > 80) this.settle(f, f.birds[0].goal);
    }
    // Hawks soaring in the thermals: wide slow circles near the top, wings held still and
    // banked into the turn; big enough to spot from far away.
    if (night < 0.5) {
      this.soar.forEach((c, ci) => {
        for (let k = 0; k < 2; k++) {
          const w = tNow * (0.32 + k * 0.05) + ci * 2 + k * Math.PI;
          const r = c.radius * (1.6 + k * 0.5);
          const y = c.top - 4 - k * 3 + Math.sin(tNow * 0.3 + ci) * 1.5;
          const x = c.x + Math.cos(w) * r;
          const z = c.z + Math.sin(w) * r;
          // Heading along the circle (tangent), banked inward.
          const yaw = Math.atan2(-Math.sin(w), Math.cos(w));
          const flap = Math.sin(tNow * 0.9 + k + ci) > 0.93 ? Math.sin(tNow * 14) * 0.5 : 0.08;
          flying.push(x, y, z, yaw, flap, 1, 0, 0.1, -0.35, 4.2, 0, 0);
        }
      });
    }
    this.perched = perched.length / 12;
    this.flying = flying.length / 12;
    this.data.set(perched, 0);
    this.data.set(flying, perched.length);
    const n = perched.length + flying.length;
    if (n) this.buffer.write(this.data.subarray(0, n));
  }

  encode(pass: FramePass): void {
    if (this.perched) pass.draw(this.draws[0], { instances: this.perched });
    if (this.flying) pass.draw(this.draws[1], { instances: this.flying, firstInstance: this.perched });
  }
}
