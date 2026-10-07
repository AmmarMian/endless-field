import type { Camera, Vec3 } from "../engine/camera";
import { surfaceHeight as terrainHeight } from "../world/height";
import type { Input } from "./input";

const CRUISE = 7.5;
const GUST = 21;
const CRUISE_ALT = 1.7;
/** The wind skims the field; it never climbs above the treetops of a meadow. */
const MAX_ALT = 10;
/** How much higher a thermal can carry the wind (m). */
const THERMAL_EXTRA = 18;
const TRAIL_LEN = 24;
/** Meters between trail samples. */
const TRAIL_STEP = 0.9;
/** Seconds for a sample's push to fade (the grass springs back). */
const TRAIL_LIFE = 2.2;

/** Ground height averaged over ~2 m (camera clearance without tracing every bump). */
function smoothGround(x: number, z: number): number {
  return (
    (terrainHeight(x, z) * 2 + terrainHeight(x + 2, z) + terrainHeight(x - 2, z) + terrainHeight(x, z + 2) + terrainHeight(x, z - 2)) / 6
  );
}

function damp(current: number, target: number, rate: number, dt: number): number {
  return target + (current - target) * Math.exp(-rate * dt);
}

/** The wind: an invisible leader that the petal stream follows and the camera chases. */
export class Player {
  readonly pos: Vec3 = [0, 0, 0];
  yaw = 0;
  pitch = 0;
  speed = CRUISE;
  /** In a vertical loop (pitch runs once round the full circle). */
  looping = false;
  /** Starts a vertical loop (ignored if already looping or too close to the ground ahead). */
  startLoop(): void {
    if (this.looping) return;
    this.looping = true;
    this.loopYaw = this.yaw;
    this.pitch = Math.max(0, this.pitch);
  }
  private loopYaw = 0;
  /** Barrel roll: the extra roll angle right now (0 when not rolling), and its progress. */
  rollAngle = 0;
  private rollT = -1;
  private rollDir = 0;
  /** Starts a barrel roll to the left (-1) or right (+1). */
  startRoll(dir: number): void {
    if (this.rollT >= 0 || this.looping) return;
    this.rollT = 0;
    this.rollDir = dir;
  }
  /** Perched (resting on a stem, a lantern, the nest), and where it is gliding down to. */
  perched = false;
  landing: Vec3 | null = null;
  private orbit = 0;
  /** Glide down and settle on `at` (a perch). */
  landAt(at: Vec3): void {
    if (this.perched || this.looping || this.riding) return;
    this.landing = [at[0], at[1], at[2]];
  }

  /** Riding a thermal: carried up in a spiral to its top. */
  riding: { x: number; z: number; top: number; radius: number; angle: number; dir: number } | null = null;

  /** Enters a thermal: the wind is carried up around it to `top`, then glides out. */
  ride(c: { x: number; z: number; ground: number; radius: number }, top: number): void {
    if (this.riding || this.looping || this.rideCooldown > 0) return;
    const angle = Math.atan2(this.pos[2] - c.z, this.pos[0] - c.x);
    // Circle the way the wind was already turning (or counter-clockwise).
    const fx = Math.sin(this.yaw);
    const fz = -Math.cos(this.yaw);
    const dir = (this.pos[0] - c.x) * fz - (this.pos[2] - c.z) * fx > 0 ? 1 : -1;
    this.riding = { x: c.x, z: c.z, top, radius: Math.max(4.5, c.radius * 0.6), angle, dir };
  }

  private rideCooldown = 0;

  /** Rising air under the wind right now (0..1, set each frame from the thermals). */
  thermal = 0;
  /** Extra ceiling won in a thermal (m), given back slowly once out of it. */
  private ceilingExtra = 0;
  /** Stooping: diving steeply with the wings tucked (0..1), building speed. */
  stoop = 0;
  /** Speed banked from a dive, spent slowly after pulling out. */
  private momentum = 0;
  /** A little burst of speed (a caught insect). */
  boost(amount: number): void {
    this.momentum = Math.min(12, this.momentum + amount);
  }
  /** Chase distance scale (closer when you fly as a bird, so it fills the view). */
  followScale = 1;
  gust = 0;
  /** Recorded leader positions, newest first, for the grass push. */
  private readonly trail: { p: Vec3; age: number }[] = [];
  private readonly camPos: Vec3 = [0, 0, 0];
  private camInit = false;
  private ceilingGround = 0;
  private ceilingInit = false;

  constructor(x: number, z: number, yaw: number) {
    this.pos[0] = x;
    this.pos[2] = z;
    this.pos[1] = terrainHeight(x, z) + CRUISE_ALT;
    this.yaw = yaw;
  }

  /** Continue from a new pose (e.g. where free roam left off), without camera snapping. */
  teleport(pos: Vec3, yaw: number): void {
    this.pos.splice(0, 3, pos[0], Math.max(pos[1], terrainHeight(pos[0], pos[2]) + 1.2), pos[2]);
    this.yaw = yaw;
    this.pitch = 0;
    this.trail.length = 0;
    this.ceilingInit = false;
    // Jumps (map teleports) cut the camera instead of sweeping across the world.
    this.camInit = false;
  }

  get forward(): Vec3 {
    const cp = Math.cos(this.pitch);
    return [Math.sin(this.yaw) * cp, Math.sin(this.pitch), -Math.cos(this.yaw) * cp];
  }

  get altitude(): number {
    return this.pos[1] - terrainHeight(this.pos[0], this.pos[2]);
  }

  /** `autopilot` steers gently by itself (attract mode before the first click). */
  update(dt: number, input: Input, autopilot: { toward?: [number, number] } | null): void {
    let steerX = input.steerX;
    let steerY = input.steerY;
    let gusting = input.gust;
    if (autopilot) {
      steerY = 0;
      gusting = false;
      steerX = 0;
      if (autopilot.toward) {
        const want = Math.atan2(autopilot.toward[0] - this.pos[0], -(autopilot.toward[1] - this.pos[2]));
        let d = want - this.yaw;
        d = Math.atan2(Math.sin(d), Math.cos(d));
        steerX = Math.max(-0.35, Math.min(0.35, d));
      }
    }
    // Dead zone in the middle of the screen so straight flight is easy.
    const dz = (v: number) => (Math.abs(v) < 0.08 ? 0 : (v - Math.sign(v) * 0.08) / 0.92);
    const sx = dz(steerX);
    const sy = dz(steerY);

    this.rideCooldown = Math.max(0, this.rideCooldown - dt);
    // Resting: still, until anything is touched; then spring back up into the air.
    if (this.perched) {
      if (input.poked) {
        this.perched = false;
        this.pitch = 0.55;
        this.speed = 6;
      } else {
        this.speed = 0;
        this.gust = 0;
        this.recordTrail();
        return;
      }
    }
    // Gliding in to a perch: aim for it, slow down, settle.
    if (this.landing) {
      if (input.poked) this.landing = null;
      else {
        const t = this.landing;
        const dx = t[0] - this.pos[0];
        const dy = t[1] - this.pos[1];
        const dz = t[2] - this.pos[2];
        const flat = Math.hypot(dx, dz);
        const dist = Math.hypot(flat, dy);
        let dyaw = Math.atan2(dx, -dz) - this.yaw;
        dyaw = Math.atan2(Math.sin(dyaw), Math.cos(dyaw));
        this.yaw += dyaw * Math.min(1, dt * 3);
        this.pitch = damp(this.pitch, Math.atan2(dy, Math.max(flat, 0.5)), 3, dt);
        this.speed = damp(this.speed, Math.min(7.5, 1.2 + dist * 0.9), 2, dt);
        this.gust = damp(this.gust, 0, 3, dt);
        const step = Math.min(dist, this.speed * dt);
        if (dist > 1e-4) for (let j = 0; j < 3; j++) this.pos[j] += ([dx, dy, dz][j] / dist) * step;
        if (dist < 0.08) {
          this.pos.splice(0, 3, t[0], t[1], t[2]);
          this.perched = true;
          this.landing = null;
          this.pitch = 0;
          this.orbit = this.yaw + Math.PI * 0.6;
        }
        this.recordTrail();
        return;
      }
    }
    if (this.riding) {
      // Carried up the thermal: a rising spiral around its core, wings still. Diving breaks
      // out early; at the top the wind glides out along the circle's tangent.
      const r = this.riding;
      const w = (r.dir * 9) / r.radius;
      r.angle += w * dt;
      const wantR = r.radius;
      const cx = r.x + Math.cos(r.angle) * wantR;
      const cz = r.z + Math.sin(r.angle) * wantR;
      this.pos[0] += (cx - this.pos[0]) * Math.min(1, dt * 2.5);
      this.pos[2] += (cz - this.pos[2]) * Math.min(1, dt * 2.5);
      const climb = Math.min(4.2, Math.max(0.6, (r.top - this.pos[1]) * 0.5));
      this.pos[1] += climb * dt;
      // Heading along the circle; nose a little up while climbing.
      const tx = -Math.sin(r.angle) * r.dir;
      const tz = Math.cos(r.angle) * r.dir;
      const want = Math.atan2(tx, -tz);
      let dy = want - this.yaw;
      dy = Math.atan2(Math.sin(dy), Math.cos(dy));
      this.yaw += dy * Math.min(1, dt * 4);
      this.pitch = damp(this.pitch, 0.18, 2, dt);
      this.speed = damp(this.speed, 9, 1.5, dt);
      this.gust = damp(this.gust, 0, 2, dt);
      this.thermal = 1;
      this.ceilingExtra = THERMAL_EXTRA;
      this.ceilingGround = damp(this.ceilingGround, terrainHeight(this.pos[0], this.pos[2]), 1.2, dt);
      if (this.pos[1] >= r.top - 0.3 || input.dive) {
        this.riding = null;
        this.rideCooldown = 5;
        this.pitch = 0;
      }
      this.recordTrail();
      return;
    }
    if (this.looping) {
      // A vertical loop: the pitch turns steadily through a full circle (radius ~6 m) at a
      // brisk speed; heading is held; out of it the wind flies on as it was going.
      this.gust = damp(this.gust, 0.6, 3, dt);
      this.speed = damp(this.speed, 13, 2, dt);
      this.yaw = this.loopYaw;
      this.pitch += (this.speed / 6.2) * dt;
      const f = this.forward;
      for (let j = 0; j < 3; j++) this.pos[j] += f[j] * this.speed * dt;
      const g = terrainHeight(this.pos[0], this.pos[2]);
      if (this.pos[1] < g + 0.6) this.pos[1] = g + 0.6;
      if (this.pitch >= Math.PI * 2) {
        this.pitch -= Math.PI * 2;
        this.looping = false;
        this.ceilingGround = g;
      }
      this.recordTrail();
      return;
    }
    this.gust = damp(this.gust, gusting ? 1 : 0, gusting ? 3 : 1.6, dt);
    // Diving trades height for speed (a stoop): the steeper the dive, the more it builds; it
    // carries on after the pull-out and fades over a few seconds.
    this.stoop = damp(this.stoop, input.dive && this.pitch < -0.2 ? 1 : 0, 4, dt);
    this.momentum = Math.max(0, this.momentum + (Math.max(0, -Math.sin(this.pitch)) * 16 * this.stoop - this.momentum * 0.35) * dt);
    this.speed = damp(this.speed, CRUISE + (GUST - CRUISE) * this.gust + Math.min(12, this.momentum), 1.8, dt);
    // Barrel roll: a full turn about the long axis in 0.8 s, sidestepping a little that way.
    if (this.rollT >= 0) {
      this.rollT += dt / 0.8;
      const k = Math.min(1, this.rollT);
      const ease = k * k * (3 - 2 * k);
      this.rollAngle = this.rollDir * ease * Math.PI * 2;
      const side = Math.sin(Math.PI * k) * 3.2 * this.rollDir * dt;
      this.pos[0] += Math.cos(this.yaw) * side;
      this.pos[2] += Math.sin(this.yaw) * side;
      if (this.rollT >= 1) {
        this.rollT = -1;
        this.rollAngle = 0;
      }
    }
    this.yaw += sx * Math.abs(sx) * 1.9 * dt;

    // Pitch: pointer height plus rise/dive; otherwise glide back to skimming altitude.
    const ground = terrainHeight(this.pos[0], this.pos[2]);
    const fwd = this.forward;
    const ahead = terrainHeight(this.pos[0] + fwd[0] * 6, this.pos[2] + fwd[2] * 6);
    const groundSlope = Math.atan2(ahead - ground, 6);
    // In rising air the wind holds its height (no pull back down to skimming altitude);
    // otherwise it glides back down toward the grass.
    this.ceilingExtra = Math.max(this.ceilingExtra - dt * 1.2, this.thermal * (THERMAL_EXTRA));
    const holdHeight = this.ceilingExtra > 1;
    let targetPitch = groundSlope + (ground + CRUISE_ALT - this.pos[1]) * (holdHeight ? 0.012 : 0.12);
    if (input.rise) targetPitch = 0.55;
    else if (input.dive) targetPitch = -0.45;
    else if (sy !== 0) targetPitch = -sy * 0.75 + groundSlope * 0.5;
    // Soft ceiling over a smoothed ground (so it does not trace every bump): the climb eases
    // off as the wind nears it instead of being clamped, which made the camera judder.
    this.ceilingGround = this.ceilingInit ? damp(this.ceilingGround, ground, 1.2, dt) : ground;
    this.ceilingInit = true;
    const room = this.ceilingGround + MAX_ALT + this.ceilingExtra - this.pos[1];
    targetPitch = Math.min(targetPitch, Math.max(-0.35, room * 0.12));
    // Soft floor: nearing the grass, a dive eases into skimming along the ground's slope
    // (instead of hitting a hard clamp every frame, which shook the camera).
    const floorY = Math.max(ground, ahead) + 0.75;
    targetPitch = Math.max(targetPitch, groundSlope + (floorY - this.pos[1]) * 0.6);
    targetPitch = Math.max(-0.9, Math.min(0.9, targetPitch));
    this.pitch = damp(this.pitch, targetPitch, 2.5, dt);

    const f = this.forward;
    this.pos[0] += f[0] * this.speed * dt;
    this.pos[1] += f[1] * this.speed * dt;
    this.pos[2] += f[2] * this.speed * dt;
    const g = terrainHeight(this.pos[0], this.pos[2]);
    // Backstop for sudden rises in the ground: lift, without snapping the pitch.
    if (this.pos[1] < g + 0.45) this.pos[1] = g + 0.45;
    // Backstop only (e.g. flying off a cliff edge): settle down smoothly.
    // The thermal carries the wind up: lift without a wingbeat.
    if (this.thermal > 0.02) this.pos[1] += this.thermal * 3.4 * dt;
    if (this.pos[1] > this.ceilingGround + MAX_ALT + this.ceilingExtra + 3) this.pos[1] = damp(this.pos[1], this.ceilingGround + MAX_ALT + this.ceilingExtra + 3, 2, dt);

    this.recordTrail();
    for (const t of this.trail) t.age += dt;
  }

  private recordTrail(): void {
    // Trail for the grass push: newest first, fading with age. Samples are dropped by distance
    // (not time) and the head follows the wind continuously, so the push slides smoothly.
    const head = this.trail[0];
    if (!head || Math.hypot(head.p[0] - this.pos[0], head.p[2] - this.pos[2]) > TRAIL_STEP) {
      this.trail.unshift({ p: [this.pos[0], this.pos[1], this.pos[2]], age: 0 });
      if (this.trail.length > TRAIL_LEN) this.trail.length = TRAIL_LEN;
    }
  }

  /** Writes trail samples as (x, y, z, strength) into the globals' trail array. */
  writeTrail(out: Float32Array): void {
    out.fill(0);
    const strengthBySpeed = 0.55 + 0.45 * Math.min(1, (this.speed - CRUISE * 0.5) / (GUST - CRUISE * 0.5));
    const gain = strengthBySpeed * (1 + this.gust * 0.6);
    // Slot 0 is the live position of the wind; history follows.
    out.set([this.pos[0], this.pos[1], this.pos[2], gain], 0);
    this.trail.slice(0, TRAIL_LEN - 1).forEach((t, i) => {
      const k = Math.max(0, 1 - t.age / TRAIL_LIFE);
      // Smoothstep fade so a sample releases the grass gently at the end of its life.
      out.set([t.p[0], t.p[1], t.p[2], k * k * (3 - 2 * k) * gain], (i + 1) * 4);
    });
  }

  /** Smooth third-person chase camera that stays above the grass. */
  updateCamera(camera: Camera, dt: number, streamLength = 0): void {
    if (this.perched) {
      // Resting: the camera drifts slowly round the bird, a little above it.
      this.orbit += dt * 0.12;
      const desired: Vec3 = [this.pos[0] + Math.sin(this.orbit) * 2.8, this.pos[1] + 0.7, this.pos[2] - Math.cos(this.orbit) * 2.8];
      const g = smoothGround(desired[0], desired[2]) + 0.4;
      desired[1] = Math.max(desired[1], g);
      for (let i = 0; i < 3; i++) this.camPos[i] = damp(this.camPos[i], desired[i], 1.2, dt);
      camera.position.splice(0, 3, ...this.camPos);
      camera.target.splice(0, 3, this.pos[0], this.pos[1] + 0.12, this.pos[2]);
      camera.fovY = (50 * Math.PI) / 180;
      return;
    }
    const f = this.forward;
    const back = (5.5 + streamLength * 0.75 + this.gust * 2.5) * this.followScale * (this.looping ? 1.6 : this.riding ? 1.5 : 1);
    // Behind along the heading (not the pitch: through a loop the pitch turns all the way
    // round, and the camera must stay put behind it rather than flip over).
    const hx = Math.sin(this.yaw);
    const hz = -Math.cos(this.yaw);
    const rise = this.looping ? 2.5 : (1.6 - f[1] * 2.5) * (0.4 + 0.6 * this.followScale);
    const desired: Vec3 = [this.pos[0] - hx * back, this.pos[1] + rise, this.pos[2] - hz * back];
    const ground = smoothGround(desired[0], desired[2]);
    desired[1] = Math.max(desired[1], ground + 1.3);
    if (!this.camInit) {
      this.camPos.splice(0, 3, ...desired);
      this.camInit = true;
    }
    for (let i = 0; i < 3; i++) this.camPos[i] = damp(this.camPos[i], desired[i], 4.5, dt);
    // Keep above the ground under the camera, eased over a few meters of ground so small
    // bumps do not jolt it; a hard limit only prevents dipping into the terrain.
    const under = smoothGround(this.camPos[0], this.camPos[2]) + 1.1;
    if (this.camPos[1] < under) this.camPos[1] = damp(this.camPos[1], under, 10, dt);
    this.camPos[1] = Math.max(this.camPos[1], terrainHeight(this.camPos[0], this.camPos[2]) + 0.35);
    camera.position.splice(0, 3, ...this.camPos);
    if (this.looping) camera.target.splice(0, 3, this.pos[0], this.pos[1], this.pos[2]);
    else camera.target.splice(0, 3, this.pos[0] + f[0] * 4, this.pos[1] + f[1] * 4 + 0.3, this.pos[2] + f[2] * 4);
    camera.fovY = ((60 + this.gust * 12) * Math.PI) / 180;
  }
}
