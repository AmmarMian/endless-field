import type { Camera, Vec3 } from "../engine/camera";
import { terrainHeightM as terrainHeight } from "../world/height";
import type { Input } from "./input";

const CRUISE = 7.5;
const GUST = 21;
const CRUISE_ALT = 1.7;
const MAX_ALT = 45;
const TRAIL_LEN = 24;
const TRAIL_INTERVAL = 0.09;
const TRAIL_LIFE = TRAIL_LEN * TRAIL_INTERVAL;

function damp(current: number, target: number, rate: number, dt: number): number {
  return target + (current - target) * Math.exp(-rate * dt);
}

/** The wind: an invisible leader that the petal stream follows and the camera chases. */
export class Player {
  readonly pos: Vec3 = [0, 0, 0];
  yaw = 0;
  pitch = 0;
  speed = CRUISE;
  gust = 0;
  /** Recorded leader positions, newest first, for the grass push. */
  private readonly trail: { p: Vec3; age: number }[] = [];
  private trailTimer = 0;
  private readonly camPos: Vec3 = [0, 0, 0];
  private camInit = false;

  constructor(x: number, z: number, yaw: number) {
    this.pos[0] = x;
    this.pos[2] = z;
    this.pos[1] = terrainHeight(x, z) + CRUISE_ALT;
    this.yaw = yaw;
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

    this.gust = damp(this.gust, gusting ? 1 : 0, gusting ? 3 : 1.6, dt);
    this.speed = damp(this.speed, CRUISE + (GUST - CRUISE) * this.gust, 1.8, dt);
    this.yaw += sx * Math.abs(sx) * 1.9 * dt;

    // Pitch: pointer height plus rise/dive; otherwise glide back to skimming altitude.
    const ground = terrainHeight(this.pos[0], this.pos[2]);
    const fwd = this.forward;
    const ahead = terrainHeight(this.pos[0] + fwd[0] * 6, this.pos[2] + fwd[2] * 6);
    const groundSlope = Math.atan2(ahead - ground, 6);
    let targetPitch = groundSlope + (ground + CRUISE_ALT - this.pos[1]) * 0.12;
    if (input.rise) targetPitch = 0.55;
    else if (input.dive) targetPitch = -0.45;
    else if (sy !== 0) targetPitch = -sy * 0.75 + groundSlope * 0.5;
    targetPitch = Math.max(-0.9, Math.min(0.9, targetPitch));
    this.pitch = damp(this.pitch, targetPitch, 2.5, dt);

    const f = this.forward;
    this.pos[0] += f[0] * this.speed * dt;
    this.pos[1] += f[1] * this.speed * dt;
    this.pos[2] += f[2] * this.speed * dt;
    const g = terrainHeight(this.pos[0], this.pos[2]);
    if (this.pos[1] < g + 0.55) {
      this.pos[1] = g + 0.55;
      this.pitch = Math.max(this.pitch, 0);
    }
    if (this.pos[1] > g + MAX_ALT) this.pos[1] = damp(this.pos[1], g + MAX_ALT, 2, dt);

    // Trail for the grass push: newest first, fading with age.
    this.trailTimer += dt;
    for (const t of this.trail) t.age += dt;
    if (this.trailTimer >= TRAIL_INTERVAL) {
      this.trailTimer = 0;
      this.trail.unshift({ p: [this.pos[0], this.pos[1], this.pos[2]], age: 0 });
      if (this.trail.length > TRAIL_LEN) this.trail.length = TRAIL_LEN;
    }
  }

  /** Writes trail samples as (x, y, z, strength) into the globals' trail array. */
  writeTrail(out: Float32Array): void {
    out.fill(0);
    const strengthBySpeed = 0.55 + 0.45 * Math.min(1, (this.speed - CRUISE * 0.5) / (GUST - CRUISE * 0.5));
    this.trail.forEach((t, i) => {
      const k = Math.max(0, 1 - t.age / TRAIL_LIFE);
      out.set([t.p[0], t.p[1], t.p[2], k * k * strengthBySpeed * (1 + this.gust * 0.6)], i * 4);
    });
  }

  /** Smooth third-person chase camera that stays above the grass. */
  updateCamera(camera: Camera, dt: number, streamLength = 0): void {
    const f = this.forward;
    const back = 5.5 + streamLength * 0.75 + this.gust * 2.5;
    const flat = Math.hypot(f[0], f[2]) || 1;
    const desired: Vec3 = [
      this.pos[0] - (f[0] / flat) * back,
      this.pos[1] + 1.6 - f[1] * 2.5,
      this.pos[2] - (f[2] / flat) * back,
    ];
    const ground = terrainHeight(desired[0], desired[2]);
    desired[1] = Math.max(desired[1], ground + 1.3);
    if (!this.camInit) {
      this.camPos.splice(0, 3, ...desired);
      this.camInit = true;
    }
    for (let i = 0; i < 3; i++) this.camPos[i] = damp(this.camPos[i], desired[i], 4.5, dt);
    this.camPos[1] = Math.max(this.camPos[1], terrainHeight(this.camPos[0], this.camPos[2]) + 1.1);
    camera.position.splice(0, 3, ...this.camPos);
    camera.target.splice(0, 3, this.pos[0] + f[0] * 4, this.pos[1] + f[1] * 4 + 0.3, this.pos[2] + f[2] * 4);
    camera.fovY = ((60 + this.gust * 12) * Math.PI) / 180;
  }
}
