import type { Camera, Vec3 } from "../engine/camera";
import { surfaceHeight as terrainHeight } from "../world/height";
import type { Input } from "./input";

const CRUISE = 7.5;
const GUST = 21;
const CRUISE_ALT = 1.7;
/** The wind skims the field; it never climbs above the treetops of a meadow. */
const MAX_ALT = 10;
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
    // Soft ceiling over a smoothed ground (so it does not trace every bump): the climb eases
    // off as the wind nears it instead of being clamped, which made the camera judder.
    this.ceilingGround = this.ceilingInit ? damp(this.ceilingGround, ground, 1.2, dt) : ground;
    this.ceilingInit = true;
    const room = this.ceilingGround + MAX_ALT - this.pos[1];
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
    if (this.pos[1] > this.ceilingGround + MAX_ALT + 3) this.pos[1] = damp(this.pos[1], this.ceilingGround + MAX_ALT + 3, 2, dt);

    // Trail for the grass push: newest first, fading with age. Samples are dropped by distance
    // (not time) and the head follows the wind continuously, so the push slides smoothly.
    for (const t of this.trail) t.age += dt;
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
    const f = this.forward;
    const back = 5.5 + streamLength * 0.75 + this.gust * 2.5;
    const flat = Math.hypot(f[0], f[2]) || 1;
    const desired: Vec3 = [
      this.pos[0] - (f[0] / flat) * back,
      this.pos[1] + 1.6 - f[1] * 2.5,
      this.pos[2] - (f[2] / flat) * back,
    ];
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
    camera.target.splice(0, 3, this.pos[0] + f[0] * 4, this.pos[1] + f[1] * 4 + 0.3, this.pos[2] + f[2] * 4);
    camera.fovY = ((60 + this.gust * 12) * Math.PI) / 180;
  }
}
