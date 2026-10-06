import { draw, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import trailShader from "../shaders/wind-trail.wgsl";
import type { Vec3 } from "../engine/camera";

const LINES = 28;
const POINTS = 24;

interface Line {
  pos: Vec3;
  vel: Vec3;
  /** Newest first. */
  trail: Vec3[];
  age: number;
  life: number;
  width: number;
  /** Curl rate (rad/s) late in life; 0 for a straight line. */
  curl: number;
  phase: number;
  /** The wind's velocity this line rides with, relative offset kept as it travels. */
  speedup: number;
}

/**
 * Wind lines: short streaks of air that appear around the wind, ride along with it (a little
 * faster, so they draw ahead), sometimes curl into a loop, and fade. More and longer in a gust.
 */
export class WindTrail {
  readonly draw: Draw;
  private readonly lines: Line[] = [];
  private readonly ptsBuf: StorageBuffer;
  private readonly metaBuf: StorageBuffer;
  private readonly pts = new Float32Array(LINES * POINTS * 4);
  private readonly meta = new Float32Array(LINES * 4);
  private spawnDebt = 0;
  private last: Vec3 | null = null;
  private count = 0;

  constructor(gpu: Gpu, globals: SharedUniforms) {
    this.ptsBuf = storage(gpu, this.pts.byteLength, "read");
    this.metaBuf = storage(gpu, this.meta.byteLength, "read");
    this.draw = draw(gpu, {
      label: "wind-lines",
      shader: trailShader,
      vertices: (POINTS - 1) * 6,
      blend: "additive",
      cull: "none",
      depth: { compare: "greater", write: false },
      set: { G: globals, pts: this.ptsBuf, lines: this.metaBuf },
    });
  }

  reset(): void {
    this.lines.length = 0;
    this.last = null;
  }

  update(dt: number, leader: Vec3, speed: number, gust: number): void {
    // The wind's actual velocity, from its motion this frame.
    const vel: Vec3 = this.last && dt > 0 ? [(leader[0] - this.last[0]) / dt, (leader[1] - this.last[1]) / dt, (leader[2] - this.last[2]) / dt] : [0, 0, 0];
    if (this.last && Math.hypot(leader[0] - this.last[0], leader[2] - this.last[2]) > 20) this.lines.length = 0;
    this.last = [leader[0], leader[1], leader[2]];
    const flat = Math.hypot(vel[0], vel[2]);
    const fx = flat > 0.1 ? vel[0] / flat : 0;
    const fz = flat > 0.1 ? vel[2] / flat : 1;

    this.spawnDebt += dt * (1.5 + speed * 0.25 + gust * 7);
    while (this.spawnDebt >= 1 && this.lines.length < LINES) {
      this.spawnDebt -= 1;
      if (flat < 2) continue;
      // Around the wind: beside and above it, from a little behind to ahead.
      const side = (Math.random() < 0.5 ? -1 : 1) * (0.6 + Math.random() * 2.6);
      const up = -0.3 + Math.random() * 2.2;
      const ahead = -2 + Math.random() * 5;
      this.lines.push({
        pos: [leader[0] + fx * ahead - fz * side, leader[1] + up, leader[2] + fz * ahead + fx * side],
        vel: [0, 0, 0],
        trail: [],
        age: 0,
        life: 0.9 + Math.random() * 0.8,
        width: 0.016 + Math.random() * 0.016,
        curl: Math.random() < 0.35 ? (Math.random() < 0.5 ? -1 : 1) * (7 + Math.random() * 4) : 0,
        phase: 0,
        speedup: 1.12 + Math.random() * 0.12,
      });
    }
    this.spawnDebt = Math.min(this.spawnDebt, 1);

    let n = 0;
    for (let i = this.lines.length - 1; i >= 0; i--) {
      const l = this.lines[i];
      l.age += dt;
      const k = l.age / l.life;
      if (k >= 1) {
        this.lines.splice(i, 1);
        continue;
      }
      // Ride with the wind, a little faster; a gentle lateral sway, and late in life an
      // optional loop: the travel direction turns through a full circle about the vertical.
      let vx = vel[0] * l.speedup;
      let vy = vel[1] * l.speedup + Math.sin(l.age * 5 + l.width * 900) * 0.4;
      let vz = vel[2] * l.speedup;
      if (l.curl && k > 0.35) {
        l.phase += l.curl * dt;
        if (Math.abs(l.phase) < Math.PI * 2) {
          // Relative to the wind, the line swings round in a loop (in the vertical plane).
          const rel = flat * 0.9;
          const a = l.phase;
          vx += fx * (Math.cos(a) - 1) * rel;
          vz += fz * (Math.cos(a) - 1) * rel;
          vy += Math.sin(a) * rel * Math.sign(l.curl);
        }
      }
      l.pos[0] += vx * dt;
      l.pos[1] += vy * dt;
      l.pos[2] += vz * dt;
      l.trail.unshift([l.pos[0], l.pos[1], l.pos[2]]);
      if (l.trail.length > POINTS) l.trail.length = POINTS;
      const alpha = Math.min(1, k / 0.2) * (1 - Math.max(0, (k - 0.6) / 0.4)) * (0.55 + 0.45 * Math.min(1, gust + speed / 21));
      const base = n * POINTS * 4;
      l.trail.forEach((p, j) => this.pts.set([p[0], p[1], p[2], 0], base + j * 4));
      this.meta.set([l.trail.length, alpha, l.width * (1 + gust * 0.4), 0], n * 4);
      n++;
    }
    this.count = n;
    if (n) {
      this.ptsBuf.write(this.pts.subarray(0, n * POINTS * 4));
      this.metaBuf.write(this.meta.subarray(0, n * 4));
    }
  }

  encode(pass: FramePass): void {
    if (this.count) pass.draw(this.draw, { instances: this.count });
  }
}
