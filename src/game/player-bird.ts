import { draw, geometry, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import birdShader from "../shaders/birds.wgsl";
import type { Player } from "./player";

interface Manifest {
  vertexBytes: number;
  indexCount: number;
  shoulder: number;
  elbow: number;
}

/** Times life size: big enough to read from the chase camera, still a swallow. */
const SIZE = 3.6;

/**
 * Playing as a bird: a barn swallow (tools/blender/model_swallow.py) flies where the wind
 * goes. It banks into turns, beats hard when gusting or climbing, and otherwise flies as
 * swallows do: a burst of wingbeats, then a glide on swept wings.
 */
export class PlayerBird {
  readonly draw: Draw;
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(12);
  private phase = 0;
  private flapAmp = 0;
  private sweep = 0;
  private roll = 0;
  private lastYaw: number | null = null;
  private yawRate = 0;
  private cycle = 0;
  private freq = 2.8;
  visible = false;

  private constructor(d: Draw, buffer: StorageBuffer, private readonly elbow: number) {
    this.draw = d;
    this.buffer = buffer;
  }

  static async load(gpu: Gpu, globals: SharedUniforms, base = "assets/swallow"): Promise<PlayerBird> {
    const [manifest, bin] = await Promise.all([
      fetch(`${base}/swallow.json`).then((r) => r.json() as Promise<Manifest>),
      fetch(`${base}/swallow.bin`).then((r) => r.arrayBuffer()),
    ]);
    const geo = geometry(gpu, {
      label: "swallow",
      buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
    });
    const buffer = storage(gpu, 48, "read");
    const d = draw(gpu, {
      label: "swallow",
      shader: birdShader,
      geometry: geo,
      cull: "none",
      depth: { compare: "greater" },
      constants: { SHOULDER: manifest.shoulder },
      set: { G: globals, birds: buffer },
    });
    return new PlayerBird(d, buffer, manifest.elbow);
  }

  update(dt: number, player: Player): void {
    // Turn rate -> bank: lean into the turn, eased.
    if (this.lastYaw !== null && dt > 0) {
      let d = player.yaw - this.lastYaw;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      this.yawRate += (d / dt - this.yawRate) * Math.min(1, dt * 6);
    }
    this.lastYaw = player.yaw;
    const wantRoll = Math.max(-0.9, Math.min(0.9, this.yawRate * 0.55));
    this.roll += (wantRoll - this.roll) * Math.min(1, dt * 5);

    // Wingbeats: hard and steady when gusting or climbing; otherwise bursts and glides.
    const climbing = player.pitch > 0.12;
    const working = player.gust > 0.25 || climbing;
    this.cycle = (this.cycle + dt) % 3.2;
    const burst = working || this.cycle < 1.1;
    // Big, unhurried strokes (it is drawn large, so it flaps slower than a real swallow),
    // easing in and out of glides: no sudden changes of rate or reach.
    const wantAmp = working ? 0.62 : burst ? 0.48 : 0.05;
    this.flapAmp += (wantAmp - this.flapAmp) * Math.min(1, dt * 2.2);
    const wantFreq = working ? 3.6 : 2.8;
    this.freq += (wantFreq - this.freq) * Math.min(1, dt * 1.5);
    this.phase += dt * 2 * Math.PI * this.freq;
    const flap = Math.sin(this.phase) * this.flapAmp + 0.06 + (1 - Math.min(1, this.flapAmp / 0.3)) * 0.06;
    // The upstroke folds the hand back a little; fast glides keep the wings swept.
    const up = (Math.cos(this.phase) * 0.5 + 0.5) * 0.45 * Math.min(1, this.flapAmp / 0.3);
    const fastSweep = Math.min(0.55, Math.max(0, (player.speed - 9) / 14)) * (1 - Math.min(1, this.flapAmp / 0.3) * 0.7);
    this.sweep += (Math.max(up, fastSweep) - this.sweep) * Math.min(1, dt * 6);

    const p = player.pos;
    // The shader's heading maps +Z to (sin, cos); the player's forward is (sin yaw, -cos yaw).
    this.data.set([p[0], p[1], p[2], Math.PI - player.yaw, flap, 1, -player.pitch * 0.85, 0.5, this.roll, SIZE, this.elbow, this.sweep]);
    this.buffer.write(this.data);
  }

  encode(pass: FramePass): void {
    if (this.visible) pass.draw(this.draw);
  }
}
