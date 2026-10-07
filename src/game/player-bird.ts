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
const SIZE = 2.8;

/**
 * Playing as a bird: a barn swallow (tools/blender/model_swallow.py) flies where the wind
 * goes. It banks into turns, beats hard when gusting or climbing, and otherwise flies as
 * swallows do: a burst of wingbeats, then a glide on swept wings.
 */
export class PlayerBird {
  readonly draw: Draw;
  readonly haloDraw: Draw;
  /** Its own soft light (0 none .. 1 full), brighter as it gets dark. */
  glow = 0;
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

  private constructor(d: Draw, halo: Draw, buffer: StorageBuffer, private readonly elbow: number) {
    this.draw = d;
    this.haloDraw = halo;
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
    const halo = draw(gpu, {
      label: "swallow-light",
      shader: birdShader,
      entry: { vertex: "vs_halo", fragment: "fs_halo" },
      vertices: 6,
      blend: "additive",
      depth: { compare: "greater", write: false },
      constants: { SHOULDER: manifest.shoulder },
      set: { G: globals, birds: buffer },
    });
    return new PlayerBird(d, halo, buffer, manifest.elbow);
  }

  /**
   * The flight cycle, keyframed on a phase (0 at the top of the stroke):
   *   downstroke (first half)  the extended wing sweeps down, the hand lagging the arm then
   *                            flicking through at the bottom
   *   upstroke (second half)   the wing flexes: the hand folds back toward the body and the
   *                            arm lifts it, short and quick
   * Beats come in a few at a time with long glides between; everything eases (no jumps).
   */
  update(dt: number, player: Player): void {
    // Turn rate -> bank: lean into the turn, eased.
    if (this.lastYaw !== null && dt > 0) {
      let d = player.yaw - this.lastYaw;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      this.yawRate += (d / dt - this.yawRate) * Math.min(1, dt * 4);
    }
    this.lastYaw = player.yaw;
    const looping = player.looping;
    const wantRoll = looping ? 0 : Math.max(-0.85, Math.min(0.85, this.yawRate * 0.5));
    this.roll += (wantRoll - this.roll) * Math.min(1, dt * 3);

    // Beating or gliding: hard work when gusting, climbing or pulling a loop; otherwise
    // three easy beats then a long glide.
    const soaring = player.thermal > 0.25;
    const working = (player.gust > 0.25 || player.pitch > 0.12 || (looping && player.pitch < 1.4)) && player.stoop < 0.5 && !soaring;
    this.cycle = (this.cycle + dt) % 4.6;
    // Soaring in a thermal: wings held wide and still.
    const beating = working || (this.cycle < 2.3 && !soaring);
    const wantAmp = working ? 1 : beating ? 0.8 : 0;
    this.flapAmp += (wantAmp - this.flapAmp) * Math.min(1, dt * 1.6);
    const wantFreq = working ? 1.9 : 1.35;
    this.freq += (wantFreq - this.freq) * Math.min(1, dt * 1.2);
    // Keep the cycle turning slowly while gliding so beats resume from where they paused.
    this.phase += dt * 2 * Math.PI * this.freq * (0.25 + 0.75 * Math.min(1, this.flapAmp * 3));
    const ph = this.phase;
    const k = this.flapAmp;
    // Arm: from +0.75 rad at the top to -0.6 at the bottom (the downstroke is the longer,
    // powered half: warp the phase so it takes ~58% of the cycle).
    const warped = ph % (2 * Math.PI);
    const down = warped < Math.PI * 1.16;
    const t = down ? warped / (Math.PI * 1.16) : (warped - Math.PI * 1.16) / (Math.PI * 0.84);
    const ease = (x: number) => x * x * (3 - 2 * x);
    const armTop = 0.72;
    const armBottom = -0.58;
    const arm = down ? armTop + (armBottom - armTop) * ease(t) : armBottom + (armTop - armBottom) * ease(t);
    // Hand: lags the arm (still raised early in the downstroke, flicking through at the end);
    // on the upstroke it trails down and folds back.
    const hand = down ? 0.25 * (1 - ease(Math.min(1, t * 1.6))) - 0.3 * Math.max(0, t - 0.6) / 0.4 : -0.35 * Math.sin(Math.PI * t);
    const fold = down ? 0 : Math.sin(Math.PI * t) ** 1.3 * 0.95;
    // Glide pose: wings held level with a slight lift, hands swept back as speed rises.
    const glideArm = 0.06;
    const glideHand = -0.04;
    const glideSweep = Math.min(0.6, 0.15 + Math.max(0, (player.speed - 8) / 18));
    let armA = glideArm + (arm - glideArm) * k;
    let handA = glideHand + (hand - glideHand) * k;
    let sweep = glideSweep + (fold - glideSweep) * k;
    // Stoop: wings tucked back along the body, like an arrow.
    const st = player.stoop;
    armA = armA + (0.18 - armA) * st;
    handA = handA + (-0.15 - handA) * st;
    sweep = sweep + (1.0 - sweep) * st;
    this.sweep += (sweep - this.sweep) * Math.min(1, dt * 12);

    const p = player.pos;
    // A little body rise and fall with each stroke (the body answers the wings).
    const bob = -Math.sin(ph) * 0.012 * SIZE * k;
    const pitch = looping ? -player.pitch : -player.pitch * 0.85;
    // The shader's heading maps +Z to (sin, cos); the player's forward is (sin yaw, -cos yaw).
    this.data.set([p[0], p[1] + bob, p[2], Math.PI - player.yaw, armA, handA, pitch, 1 + this.glow * 0.6, this.roll + player.rollAngle, SIZE, this.elbow, this.sweep]);
    this.buffer.write(this.data);
  }

  encode(pass: FramePass): void {
    if (this.visible) {
      pass.draw(this.draw);
      if (this.glow > 0.02) pass.draw(this.haloDraw);
    }
  }
}
