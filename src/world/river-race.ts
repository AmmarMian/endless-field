import { draw, geometry, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import gateShader from "../shaders/river-gate.wgsl";
import birdShader from "../shaders/birds.wgsl";
import type { Vec3 } from "../engine/camera";
import { riverCenter, riverHalfWidth, riverWater } from "./height";

interface Manifest {
  vertexBytes: number;
  indexCount: number;
  variants?: { name: string; firstIndex: number; indexCount: number }[];
  shoulder?: number;
  ringY?: number;
  radius?: number;
}

const GATES = 11;
/** After a win, the kingfisher flies with the wind this long (s), then goes home. */
const COMPANION_SECONDS = 180;
const SPACING = 36;
/** Where along the river (x) the course starts. */
const START_X = 70;
const KEY = "endless-field-river-race";

interface Gate {
  base: Vec3;
  centre: Vec3;
  /** Direction of travel through it (unit, horizontal). */
  dir: [number, number];
  yaw: number;
  state: number;
  flash: number;
}

type Phase = "waiting" | "racing" | "returning" | "companion";

/**
 * The kingfisher's course: a kingfisher waits on the first gate's pole. When the wind comes
 * close it darts off down the river, through a line of rope gates; the wind must follow it
 * through every gate. Fall too far behind or miss a gate and the bird flies home to wait.
 */
export class RiverRace {
  readonly draws: Draw[];
  readonly gates: Gate[];
  private readonly gateBuf: StorageBuffer;
  private readonly gateData: Float32Array<ArrayBuffer>;
  private readonly birdBuf: StorageBuffer;
  private readonly birdData = new Float32Array(8);
  private readonly gateDraw: Draw;
  private readonly birdDraws: Draw[];
  /** Kingfisher route: perch, every gate's centre, then the last gate's pole. */
  private readonly route: Vec3[];
  private phase: Phase = "waiting";
  private along = 0;
  private next = 0;
  private prevSide = 0;
  private bird: Vec3;
  private birdVel: Vec3 = [0, 0, 0];
  private flapPhase = 0;
  private readonly perch: Vec3;
  private readonly perchEnd: Vec3;
  private readonly radius: number;
  done: boolean;
  onStart: (() => void) | null = null;
  onGate: ((index: number, count: number) => void) | null = null;
  onFail: (() => void) | null = null;
  onComplete: (() => void) | null = null;

  private readonly haloDraw: Draw;

  private constructor(gateDraw: Draw, haloDraw: Draw, birdDraws: Draw[], gateBuf: StorageBuffer, birdBuf: StorageBuffer, gates: Gate[], ringY: number, radius: number) {
    this.gateDraw = gateDraw;
    this.birdDraws = birdDraws;
    this.draws = [gateDraw, ...birdDraws];
    this.haloDraw = haloDraw;
    this.draws.push(haloDraw);
    this.gateBuf = gateBuf;
    this.birdBuf = birdBuf;
    this.gates = gates;
    this.radius = radius;
    this.gateData = new Float32Array(gates.length * 8);
    // The kingfisher waits on top of the first floating ring, and finishes on the last.
    const top = (g: Gate): Vec3 => [g.centre[0], g.centre[1] + radius + 0.04, g.centre[2]];
    this.perch = top(gates[0]);
    this.perchEnd = top(gates[gates.length - 1]);
    this.bird = [...this.perch];
    this.route = [this.perch, ...gates.map((g) => g.centre), this.perchEnd];
    try {
      this.done = localStorage.getItem(KEY) === "1";
    } catch {
      this.done = false;
    }
    // A past win: the rings stay lit, but the kingfisher is home at the river (race again).
    if (this.done) for (const g of gates) g.state = 2;
  }

  static async load(gpu: Gpu, globals: SharedUniforms): Promise<RiverRace> {
    const get = (base: string, name: string) =>
      Promise.all([fetch(`${base}/${name}.json`).then((r) => r.json() as Promise<Manifest>), fetch(`${base}/${name}.bin`).then((r) => r.arrayBuffer())]);
    const [[gm, gbin], [bm, bbin]] = await Promise.all([get("assets/gate", "gate"), get("assets/bird", "bird")]);
    const ringY = gm.ringY ?? 2.3;
    const radius = gm.radius ?? 1.5;
    // Course: gates down the river, weaving a little across it, each turned along the flow.
    const gates: Gate[] = [];
    for (let i = 0; i < GATES; i++) {
      const x = START_X + 20 + i * SPACING;
      const hw = riverHalfWidth(x);
      const z = riverCenter(x) + Math.sin(i * 1.4) * hw * 0.3;
      const dz = (riverCenter(x + 1) - riverCenter(x - 1)) / 2;
      const len = Math.hypot(1, dz);
      const dir: [number, number] = [1 / len, dz / len];
      const base: Vec3 = [x, riverWater(x) - 0.05, z];
      gates.push({ base, centre: [x, base[1] + ringY, z], dir, yaw: Math.atan2(dir[0], dir[1]), state: 0, flash: 0 });
    }
    const gateGeo = geometry(gpu, {
      label: "river-gate",
      buffers: [{ data: new Uint8Array(gbin, 0, gm.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(gbin, gm.vertexBytes, gm.indexCount),
    });
    const gateBuf = storage(gpu, GATES * 32, "read");
    const gateDraw = draw(gpu, { label: "river-gates", shader: gateShader, geometry: gateGeo, cull: "none", depth: { compare: "greater" }, set: { G: globals, gates: gateBuf } });
    const birdGeo = geometry(gpu, {
      label: "kingfisher",
      buffers: [{ data: new Uint8Array(bbin, 0, bm.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(bbin, bm.vertexBytes, bm.indexCount),
    });
    const birdBuf = storage(gpu, 32, "read");
    const kf = (bm.variants ?? []).filter((v) => v.name.startsWith("kingfisher"));
    const birdDraws = kf.map((v) =>
      draw(gpu, {
        label: v.name,
        shader: birdShader,
        geometry: birdGeo.slice({ firstIndex: v.firstIndex, indexCount: v.indexCount }),
        cull: "none",
        depth: { compare: "greater" },
        constants: { SHOULDER: bm.shoulder ?? 0.022 },
        set: { G: globals, birds: birdBuf },
      }),
    );
    const haloDraw = draw(gpu, {
      label: "kingfisher-light",
      shader: birdShader,
      entry: { vertex: "vs_halo", fragment: "fs_halo" },
      vertices: 6,
      blend: "additive",
      depth: { compare: "greater", write: false },
      constants: { SHOULDER: bm.shoulder ?? 0.022 },
      set: { G: globals, birds: birdBuf },
    });
    return new RiverRace(gateDraw, haloDraw, birdDraws, gateBuf, birdBuf, gates, ringY, radius);
  }

  /** Where the race begins (for the birds to lead the wind to). */
  get start(): [number, number] {
    return [this.perch[0], this.perch[2]];
  }

  /** Point on the route at arc length `s` (straight legs between route points). */
  private routeAt(s: number): Vec3 {
    let left = s;
    for (let i = 0; i < this.route.length - 1; i++) {
      const a = this.route[i];
      const b = this.route[i + 1];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      if (left <= len) {
        const k = left / len;
        return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
      }
      left -= len;
    }
    return [...this.route[this.route.length - 1]];
  }

  private routeLength(): number {
    let l = 0;
    for (let i = 0; i < this.route.length - 1; i++) {
      const a = this.route[i];
      const b = this.route[i + 1];
      l += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    }
    return l;
  }

  /** Arc length at which the route reaches gate `i`. */
  private gateAlong(i: number): number {
    let l = 0;
    for (let k = 0; k <= i; k++) {
      const a = this.route[k];
      const b = this.route[k + 1];
      l += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    }
    return l;
  }

  private fail(): void {
    this.phase = "returning";
    for (const g of this.gates) g.state = this.done ? 2 : 0;
    this.onFail?.();
  }

  /** `windSpeed`: the wind's speed right now; the kingfisher flies exactly as fast. */
  update(dt: number, wind: Vec3, windSpeed: number): void {
    const toBird = Math.hypot(wind[0] - this.bird[0], wind[1] - this.bird[1], wind[2] - this.bird[2]);
    let speed = 0;
    let target: Vec3 = this.bird;
    if (this.phase === "waiting") {
      target = this.perch;
      // Gates show the way only once the race is on; the first one hints where to begin.
      for (const [i, g] of this.gates.entries()) g.state = this.done ? (i === 0 ? 1 : 2) : i === 0 ? 1 : 0;
      if (toBird < 7) {
        this.phase = "racing";
        this.along = 0;
        this.next = 0;
        const g0 = this.gates[0];
        this.prevSide = (wind[0] - g0.centre[0]) * g0.dir[0] + (wind[2] - g0.centre[2]) * g0.dir[1];
        for (const g of this.gates) g.state = 0;
        this.gates[0].state = 1;
        this.onStart?.();
      }
    } else if (this.phase === "racing") {
      // Exactly as fast as the wind, gusting when it gusts: no rubber band. Falling behind
      // only happens by flying a longer line than the bird.
      speed = windSpeed;
      this.along += speed * dt;
      target = this.routeAt(this.along);
      // The wind's progress through the next gate.
      const g = this.gates[this.next];
      const side = (wind[0] - g.centre[0]) * g.dir[0] + (wind[2] - g.centre[2]) * g.dir[1];
      if (this.prevSide < 0 && side >= 0) {
        const px = wind[0] - g.centre[0] - g.dir[0] * side;
        const pz = wind[2] - g.centre[2] - g.dir[1] * side;
        const off = Math.hypot(px, pz, wind[1] - g.centre[1]);
        if (off < this.radius + 0.6) {
          g.state = 2;
          g.flash = 1;
          this.onGate?.(this.next, this.gates.length);
          this.next++;
          if (this.next >= this.gates.length) {
            // Won: a wave of light runs back through every ring, and the kingfisher stays
            // with the wind from now on.
            this.phase = "companion";
            this.companionLeft = COMPANION_SECONDS;
            this.wave = 0;
            this.done = true;
            try {
              localStorage.setItem(KEY, "1");
            } catch {
              // Non-essential.
            }
            this.onComplete?.();
          } else {
            this.gates[this.next].state = 1;
            const n = this.gates[this.next];
            this.prevSide = (wind[0] - n.centre[0]) * n.dir[0] + (wind[2] - n.centre[2]) * n.dir[1];
          }
        } else if (off < this.radius + 8) {
          // Went past it, not through it.
          this.fail();
        } else this.prevSide = side;
      } else this.prevSide = side;
      // Lost the bird: it is two gates further on, or far away.
      if (this.phase === "racing" && (this.along > this.gateAlong(Math.min(this.next + 2, this.gates.length - 1)) + 4 || toBird > 70)) this.fail();
      if (this.phase === "racing" && this.along >= this.routeLength()) this.along = this.routeLength();
    } else if (this.phase === "companion" && (this.companionLeft -= dt) <= 0) {
      // Its time with the wind is over: home to the river.
      this.phase = "returning";
      this.next = 0;
      target = this.perch;
    } else if (this.phase === "companion") {
      // Flies beside the wind wherever it goes: a little to its right, a little above.
      const mv: Vec3 = this.lastWind ? [wind[0] - this.lastWind[0], 0, wind[2] - this.lastWind[2]] : [0, 0, 1];
      const ml = Math.hypot(mv[0], mv[2]);
      if (ml > 1e-4) this.heading = [mv[0] / ml, mv[2] / ml];
      const [fx, fz] = this.heading;
      const sway = Math.sin(performance.now() / 900) * 0.4;
      target = [wind[0] - fz * (1.9 + sway) - fx * 0.4, wind[1] + 0.7 + Math.sin(performance.now() / 650) * 0.25, wind[2] + fx * (1.9 + sway) - fz * 0.4];
      speed = Math.max(windSpeed * 1.4, 3) + Math.hypot(target[0] - this.bird[0], target[2] - this.bird[2]) * 2;
    } else {
      // Home to the perch to wait for another try (it flies the whole way, however far).
      target = this.perch;
      speed = 14;
      if (Math.hypot(this.bird[0] - target[0], this.bird[1] - target[1], this.bird[2] - target[2]) < 0.3) this.phase = "waiting";
    }
    this.lastWind = [wind[0], wind[1], wind[2]];
    // The wave of light after a win.
    if (this.wave >= 0) {
      this.wave += dt;
      this.gates.forEach((g, i) => {
        if (Math.abs(this.wave - i * 0.14) < dt) g.flash = 1;
      });
      if (this.wave > this.gates.length * 0.14 + 1) this.wave = -1;
    }
    // Fly toward the target: quick, straight darts low over the water.
    const dx = target[0] - this.bird[0];
    const dy = target[1] - this.bird[1];
    const dz = target[2] - this.bird[2];
    const dist = Math.hypot(dx, dy, dz);
    const perched = this.phase === "waiting" && dist < 0.05;
    if (!perched) {
      const v = this.phase === "returning" ? Math.max(14, dist * 0.2) : Math.max(speed, 2);
      if (dist > 120 && this.phase === "companion") this.bird = [...target];
      const k = Math.min(1, (v * dt) / Math.max(dist, 1e-4));
      this.birdVel = [(dx * k) / Math.max(dt, 1e-4), (dy * k) / Math.max(dt, 1e-4), (dz * k) / Math.max(dt, 1e-4)];
      this.bird = [this.bird[0] + dx * k, this.bird[1] + dy * k, this.bird[2] + dz * k];
    } else this.birdVel = [0, 0, 0];
    const flying = !perched;
    this.flapPhase += dt * 32;
    const yaw = flying && Math.hypot(this.birdVel[0], this.birdVel[2]) > 0.1 ? Math.atan2(this.birdVel[0], this.birdVel[2]) : this.gates[0].yaw;
    const pitch = flying ? -Math.atan2(this.birdVel[1], Math.hypot(this.birdVel[0], this.birdVel[2]) + 1e-3) * 0.6 : Math.max(0, Math.sin(performance.now() / 700)) * 0.25;
    // It glows (and casts its light) always: easy to spot waiting, racing or beside the wind.
    this.birdData.set([this.bird[0], this.bird[1] + (flying ? 0.06 : 0), this.bird[2], yaw, flying ? Math.sin(this.flapPhase) * 0.9 + 0.1 : 0, flying ? 1 : 0, pitch, 1.9]);
    this.birdBuf.write(this.birdData);
    this.flying = flying;
    // Gates.
    this.gates.forEach((g, i) => {
      g.flash = Math.max(0, g.flash - dt * 1.5);
      this.gateData.set([g.base[0], g.base[1], g.base[2], g.yaw, g.state, g.flash, 0, 0], i * 8);
    });
    this.gateBuf.write(this.gateData);
  }

  private flying = false;
  private companionLeft = 0;
  private lastWind: Vec3 | null = null;
  private heading: [number, number] = [1, 0];
  /** Seconds into the victory wave (-1 when none). */
  private wave = -1;

  /** Where the kingfisher waits (for a glint), or null once it has joined the wind. */
  get waitingAt(): Vec3 | null {
    return this.phase === "waiting" ? this.perch : null;
  }

  get racing(): boolean {
    return this.phase === "racing";
  }

  encode(pass: FramePass): void {
    pass.draw(this.gateDraw, { instances: this.gates.length });
    const d = this.birdDraws[this.flying ? 1 : 0];
    if (d) pass.draw(d);
    pass.draw(this.haloDraw);
  }
}
