import { draw, geometry, storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import shader from "../shaders/fish.wgsl";
import type { Vec3 } from "../engine/camera";
import { riverCenter, riverHalfWidth, riverInfo, riverWater, terrainHeightM as terrainHeight } from "./height";

const COUNT = 16;
/** Fish live within this distance (along the river) of the camera; others swim in. */
const RANGE = 35;

interface Fish {
  pos: Vec3;
  heading: number;
  pitch: number;
  length: number;
  phase: number;
  beat: number;
  speed: number;
  /** Seconds until the next leap. */
  jumpIn: number;
  /** Airborne: velocity of the leap. */
  leap: Vec3 | null;
  depth: number;
  drift: number;
}

function rand(a: number, b: number): number {
  return a + Math.random() * (b - a);
}

/**
 * River fish: trout holding in the current, nosing upstream and drifting; now and then one
 * near the camera leaps clear of the surface in an arc and drops back with a splash.
 */
export class RiverFish {
  readonly draw: Draw;
  private readonly fish: Fish[] = [];
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(COUNT * 8);
  private count = 0;
  /** Splash events (for spray and sound): where, and how big (0..1). */
  onSplash: ((at: Vec3, size: number) => void) | null = null;

  private constructor(d: Draw, buffer: StorageBuffer) {
    this.draw = d;
    this.buffer = buffer;
  }

  static async load(gpu: Gpu, globals: SharedUniforms, base = "assets/fish"): Promise<RiverFish> {
    const [manifest, bin] = await Promise.all([
      fetch(`${base}/fish.json`).then((r) => r.json() as Promise<{ vertexBytes: number; indexCount: number }>),
      fetch(`${base}/fish.bin`).then((r) => r.arrayBuffer()),
    ]);
    const geo = geometry(gpu, {
      label: "fish",
      buffers: [{ data: new Uint8Array(bin, 0, manifest.vertexBytes), attributes: { p: "float16x4", n: "snorm8x4", t: "float16x2", e: "unorm8x4" } }],
      indices: new Uint32Array(bin, manifest.vertexBytes, manifest.indexCount),
    });
    const buffer = storage(gpu, COUNT * 32, "read");
    const d = draw(gpu, { label: "fish", shader, geometry: geo, cull: "none", depth: { compare: "greater" }, set: { G: globals, fish: buffer } });
    return new RiverFish(d, buffer);
  }

  /** A spot in deep enough water near `x`, or null. */
  private spot(x: number): Vec3 | null {
    for (let k = 0; k < 6; k++) {
      const px = x + rand(-RANGE, RANGE);
      const hw = riverHalfWidth(px);
      const z = riverCenter(px) + rand(-0.55, 0.55) * hw;
      const water = riverWater(px);
      const bed = terrainHeight(px, z);
      if (water - bed > 1.0) return [px, rand(bed + 0.35, water - 0.4), z];
    }
    return null;
  }

  update(dt: number, cam: readonly number[], t: number): void {
    const [cd] = riverInfo(cam[0], cam[2]);
    const near = cd < 140;
    // Keep a school around the camera when the river is near.
    if (near) {
      while (this.fish.length < COUNT) {
        const p = this.spot(cam[0]);
        if (!p) break;
        const length = rand(0.38, 0.62);
        this.fish.push({ pos: p, heading: 0, pitch: 0, length, phase: rand(0, 6), beat: 1, speed: rand(0.4, 1.2), jumpIn: rand(4, 25), leap: null, depth: 0, drift: rand(0, 100) });
      }
    }
    let o = 0;
    for (let i = this.fish.length - 1; i >= 0; i--) {
      const f = this.fish[i];
      if (Math.abs(f.pos[0] - cam[0]) > RANGE * 1.8 || !near) {
        this.fish.splice(i, 1);
        continue;
      }
      const x = f.pos[0];
      const water = riverWater(x);
      if (f.leap) {
        // Airborne: a ballistic arc, nose following the velocity, body curling.
        f.leap[1] -= 9.8 * dt;
        for (let j = 0; j < 3; j++) f.pos[j] += f.leap[j] * dt;
        f.pitch = Math.atan2(f.leap[1], Math.hypot(f.leap[0], f.leap[2]));
        f.phase += dt * 18;
        f.beat = 1.8;
        if (f.pos[1] < water && f.leap[1] < 0) {
          this.onSplash?.([f.pos[0], water, f.pos[2]], f.length * 1.6);
          f.pos[1] = water - 0.3;
          f.leap = null;
          f.jumpIn = rand(12, 40);
        }
      } else {
        // Holding in the current: nose upstream (-x is upstream here when the water falls
        // toward +x), drifting slowly and sliding across the channel.
        const downhill = riverWater(x + 4) < riverWater(x - 4) ? 1 : -1;
        f.drift += dt * 0.3;
        const along = Math.sin(f.drift + i) * f.speed;
        const cz = riverCenter(x);
        const hw = riverHalfWidth(x);
        const across = Math.sin(f.drift * 0.7 + i * 2.1) * hw * 0.45;
        const wantZ = cz + across;
        f.pos[0] += along * dt;
        f.pos[2] += (wantZ - f.pos[2]) * Math.min(1, dt * 0.4);
        const bed = terrainHeight(f.pos[0], f.pos[2]);
        const wantY = Math.min(water - 0.35, Math.max(bed + 0.3, water - 0.6 - (Math.sin(f.drift * 0.5 + i) * 0.5 + 0.5) * (water - bed - 0.9)));
        f.pos[1] += (wantY - f.pos[1]) * Math.min(1, dt * 0.6);
        const dz = (wantZ - f.pos[2]) * 0.4;
        f.heading = Math.atan2(-downhill * 1 + along * 0.2, dz * 0.5);
        f.pitch *= Math.exp(-dt * 3);
        f.beat = 0.6 + Math.abs(along) * 0.6;
        f.phase += dt * (5 + Math.abs(along) * 4);
        // Leap: only near the camera, where it can be seen.
        f.jumpIn -= dt;
        const dCam = Math.hypot(f.pos[0] - cam[0], f.pos[2] - cam[2]);
        if (f.jumpIn <= 0 && dCam < 60 && t > 3) {
          const dir = Math.random() < 0.5 ? 1 : -1;
          const h = rand(-0.4, 0.4);
          f.pos[1] = water - 0.05;
          f.leap = [Math.cos(h) * dir * rand(2, 3.5), rand(3.6, 5), Math.sin(h) * rand(1, 2)];
          f.heading = Math.atan2(f.leap[0], f.leap[2]);
          this.onSplash?.([f.pos[0], water, f.pos[2]], f.length);
        } else if (f.jumpIn <= 0) f.jumpIn = rand(4, 12);
      }
      if (f.leap) f.heading = Math.atan2(f.leap[0], f.leap[2]);
      this.data.set([f.pos[0], f.pos[1], f.pos[2], f.heading, f.pitch, f.length, f.phase, f.beat], o);
      o += 8;
    }
    this.count = o / 8;
    if (o) this.buffer.write(this.data.subarray(0, o));
  }

  encode(pass: FramePass): void {
    if (this.count) pass.draw(this.draw, { instances: this.count });
  }
}
