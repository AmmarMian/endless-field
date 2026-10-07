import { draw, geometry, type Draw, type FramePass, type Gpu, type SharedUniforms } from "vgpu";
import shader from "../shaders/nest.wgsl";
import type { Vec3 } from "../engine/camera";
import { terrainHeightM as terrainHeight } from "./height";
import { toriiGates } from "./torii";

const KEY = "endless-field-nest";
/** Insects caught per chick that hatches (up to four). */
const PER_CHICK = 12;

/** A deterministic little noise for the mud pellets. */
function bump(a: number, b: number): number {
  return Math.sin(a * 12.9898 + b * 78.233) * 43758.5453 % 1;
}

/**
 * Builds the nest mesh (model space, rim centre at the origin, cup below): a half cup of mud
 * in rows of pellets, a straw-flecked rim, and four chick heads (round, fuzzy-dark, with a
 * wide yellow gape and two eyes) that show as the brood grows.
 */
function nestMesh(): { verts: Float32Array<ArrayBuffer>; indices: Uint32Array<ArrayBuffer> } {
  const v: number[] = [];
  const idx: number[] = [];
  const vert = (p: number[], n: number[], part: number, chick = 0) => {
    v.push(p[0], p[1], p[2], part, n[0], n[1], n[2], 0, chick, 0, 0, 0);
    return v.length / 12 - 1;
  };
  // Cup: rings from the rim down; outer and inner surfaces, bumpy like stacked mud pellets.
  const rings = 9;
  const seg = 28;
  const R = 0.11;
  const depth = 0.08;
  for (const inner of [false, true]) {
    const base = v.length / 12;
    for (let i = 0; i <= rings; i++) {
      const u = i / rings;
      for (let k = 0; k < seg; k++) {
        const a = (k / seg) * Math.PI * 2;
        const pellet = 1 + 0.08 * Math.abs(Math.sin(a * 9 + i * 1.7)) * Math.abs(Math.sin(u * 14 + a * 3)) + 0.03 * bump(i, k);
        const r = R * Math.cos(u * Math.PI * 0.5) * (inner ? 0.86 : pellet);
        const y = -depth * Math.sin(u * Math.PI * 0.5) * (inner ? 0.85 : 1) + (inner ? -0.004 : 0);
        const nx = Math.cos(a) * Math.cos(u * 1.4);
        const ny = -Math.sin(u * 1.4);
        const nz = Math.sin(a) * Math.cos(u * 1.4);
        vert([Math.cos(a) * r, y, Math.sin(a) * r], inner ? [-nx, -ny, -nz] : [nx, ny, nz], 0);
      }
    }
    for (let i = 0; i < rings; i++) {
      for (let k = 0; k < seg; k++) {
        const a = base + i * seg + k;
        const b = base + i * seg + ((k + 1) % seg);
        const c = base + (i + 1) * seg + ((k + 1) % seg);
        const d = base + (i + 1) * seg + k;
        if (inner) idx.push(a, c, b, a, d, c);
        else idx.push(a, b, c, a, c, d);
      }
    }
  }
  // Straw: short blades tucked into the rim, sticking out at angles.
  for (let k = 0; k < 18; k++) {
    const a = (k / 18) * Math.PI * 2 + bump(k, 3);
    const len = 0.05 + 0.04 * Math.abs(bump(k, 7));
    const p0 = [Math.cos(a) * R * 0.95, -0.005, Math.sin(a) * R * 0.95];
    const dir = [Math.cos(a + 0.6), 0.25 + 0.3 * bump(k, 9), Math.sin(a + 0.6)];
    const side = [-Math.sin(a + 0.6) * 0.003, 0, Math.cos(a + 0.6) * 0.003];
    const i0 = vert([p0[0] - side[0], p0[1], p0[2] - side[2]], [0, 1, 0], 4);
    const i1 = vert([p0[0] + side[0], p0[1], p0[2] + side[2]], [0, 1, 0], 4);
    const i2 = vert([p0[0] + dir[0] * len, p0[1] + dir[1] * len, p0[2] + dir[2] * len], [0, 1, 0], 4);
    idx.push(i0, i1, i2);
  }
  // Chick heads (with gape and eyes) across the cup.
  const sphere = (cx: number, cy: number, cz: number, r: number, part: number, chick: number, sx = 1, sy = 1, sz = 1) => {
    const base = v.length / 12;
    const la = 8;
    const lo = 12;
    for (let i = 0; i <= la; i++) {
      const th = (i / la) * Math.PI;
      for (let k = 0; k <= lo; k++) {
        const ph = (k / lo) * Math.PI * 2;
        const n = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
        vert([cx + n[0] * r * sx, cy + n[1] * r * sy, cz + n[2] * r * sz], n, part, chick);
      }
    }
    for (let i = 0; i < la; i++) {
      for (let k = 0; k < lo; k++) {
        const a = base + i * (lo + 1) + k;
        const b = a + lo + 1;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
  };
  for (let c = 0; c < 4; c++) {
    const x = (c - 1.5) * 0.045;
    const z = 0.01 * ((c % 2) * 2 - 1);
    sphere(x, 0.026, z, 0.024, 1, c);
    // The gape: a flattened yellow wedge at the front, and two eyes.
    sphere(x, 0.024, z + 0.024, 0.013, 2, c, 1.3, 0.55, 0.9);
    sphere(x - 0.011, 0.034, z + 0.016, 0.0045, 3, c);
    sphere(x + 0.011, 0.034, z + 0.016, 0.0045, 3, c);
  }
  return { verts: new Float32Array(v), indices: new Uint32Array(idx) };
}

/**
 * The swallows' nest, under the beam of the first torii. Insects the swallow catches feed the
 * brood: chicks hatch as it grows (kept across sessions). At dusk or night the swallow can rest
 * here while the night passes.
 */
export class Nest {
  readonly draw: Draw;
  /** Where the parent perches (on the rim). */
  readonly perch: Vec3;
  readonly pos: Vec3;
  private fed: number;
  private readonly yaw: number;

  constructor(gpu: Gpu, globals: SharedUniforms) {
    const g = toriiGates()[0];
    // Tucked under the top beam, just outside one pillar (as swallows nest under eaves).
    const local: Vec3 = [0, 5.46, 2.95];
    const cs = [g.cos, g.sin];
    const ground = terrainHeight(g.x, g.z) - 0.05;
    this.pos = [g.x + cs[0] * local[0] + cs[1] * local[2], ground + local[1], g.z - cs[1] * local[0] + cs[0] * local[2]];
    this.yaw = Math.atan2(g.sin, g.cos);
    this.perch = [this.pos[0], this.pos[1] + 0.08, this.pos[2]];
    try {
      this.fed = Number(localStorage.getItem(KEY) ?? "0") || 0;
    } catch {
      this.fed = 0;
    }
    const m = nestMesh();
    this.draw = draw(gpu, {
      label: "nest",
      shader,
      geometry: geometry(gpu, { buffers: [{ data: m.verts, attributes: { p: "float32x4", n: "float32x4", c: "float32x4" } }], indices: m.indices }),
      cull: "none",
      depth: { compare: "greater" },
      set: { G: globals, N: { pos: [...this.pos, this.yaw], info: [this.chicks, 0, 2.8, 0] } },
    });
  }

  get chicks(): number {
    return Math.min(4, Math.floor(this.fed / PER_CHICK));
  }

  /** An insect caught: the brood is fed (a chick hatches every few). */
  feed(): void {
    this.fed++;
    try {
      localStorage.setItem(KEY, String(this.fed));
    } catch {
      // Non-essential.
    }
  }

  update(parent: Vec3): void {
    const near = Math.hypot(parent[0] - this.pos[0], parent[1] - this.pos[1], parent[2] - this.pos[2]);
    const beg = Math.max(0, Math.min(1, (6 - near) / 4));
    this.draw.set({ N: { info: [this.chicks, beg, 2.8, 0] } });
  }

  encode(pass: FramePass): void {
    pass.draw(this.draw);
  }
}
