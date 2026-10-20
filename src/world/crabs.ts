import { storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import crabShader from "../shaders/crabs.wgsl";
import { BIRD_STRIDE, BirdAnimator, BirdSpecies } from "./bird-model";
import { SEA_Y, coastInfo, duneAmount, terrainHeightM } from "./height";
import type { Vec3 } from "../engine/camera";

const MAX = 64;
/** Crabs live within this distance of the camera (they come out of and go back into the sand). */
const RANGE = 55;

type State = "hidden" | "emerge" | "feed" | "walk" | "flee" | "threat" | "burrow";

interface Crab {
  pos: Vec3;
  heading: number;
  size: number;
  seed: number;
  state: State;
  /** Seconds left in this state (feeding, hiding...). */
  timer: number;
  /** Sideways walking: +1 to its right, -1 to its left; and how far to go. */
  dir: number;
  togo: number;
  speed: number;
  anim: BirdAnimator;
}

/** A spot crabs like: damp sand from the swash up to the strand line, and around rocks. */
function goodSand(x: number, z: number): boolean {
  const [s] = coastInfo(x, z);
  if (s < -1 || s > 40) return false;
  const above = terrainHeightM(x, z) - SEA_Y;
  return above > 0.05 && above < 1.6;
}

/**
 * Shore crabs on the beach. They live their own lives: come up out of the sand, pick at it,
 * scuttle sideways along the shore, and if something rushes at them they dash off sideways and
 * dig themselves back in (or, cornered by a slow approach, rear up with claws open). They never
 * follow anyone; a while after being left in peace they come up again.
 */
export class Crabs {
  private readonly crabs: Crab[] = [];
  private readonly data = new Float32Array(MAX * BIRD_STRIDE);
  private count = 0;
  private spawnTimer = 0;

  private constructor(
    private readonly species: BirdSpecies,
    private readonly buffer: StorageBuffer,
    readonly draw: Draw,
  ) {}

  /** A crab sprinted away near the camera (for a little scuttle sound): position. */
  onScuttle?: (at: Vec3) => void;

  static async load(gpu: Gpu, globals: SharedUniforms): Promise<Crabs> {
    const species = await BirdSpecies.load(gpu, "crab", "assets/crabs");
    species.shader = crabShader;
    species.instanceKey = "crabs";
    const buffer = storage(gpu, MAX * BIRD_STRIDE * 4, "read");
    const { body } = species.draws(globals, buffer, "crabs");
    return new Crabs(species, buffer, body);
  }

  private spawn(cam: Vec3): void {
    for (let k = 0; k < 12; k++) {
      const a = Math.random() * Math.PI * 2;
      const r = 6 + Math.random() * (RANGE - 10);
      const x = cam[0] + Math.cos(a) * r;
      const z = cam[2] + Math.sin(a) * r;
      if (!goodSand(x, z)) continue;
      // More on the rocky stretches (shelter), fewer on open dune beaches.
      if (Math.random() < duneAmount(x, z) * 0.5) continue;
      const anim = new BirdAnimator(this.species.clips, "emerge");
      this.crabs.push({
        pos: [x, terrainHeightM(x, z), z],
        heading: Math.random() * Math.PI * 2,
        size: 0.55 + Math.random() * 0.6,
        seed: Math.random(),
        state: "emerge",
        timer: 0,
        dir: 1,
        togo: 0,
        speed: 0,
        anim,
      });
      return;
    }
  }

  update(dt: number, cam: Vec3, threat: Vec3, threatSpeed: number): void {
    // Keep a small population around the camera while it is near the shore.
    const [sc] = coastInfo(cam[0], cam[2]);
    const want = sc < 160 && cam[1] - SEA_Y < 60 ? MAX * 0.6 : 0;
    this.spawnTimer -= dt;
    if (this.crabs.length < want && this.spawnTimer <= 0) {
      this.spawnTimer = 0.15;
      this.spawn(cam);
    }
    for (let i = this.crabs.length - 1; i >= 0; i--) {
      const c = this.crabs[i];
      const far = Math.hypot(c.pos[0] - cam[0], c.pos[2] - cam[2]) > RANGE + 10;
      if (far && (c.state === "hidden" || want === 0)) {
        this.crabs.splice(i, 1);
        continue;
      }
      this.think(c, dt, threat, threatSpeed);
      c.anim.update(dt);
    }
    this.count = 0;
    for (const c of this.crabs) {
      if (c.state === "hidden" || this.count >= MAX) continue;
      const o = this.count * BIRD_STRIDE;
      this.data[o] = c.pos[0];
      this.data[o + 1] = c.pos[1];
      this.data[o + 2] = c.pos[2];
      this.data[o + 3] = c.heading;
      this.data[o + 4] = 0;
      this.data[o + 5] = 0;
      this.data[o + 6] = c.size;
      this.data[o + 7] = 0;
      c.anim.write(this.data, o + 8);
      this.data[o + 11] = c.seed;
      this.count++;
    }
    if (this.count) this.buffer.write(this.data.subarray(0, this.count * BIRD_STRIDE));
  }

  private think(c: Crab, dt: number, threat: Vec3, threatSpeed: number): void {
    const dx = c.pos[0] - threat[0];
    const dz = c.pos[2] - threat[2];
    const near = Math.hypot(dx, dz);
    const low = threat[1] - c.pos[1] < 4;
    // Its own right (it walks sideways): (cos h, -sin h).
    const rx = Math.cos(c.heading);
    const rz = -Math.sin(c.heading);
    const alarmed = low && (near < 2.5 || (near < 7 && threatSpeed > 3));
    c.timer -= dt;
    switch (c.state) {
      case "hidden":
        if (c.timer <= 0 && near > 8) {
          c.state = "emerge";
          c.anim.play("emerge", 0, 1, true);
        }
        return;
      case "emerge":
        if (c.anim.done) this.feed(c);
        return;
      case "burrow":
        if (c.anim.done) {
          c.state = "hidden";
          c.timer = 8 + Math.random() * 20;
        }
        return;
    }
    if (alarmed && c.state !== "flee") {
      if (threatSpeed < 1.5 && near < 2.5 && c.state !== "threat") {
        // Cornered by something slow: stand its ground, claws up.
        c.state = "threat";
        c.timer = 1.5 + Math.random();
        c.anim.play("threat", 0.12, 1);
        return;
      }
      if (c.state !== "threat" || threatSpeed > 1.5) {
        // Bolt sideways, away.
        c.state = "flee";
        c.dir = rx * dx + rz * dz > 0 ? 1 : -1;
        c.togo = 1.2 + Math.random() * 1.5;
        c.speed = 0.9 + Math.random() * 0.4;
        c.anim.play("run", 0.08, 1);
        this.onScuttle?.(c.pos);
      }
    }
    if (c.state === "threat") {
      // Keep facing the threat.
      c.heading = Math.atan2(-dx, -dz);
      if (c.timer <= 0) this.feed(c);
      return;
    }
    if (c.state === "feed") {
      if (c.timer <= 0) {
        // Off along the shore, sideways, a few body lengths at a time.
        c.state = "walk";
        c.dir = Math.random() < 0.5 ? 1 : -1;
        c.togo = 0.3 + Math.random() * 1.6;
        c.speed = 0.12 + Math.random() * 0.1;
        c.anim.play("walk", 0.2, 1);
      }
      return;
    }
    // Walking or fleeing: sideways along its own x axis, turning a little as it goes.
    const step = c.speed * c.size * dt * (c.state === "flee" ? 3.2 : 1);
    const nx = c.pos[0] + rx * c.dir * step;
    const nz = c.pos[2] + rz * c.dir * step;
    if (c.state === "walk" && !goodSand(nx, nz)) {
      this.feed(c);
      return;
    }
    c.pos[0] = nx;
    c.pos[2] = nz;
    c.pos[1] = terrainHeightM(nx, nz);
    c.heading += (Math.sin(c.seed * 50 + c.togo * 3) * 0.6) * dt;
    c.togo -= step;
    // Legs keep pace with the ground covered (stride ~3 cm at life size).
    c.anim.speed = c.state === "flee" ? (c.speed * 3.2) / 0.9 : c.speed / 0.12;
    if (c.togo <= 0) {
      if (c.state === "flee") {
        c.state = "burrow";
        c.anim.play("burrow", 0.1, 1.3, true);
      } else this.feed(c);
    }
  }

  private feed(c: Crab): void {
    c.state = "feed";
    c.timer = 2 + Math.random() * 7;
    c.anim.play("idle", 0.25, 0.8 + Math.random() * 0.4, false, Math.random() * 2);
  }

  encode(pass: FramePass): void {
    if (this.count) pass.draw(this.draw, { instances: this.count });
  }
}
