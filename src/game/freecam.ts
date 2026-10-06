import type { Camera, Vec3 } from "../engine/camera";
import { riverInfo, terrainHeightM as terrainHeight } from "../world/height";

const EYE = 1.65;
const WALK = 6;
const FLY = 30;
/** Highest a flying explorer may go above the ground (meters). */
const MAX_FLY = 10;
const TRAIL_LEN = 24;

/**
 * Free-roam explorer: pointer-lock mouse look, WASD to move, walk at eye height or fly.
 * Leaves a short trail so the grass parts around you as you walk through it.
 */
export class FreeCam {
  readonly pos: Vec3 = [0, 0, 0];
  yaw = 0;
  pitch = 0;
  fly = false;
  private bob = 0;
  private ceilingGround = 0;
  private readonly keys = new Set<string>();
  private readonly trail: { p: Vec3; age: number }[] = [];
  private active = false;

  constructor(private readonly canvas: HTMLCanvasElement) {
    window.addEventListener("keydown", (e) => {
      if (!this.active) return;
      this.keys.add(e.code);
      if (!e.repeat && e.key.toLowerCase() === "v") this.fly = !this.fly;
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.code));
    window.addEventListener("blur", () => this.keys.clear());
    window.addEventListener("mousemove", (e) => {
      if (!this.active || document.pointerLockElement !== this.canvas) return;
      this.yaw += e.movementX * 0.0022;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch - e.movementY * 0.0022));
    });
    canvas.addEventListener("click", () => {
      if (this.active && document.pointerLockElement !== canvas) void canvas.requestPointerLock();
    });
  }

  /** Starts exploring from the given camera pose. */
  enter(from: Camera): void {
    this.active = true;
    this.pos.splice(0, 3, ...from.position);
    const d = from.target.map((v, i) => v - from.position[i]);
    this.yaw = Math.atan2(d[0], -d[2]);
    this.pitch = Math.atan2(d[1], Math.hypot(d[0], d[2]));
    const above = this.pos[1] - terrainHeight(this.pos[0], this.pos[2]);
    this.fly = above > 4;
    if (above > MAX_FLY) this.pos[1] -= above - MAX_FLY;
    this.ceilingGround = terrainHeight(this.pos[0], this.pos[2]);
    void this.canvas.requestPointerLock?.();
  }

  exit(): void {
    this.active = false;
    this.keys.clear();
    if (document.pointerLockElement === this.canvas) document.exitPointerLock();
  }

  private ground(x: number, z: number): number {
    // Wade the shallows: never sink below the river surface.
    const [, water] = riverInfo(x, z);
    return Math.max(terrainHeight(x, z), water - 0.35);
  }

  update(dt: number, obstacles: { x: number; z: number; r: number }[]): void {
    const k = this.keys;
    const fwd = (k.has("KeyW") ? 1 : 0) - (k.has("KeyS") ? 1 : 0);
    const strafe = (k.has("KeyD") ? 1 : 0) - (k.has("KeyA") ? 1 : 0);
    // Arrows look around, so the whole mode works from the keyboard (no pointer lock needed).
    const turn = (k.has("ArrowRight") ? 1 : 0) - (k.has("ArrowLeft") ? 1 : 0);
    const look = (k.has("ArrowUp") ? 1 : 0) - (k.has("ArrowDown") ? 1 : 0);
    this.yaw += turn * 1.8 * dt;
    this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + look * 1.2 * dt));
    const sprinting = k.has("ShiftLeft") || k.has("ShiftRight");
    const sprint = sprinting ? (this.fly ? 4 : 3.5) : 1;
    const speed = (this.fly ? FLY : WALK) * sprint;
    const sy = Math.sin(this.yaw);
    const cy = Math.cos(this.yaw);
    let vx = (sy * fwd + cy * strafe) * speed;
    let vz = (-cy * fwd + sy * strafe) * speed;
    let vy = 0;
    if (this.fly) {
      // Flying follows the look direction; Space / C move straight up and down.
      vy = Math.sin(this.pitch) * fwd * speed;
      const cp = Math.cos(this.pitch);
      vx = (sy * fwd * cp + cy * strafe) * speed;
      vz = (-cy * fwd * cp + sy * strafe) * speed;
      if (k.has("Space")) vy += speed * 0.7;
      if (k.has("KeyC") || k.has("ControlLeft") || k.has("ControlRight")) vy -= speed * 0.7;
    }
    this.pos[0] += vx * dt;
    this.pos[2] += vz * dt;
    for (const o of obstacles) {
      const dx = this.pos[0] - o.x;
      const dz = this.pos[2] - o.z;
      const d = Math.hypot(dx, dz);
      if (d < o.r) {
        this.pos[0] = o.x + (dx / (d || 1)) * o.r;
        this.pos[2] = o.z + (dz / (d || 1)) * o.r;
      }
    }
    const g = this.ground(this.pos[0], this.pos[2]);
    const moving = Math.hypot(vx, vz) > 0.1;
    if (this.fly) {
      // Stay close to the ground: the world is built to be seen from within it. The ceiling
      // follows a smoothed ground and is approached softly, so it never judders.
      this.ceilingGround += (g - this.ceilingGround) * Math.min(1, dt * 1.2);
      const top = this.ceilingGround + MAX_FLY;
      let y = this.pos[1] + vy * dt;
      if (y > top) y += (top - y) * Math.min(1, dt * 3);
      this.pos[1] = Math.max(y, g + 0.8);
    } else {
      this.bob += dt * (moving ? 9 * Math.sqrt(sprint) : 0);
      const target = g + EYE + (moving ? Math.sin(this.bob) * 0.035 : 0);
      this.pos[1] += (target - this.pos[1]) * Math.min(1, dt * 12);
    }

    for (const t of this.trail) t.age += dt;
    const head = this.trail[0];
    if (!this.fly && (!head || Math.hypot(head.p[0] - this.pos[0], head.p[2] - this.pos[2]) > 0.5)) {
      this.trail.unshift({ p: [this.pos[0], g + 0.3, this.pos[2]], age: 0 });
      if (this.trail.length > TRAIL_LEN) this.trail.length = TRAIL_LEN;
    }
  }

  /** Footsteps part the grass gently (smaller and weaker than the wind's stream). */
  writeTrail(out: Float32Array): void {
    out.fill(0);
    this.trail.forEach((t, i) => {
      const k = Math.max(0, 1 - t.age / 2.5);
      out.set([t.p[0], t.p[1], t.p[2], k * k * (3 - 2 * k) * 0.45], i * 4);
    });
  }

  apply(camera: Camera): void {
    const cp = Math.cos(this.pitch);
    camera.position.splice(0, 3, ...this.pos);
    camera.target.splice(0, 3, this.pos[0] + Math.sin(this.yaw) * cp, this.pos[1] + Math.sin(this.pitch), this.pos[2] - Math.cos(this.yaw) * cp);
    camera.fovY = (65 * Math.PI) / 180;
  }
}
