import { storage, type Draw, type FramePass, type Gpu, type SharedUniforms, type StorageBuffer } from "vgpu";
import { BIRD_STRIDE, BirdAnimator, BirdSpecies } from "../world/bird-model";
import type { Player } from "./player";

/** Times life size: big enough to read from the chase camera, still a swallow. */
const SIZE = 2.8;

/**
 * Playing as a bird: a barn swallow (tools/blender/model_birds.py) flies where the wind goes.
 * Its motion is the clips animated in Blender: bursts of wingbeats and long glides, beating
 * hard when gusting or climbing, wings tucked in a stoop. Left alone it comes in to land (a
 * flare with braking beats, feet forward, touchdown, wings folding), then rests: breathing,
 * looking about, now and then preening. Touched, it springs back into the air.
 */
export class PlayerBird {
  readonly draw: Draw;
  readonly haloDraw: Draw;
  /** Its own soft light (0 none .. 1 full), brighter as it gets dark. */
  glow = 0;
  /** On its side with a wingtip in the water (0..1), held while it skims low. */
  dip = 0;
  private dipSide = 1;
  /** Where the lowered wingtip touches the water (valid while dipping). */
  readonly tip: [number, number, number] = [0, 0, 0];
  /** From the soles to the body's centre when perched (m): perches are given at the feet. */
  readonly perchLift: number;
  private readonly buffer: StorageBuffer;
  private readonly data = new Float32Array(BIRD_STRIDE);
  private readonly anim: BirdAnimator;
  private roll = 0;
  private lastYaw: number | null = null;
  private yawRate = 0;
  private cycle = 0;
  private wasPerched = false;
  private wasLanding = false;
  private restTimer = 6;
  private lift = 0;
  visible = false;

  private constructor(d: Draw, halo: Draw, buffer: StorageBuffer, species: BirdSpecies) {
    this.draw = d;
    this.haloDraw = halo;
    this.buffer = buffer;
    this.anim = new BirdAnimator(species.clips, "glide");
    this.perchLift = species.perchHeight * SIZE;
  }

  static async load(gpu: Gpu, globals: SharedUniforms): Promise<PlayerBird> {
    const species = await BirdSpecies.load(gpu, "swallow");
    const buffer = storage(gpu, BIRD_STRIDE * 4, "read");
    const { body, halo } = species.draws(globals, buffer, "swallow");
    return new PlayerBird(body, halo, buffer, species);
  }

  /** `water`: the river's surface under the bird when over it, else null. */
  update(dt: number, player: Player, water: number | null = null): void {
    // Turn rate -> bank: lean into the turn, eased.
    if (this.lastYaw !== null && dt > 0) {
      let d = player.yaw - this.lastYaw;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      this.yawRate += (d / dt - this.yawRate) * Math.min(1, dt * 4);
    }
    this.lastYaw = player.yaw;
    const looping = player.looping;
    const perched = player.perched;
    const landing = player.landing !== null;
    // Low over the river: roll onto its side and slice the water with a wingtip, held as
    // long as it stays low over the water (toward the way it is turning).
    const skimming = water !== null && !perched && !landing && player.pos[1] - water < 2.1 && !looping && !player.riding;
    this.dip += ((skimming ? 1 : 0) - this.dip) * Math.min(1, dt * (skimming ? 2.5 : 4));
    if (Math.abs(this.yawRate) > 0.25) this.dipSide = this.yawRate > 0 ? 1 : -1;
    const turnRoll = Math.max(-0.85, Math.min(0.85, this.yawRate * 0.5));
    const wantRoll = looping || perched || landing ? 0 : turnRoll + (this.dipSide * 1.2 - turnRoll) * this.dip;
    this.roll += (wantRoll - this.roll) * Math.min(1, dt * 3);

    const a = this.anim;
    if (perched) {
      // Resting: breathe and look about; preen now and then.
      if (!this.wasPerched && a.clip !== "land") a.play("perch", 0.3);
      if (a.clip === "land" && a.done) a.play("perch", 0.35);
      this.restTimer -= dt;
      if (a.clip === "perch" && this.restTimer <= 0) {
        a.play("preen", 0.3);
        this.restTimer = 12 + Math.random() * 14;
      }
      if (a.clip === "preen" && a.done) a.play("perch", 0.4, 1, true, Math.random() * a.duration("perch"));
    } else if (landing) {
      // Coming in: glide, then the landing clip timed so its touchdown meets the perch.
      const touchdown = 0.9;
      if (a.clip !== "land" && player.landingLeft <= touchdown) a.play("land", 0.2, 1, true, Math.max(0, touchdown - player.landingLeft));
      else if (a.clip !== "land") a.play("glide", 0.5);
    } else if (this.wasPerched) {
      // Springing up into the air.
      a.play("takeoff", 0.12, 1.2, true);
      this.cycle = 0;
    } else if (this.wasLanding && a.clip === "land") {
      // Waved off before touching down: beat back up.
      a.play("flap", 0.3, 2.4);
    } else if (a.clip === "takeoff" && !a.done) {
      // Let the take-off finish.
    } else {
      // Beating or gliding: hard work when gusting, climbing or pulling a loop; otherwise
      // a few easy beats then a long glide. Wings tucked in a stoop, still in a thermal.
      const soaring = player.thermal > 0.25 || player.riding !== null;
      const working = (player.gust > 0.25 || player.pitch > 0.12 || (looping && player.pitch < 1.4)) && player.stoop < 0.5 && !soaring;
      this.cycle = (this.cycle + dt) % 4.6;
      const beating = working || (this.cycle < 2.0 && !soaring);
      if (player.stoop > 0.5) a.play("tuck", 0.25);
      else if (this.dip > 0.3) a.play("glide", 0.4);
      else if (beating) a.play("flap", a.clip === "takeoff" ? 0.15 : 0.3, working ? 2.4 : 1.7);
      else a.play("glide", 0.45);
    }
    this.wasPerched = perched;
    this.wasLanding = landing;
    a.update(dt);

    const p = player.pos;
    // Dipping: hold the body just high enough that the lowered tip cuts the surface.
    let bodyY = p[1];
    if (water !== null && this.dip > 0.001) bodyY = p[1] + (Math.max(water + 0.38, p[1] - 1.2) - p[1]) * this.dip;
    const halfSpan = 0.47;
    this.tip[0] = p[0] + this.dipSide * Math.cos(player.yaw) * halfSpan * Math.cos(1.2);
    this.tip[1] = water ?? bodyY;
    this.tip[2] = p[2] + this.dipSide * Math.sin(player.yaw) * halfSpan * Math.cos(1.2);
    // The landing and perching clips pitch the body themselves; in flight it follows the climb.
    const settled = perched || landing;
    this.lift += ((settled ? 0 : 1) - this.lift) * Math.min(1, dt * 4);
    const pitch = (looping ? -player.pitch : -player.pitch * 0.85) * this.lift;
    // The shader's heading maps +Z to (sin, cos); the player's forward is (sin yaw, -cos yaw).
    this.data.set([p[0], bodyY, p[2], Math.PI - player.yaw, pitch, this.roll + player.rollAngle * this.lift, SIZE, this.glow * 0.6]);
    a.write(this.data, 8);
    this.data[11] = 0.5;
    this.buffer.write(this.data);
  }

  encode(pass: FramePass): void {
    if (this.visible) {
      pass.draw(this.draw);
      if (this.glow > 0.02) pass.draw(this.haloDraw);
    }
  }
}
