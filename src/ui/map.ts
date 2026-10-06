import { biome } from "../world/biome";
import { mountainHeight, riverInfo, riverUpper, terrainHeightM } from "../world/height";

/** World window shown by the map (meters): mountains west, forest east, river through the middle. */
const X0 = -2700;
const X1 = 2500;
const Z0 = -2100;
const Z1 = 1700;
/** Map texels across; the shaded relief is computed once, in slices so the frame never stalls. */
const RES_X = 360;

export interface MapPlace {
  label: string;
  x: number;
  z: number;
  /** Heading on arrival (radians, 0 = north / -z). */
  yaw: number;
}

/**
 * Debug / exploration map (K): shaded relief with river, forest and mountains, the player's
 * position, and click-to-teleport. Quick buttons jump straight to notable places.
 */
export class WorldMap {
  readonly el: HTMLDivElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly overlay: HTMLCanvasElement;
  private readonly resY = Math.round((RES_X * (Z1 - Z0)) / (X1 - X0));
  private image: ImageData | null = null;
  private row = 0;
  private visible = false;

  constructor(
    parent: HTMLElement,
    places: MapPlace[],
    private readonly onTeleport: (x: number, z: number, yaw: number) => void,
  ) {
    this.el = document.createElement("div");
    this.el.id = "map";
    this.el.hidden = true;
    this.canvas = document.createElement("canvas");
    this.canvas.width = RES_X;
    this.canvas.height = this.resY;
    this.overlay = document.createElement("canvas");
    this.overlay.width = RES_X * 2;
    this.overlay.height = this.resY * 2;
    const frame = document.createElement("div");
    frame.className = "map-frame";
    frame.append(this.canvas, this.overlay);
    const buttons = document.createElement("div");
    buttons.className = "map-places";
    for (const p of places) {
      const b = document.createElement("button");
      b.textContent = p.label;
      b.addEventListener("click", () => this.go(p.x, p.z, p.yaw));
      buttons.append(b);
    }
    const title = document.createElement("div");
    title.className = "map-title";
    title.textContent = "map · click to travel · K to close";
    this.el.append(title, frame, buttons);
    parent.append(this.el);
    this.overlay.addEventListener("click", (e) => {
      const r = this.overlay.getBoundingClientRect();
      const x = X0 + ((e.clientX - r.left) / r.width) * (X1 - X0);
      const z = Z0 + ((e.clientY - r.top) / r.height) * (Z1 - Z0);
      this.go(x, z, this.lastYaw);
    });
  }

  private lastYaw = 0;

  get open(): boolean {
    return this.visible;
  }

  toggle(): void {
    this.visible = !this.visible;
    this.el.hidden = !this.visible;
  }

  private go(x: number, z: number, yaw: number): void {
    this.onTeleport(x, z, yaw);
    this.toggle();
  }

  /** Shades a few rows of relief per call, then draws the player marker. */
  update(px: number, pz: number, yaw: number): void {
    this.lastYaw = yaw;
    if (!this.visible) return;
    if (!this.image) this.image = new ImageData(RES_X, this.resY);
    const budget = performance.now() + 6;
    while (this.row < this.resY && performance.now() < budget) this.shadeRow(this.row++);
    this.canvas.getContext("2d")!.putImageData(this.image, 0, 0);
    this.drawMarker(px, pz, yaw);
  }

  private shadeRow(j: number): void {
    const img = this.image!.data;
    const sx = (X1 - X0) / RES_X;
    const z = Z0 + (j + 0.5) * sx;
    for (let i = 0; i < RES_X; i++) {
      const x = X0 + (i + 0.5) * sx;
      const h = terrainHeightM(x, z);
      const hx = terrainHeightM(x + sx, z);
      const hz = terrainHeightM(x, z + sx);
      // Light from the north-west, exaggerated so gentle hills read.
      const shade = Math.max(0.35, Math.min(1.35, 1 + ((h - hx) * 0.7 + (h - hz) * 0.7) / sx));
      const [grove, , forest] = biome(x, z);
      const mtn = mountainHeight(x, z);
      let c: [number, number, number] = [196, 178, 104];
      c = mix(c, [120, 150, 70], grove * 0.6);
      c = mix(c, [160, 96, 44], forest * 0.75);
      c = mix(c, [64, 70, 40], forest * Math.max(0, Math.min(1, (x - 1000) / 900)) * 0.7);
      c = mix(c, [128, 122, 112], Math.min(1, mtn / 60));
      c = mix(c, [240, 240, 244], Math.max(0, Math.min(1, (h - 190) / 40)));
      const [d, water, hw] = riverInfo(x, z);
      if (d < hw * 1.1 || (d < hw * 2 && h < water)) c = riverUpper(x) > 0.3 ? [120, 170, 200] : [70, 120, 160];
      const o = (j * RES_X + i) * 4;
      img[o] = c[0] * shade;
      img[o + 1] = c[1] * shade;
      img[o + 2] = c[2] * shade;
      img[o + 3] = 255;
    }
  }

  private drawMarker(px: number, pz: number, yaw: number): void {
    const ctx = this.overlay.getContext("2d")!;
    const w = this.overlay.width;
    const h = this.overlay.height;
    ctx.clearRect(0, 0, w, h);
    if (this.row < this.resY) {
      ctx.fillStyle = "rgba(255,245,225,0.8)";
      ctx.font = "22px serif";
      ctx.fillText(`charting… ${Math.round((this.row / this.resY) * 100)}%`, 16, 32);
    }
    const x = ((px - X0) / (X1 - X0)) * w;
    const y = ((pz - Z0) / (Z1 - Z0)) * h;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(yaw);
    ctx.beginPath();
    ctx.moveTo(0, -14);
    ctx.lineTo(8, 9);
    ctx.lineTo(0, 4);
    ctx.lineTo(-8, 9);
    ctx.closePath();
    ctx.fillStyle = "#ff5a4a";
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2.5;
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

function mix(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
