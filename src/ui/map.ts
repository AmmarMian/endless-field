import { biome } from "../world/biome";
import { mountainHeight, riverInfo, riverUpper, terrainHeightM } from "../world/height";
import { BED_CELL, bedInCell } from "../world/bed-shape";
import { SUNFLOWERS, sunflowerField, sunflowerLocal } from "../world/sunflower-field";

/** World window shown by the map (meters): meadows west, forest east, river through the middle. */
const X0 = -1300;
const X1 = 1900;
const Z0 = -1100;
const Z1 = 900;
/** Map texels across; the shaded relief is computed once, in slices so the frame never stalls. */
const RES_X = 480;

export interface MapPlace {
  label: string;
  x: number;
  z: number;
  /** Heading on arrival (radians, 0 = north / -z). */
  yaw: number;
  /** Where the label sits, if not at the arrival point (e.g. a field's center). */
  at?: [number, number];
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
    this.places = places;
    for (let cx = Math.floor(X0 / BED_CELL); cx <= Math.ceil(X1 / BED_CELL); cx++) {
      for (let cz = Math.floor(Z0 / BED_CELL); cz <= Math.ceil(Z1 / BED_CELL); cz++) {
        const b = bedInCell(cx, cz);
        if (b) this.beds.push({ x: b.x, z: b.z, r: b.radius, species: b.species });
      }
    }
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
  private readonly places: MapPlace[];
  /** Flower beds in the map window (colored dots). */
  private readonly beds: { x: number; z: number; r: number; species: number }[] = [];

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
      // Sunflower field: gold, striped by its drill rows.
      const sf = sunflowerField(x, z);
      if (sf > 0) {
        const [, b] = sunflowerLocal(x, z);
        const stripe = 0.85 + 0.15 * Math.cos((b / (SUNFLOWERS.row * 6)) * Math.PI * 2);
        c = mix(c, [236 * stripe, 178 * stripe, 34 * stripe], sf);
      }
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
    const toMap = (wx: number, wz: number): [number, number] => [((wx - X0) / (X1 - X0)) * w, ((wz - Z0) / (Z1 - Z0)) * h];
    const BED_COLORS = ["#e9a3c9", "#f4d35e", "#9b8bf4", "#f08a5d", "#f7f7f7"];
    for (const b of this.beds) {
      const [bx, by] = toMap(b.x, b.z);
      ctx.beginPath();
      ctx.arc(bx, by, Math.max(1.6, (b.r / (X1 - X0)) * w), 0, Math.PI * 2);
      ctx.fillStyle = BED_COLORS[b.species % BED_COLORS.length];
      ctx.globalAlpha = 0.6;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.font = "italic 20px 'Cormorant Garamond', serif";
    ctx.textAlign = "center";
    for (const p of this.places) {
      const [lx, ly] = toMap(p.at?.[0] ?? p.x, p.at?.[1] ?? p.z);
      ctx.beginPath();
      ctx.arc(lx, ly, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#fff5e1";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(30,18,8,0.8)";
      ctx.strokeText(p.label, lx, ly - 9);
      ctx.fillText(p.label, lx, ly - 9);
    }
    ctx.textAlign = "start";
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
