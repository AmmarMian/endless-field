/**
 * The living things of the world, found by playing. Each is celebrated once (remembered in
 * local storage); after a while without a new find, a riddle toward the next is whispered.
 */
export interface Note {
  id: string;
  name: string;
  /** Shown once found. */
  text: string;
  /** Shown before: a nudge toward it. */
  hint: string;
}

export const NOTES: Note[] = [
  { id: "flowers", name: "Wildflowers", text: "They open as the wind brushes past.", hint: "Brush low over the wildflowers." },
  { id: "sparrows", name: "Sparrows", text: "Flocks feed hidden in the long grass and burst up as the wind sweeps through.", hint: "Something chirps in the long grass. Listen for it, then sweep through." },
  { id: "lantern", name: "Stone lanterns", text: "A breath of wind wakes their glow.", hint: "Old stones line a winding path. Pass close by them." },
  { id: "torii", name: "Torii", text: "A vermilion gate at each end of the lantern path.", hint: "Find the red gate where the lanterns begin." },
  { id: "chain", name: "A path of light", text: "Every lantern lit in one breath, a melody from end to end.", hint: "Light every lantern on the path without stopping." },
  { id: "leaves", name: "Falling leaves", text: "The forest lends the wind its leaves.", hint: "Fly low through the trees to the east." },
  { id: "pollen", name: "Sunflower pollen", text: "Golden dust from a field of faces.", hint: "Across the river, a sea of faces turns east." },
  { id: "spray", name: "River spray", text: "Skimming the river lifts a fine mist.", hint: "Skim low over running water." },
  { id: "fireflies", name: "Fireflies", text: "Small lanterns of their own; the wind carries a few along.", hint: "At night, small lights drift low over the meadow." },
  { id: "snow", name: "Snow", text: "In winter, the wind lifts the powder.", hint: "Come back when the fields are white." },
];

const KEY = "endless-field-notes";
/** Seconds of play without a new find before a hint is offered. */
const HINT_AFTER = 75;

export class Discoveries {
  private readonly found: Set<string>;
  private sinceFind = 0;
  private hinted = new Set<string>();
  /** Called with the note when something is found for the first time. */
  onFind: ((n: Note) => void) | null = null;
  /** Called with a riddle when the player could use a nudge. */
  onHint: ((n: Note) => void) | null = null;

  constructor() {
    let ids: string[] = [];
    try {
      ids = JSON.parse(localStorage.getItem(KEY) ?? "[]") as string[];
    } catch {
      // Storage can be unavailable (private mode).
    }
    this.found = new Set(ids);
  }

  has(id: string): boolean {
    return this.found.has(id);
  }

  get count(): number {
    return this.found.size;
  }

  find(id: string): void {
    if (this.found.has(id)) return;
    const note = NOTES.find((n) => n.id === id);
    if (!note) return;
    this.found.add(id);
    this.sinceFind = 0;
    try {
      localStorage.setItem(KEY, JSON.stringify([...this.found]));
    } catch {
      // Non-essential.
    }
    this.onFind?.(note);
  }

  /** Ticks the hint clock while playing; `available` filters riddles that make sense now. */
  update(dt: number, available: (id: string) => boolean): void {
    this.sinceFind += dt;
    if (this.sinceFind < HINT_AFTER) return;
    this.sinceFind = 0;
    const next = NOTES.find((n) => !this.found.has(n.id) && !this.hinted.has(n.id) && available(n.id));
    if (next) {
      this.hinted.add(next.id);
      this.onHint?.(next);
    }
  }
}
