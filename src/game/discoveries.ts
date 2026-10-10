/**
 * The living things of the world, found by playing. Each is celebrated once with a little
 * jingle (remembered in local storage).
 */
export interface Note {
  id: string;
  name: string;
  /** Shown once found. */
  text: string;
}

export const NOTES: Note[] = [
  { id: "flowers", name: "Wildflowers", text: "They open as the wind brushes past." },
  { id: "sparrows", name: "Sparrows", text: "Flocks feed hidden in the long grass and burst up as the wind sweeps through." },
  { id: "lantern", name: "Stone lanterns", text: "A breath of wind wakes their glow." },
  { id: "torii", name: "Torii", text: "A vermilion gate at each end of the lantern path." },
  { id: "chain", name: "A path of light", text: "Every lantern lit in one breath, a melody from end to end." },
  { id: "leaves", name: "Falling leaves", text: "The forest lends the wind its leaves." },
  { id: "pollen", name: "Sunflower pollen", text: "Golden dust from a field of faces." },
  { id: "spray", name: "River spray", text: "Skimming the river lifts a fine mist." },
  { id: "fireflies", name: "Fireflies", text: "Small lanterns of their own; the wind carries a few along." },
  { id: "snow", name: "Snow", text: "In winter, the wind lifts the powder." },
  { id: "hare", name: "Brown hares", text: "They bolt from a rushing wind; come gently and they go on grazing, watching you." },
  { id: "deer", name: "Roe deer", text: "Shy at the edge of the woods: a soft wind can pass among them." },
  { id: "fox", name: "The red fox", text: "Hunting voles in the grass; skim low past it and it springs at you." },
  { id: "butterflies", name: "Butterflies", text: "Over the flower beds; a gust tumbles them, a breeze lets them be." },
  { id: "ducks", name: "Mallards", text: "Dabbling on the slow river; they let a gentle wind come near." },
  { id: "dragonflies", name: "Dragonflies", text: "Hanging and darting over the water." },
  { id: "frogs", name: "Frogs", text: "Rest by the river and they sing to you, one after another." },
];

const KEY = "endless-field-notes";

export class Discoveries {
  private readonly found: Set<string>;
  /** Called with the note when something is found for the first time. */
  onFind: ((n: Note) => void) | null = null;

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
    try {
      localStorage.setItem(KEY, JSON.stringify([...this.found]));
    } catch {
      // Non-essential.
    }
    this.onFind?.(note);
  }

}
