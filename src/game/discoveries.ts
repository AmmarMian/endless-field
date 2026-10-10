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
  { id: "hare", name: "Brown hares", text: "Come gently, without a gust, and linger: a hare will run with you." },
  { id: "deer", name: "Roe deer", text: "Shy at the edge of the woods; approached softly they bound along beside you." },
  { id: "fox", name: "The red fox", text: "Skim low past it and it springs at you in play; linger and it trots along." },
  { id: "butterflies", name: "Butterflies", text: "Drift through them slowly and they dance around you for a while." },
  { id: "ducks", name: "Mallards", text: "Glide low over the river beside them and they paddle in your wake." },
  { id: "dragonflies", name: "Dragonflies", text: "Hang near one over the water and it flies at your wingtip." },
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
