import type { GrassQuality } from "../world/grass";

export type Preset = "low" | "medium" | "high" | "ultra";

export interface Settings {
  preset: Preset | "custom";
  renderScale: number;
  grass: GrassQuality;
  drawDistance: number;
  bloom: boolean;
  showStats: boolean;
  night: boolean;
  /** 0 = follow the display refresh rate. */
  fpsTarget: number;
  autoResolution: boolean;
  /** "cycle" turns the seasons; otherwise one season is held (the default: summer). */
  seasonMode: "cycle" | "summer" | "autumn" | "winter" | "spring";
  /** World seed: 0 is the original landscape; any other value generates a different one. */
  seed: number;
}

export const PRESETS: Record<Preset, Omit<Settings, "preset" | "showStats" | "night" | "fpsTarget" | "autoResolution" | "seed" | "seasonMode">> = {
  low: { renderScale: 0.6, grass: "low", drawDistance: 0.7, bloom: false },
  medium: { renderScale: 0.8, grass: "medium", drawDistance: 0.9, bloom: true },
  high: { renderScale: 1, grass: "high", drawDistance: 1, bloom: true },
  ultra: { renderScale: 1, grass: "ultra", drawDistance: 1.35, bloom: true },
};

const KEY = "endless-field-settings";

function defaults(): Settings {
  // Phones and small integrated GPUs start on medium.
  const mobile = matchMedia("(pointer: coarse)").matches;
  const preset: Preset = mobile ? "medium" : "high";
  return { preset, ...PRESETS[preset], showStats: true, night: false, fpsTarget: 60, autoResolution: true, seed: 0, seasonMode: "summer" };
}

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...defaults(), ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    // Storage can be unavailable (private mode); fall back to defaults.
  }
  return defaults();
}

function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // Non-essential.
  }
}

/** Small settings panel; calls `onChange` with the full settings whenever anything changes. */
export class SettingsPanel {
  private readonly root: HTMLElement;
  private readonly fields: Record<string, HTMLInputElement | HTMLSelectElement> = {};

  constructor(private settings: Settings, private readonly onChange: (s: Settings) => void) {
    const gear = document.createElement("button");
    gear.id = "gear";
    gear.title = "Settings (O)";
    gear.textContent = "⚙";
    this.root = document.createElement("div");
    this.root.id = "settings";
    this.root.hidden = true;
    this.root.innerHTML = `
      <h2>Settings</h2>
      <label>Quality <select data-k="preset">
        <option value="low">Low</option><option value="medium">Medium</option>
        <option value="high">High</option><option value="ultra">Ultra</option><option value="custom">Custom</option>
      </select></label>
      <label>Render scale <input data-k="renderScale" type="range" min="0.5" max="1" step="0.05"><output></output></label>
      <label>Grass density <select data-k="grass">
        <option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="ultra">Ultra</option>
      </select></label>
      <label>Draw distance <input data-k="drawDistance" type="range" min="0.5" max="1.5" step="0.05"><output></output></label>
      <label>Target frame rate <select data-k="fpsTarget">
        <option value="0">Display</option><option value="60">60</option><option value="30">30</option>
      </select></label>
      <label class="check"><input data-k="autoResolution" type="checkbox"> Auto resolution (hold frame rate)</label>
      <label class="check"><input data-k="bloom" type="checkbox"> Bloom</label>
      <label class="check"><input data-k="showStats" type="checkbox"> Frame counter</label>
      <label class="check"><input data-k="night" type="checkbox"> Night</label>
      <label>Season <select data-k="seasonMode">
        <option value="cycle">Turning</option><option value="summer">Summer</option><option value="autumn">Autumn</option>
        <option value="winter">Winter</option><option value="spring">Spring</option>
      </select></label>
      <label>World seed <input data-k="seed" type="number" min="0" max="999999" step="1"></label>
      <button type="button" class="new-world">New world</button>
      <p class="keys">O settings · F frame counter · M free roam · N day / night · Y next season · K map</p>`;
    document.getElementById("hud")!.append(gear, this.root);
    for (const el of this.root.querySelectorAll<HTMLInputElement | HTMLSelectElement>("[data-k]")) {
      this.fields[el.dataset.k!] = el;
      // The seed regenerates the world (a reload), so it applies on commit, not per keystroke.
      el.addEventListener(el.dataset.k === "seed" ? "change" : "input", () => this.read(el.dataset.k!));
    }
    this.root.querySelector(".new-world")!.addEventListener("click", () => this.set({ seed: 1 + Math.floor(Math.random() * 999998) }));
    gear.addEventListener("click", () => this.toggle());
    // Keep clicks on the panel from reaching the game.
    for (const el of [gear, this.root]) el.addEventListener("pointerdown", (e) => e.stopPropagation());
    this.write();
  }

  get open(): boolean {
    return !this.root.hidden;
  }

  get current(): Settings {
    return this.settings;
  }

  toggle(force?: boolean): void {
    this.root.hidden = force === undefined ? !this.root.hidden : !force;
  }

  set(partial: Partial<Settings>): void {
    this.settings = { ...this.settings, ...partial };
    this.write();
    saveSettings(this.settings);
    this.onChange(this.settings);
  }

  private read(key: string): void {
    const el = this.fields[key];
    let next: Settings;
    if (key === "preset") {
      const p = el.value as Preset | "custom";
      next = p === "custom" ? { ...this.settings, preset: p } : { ...this.settings, preset: p, ...PRESETS[p] };
    } else {
      const value =
        el instanceof HTMLInputElement && el.type === "checkbox"
          ? el.checked
          : (el instanceof HTMLInputElement && (el.type === "range" || el.type === "number")) || key === "fpsTarget"
            ? Number(el.value)
            : el.value;
      next = { ...this.settings, [key]: value };
      if (key === "seed") next.seed = Math.max(0, Math.floor(Number(el.value) || 0));
      if (!["showStats", "night", "fpsTarget", "autoResolution", "seed", "seasonMode"].includes(key)) next.preset = "custom";
    }
    this.settings = next;
    this.write();
    saveSettings(next);
    this.onChange(next);
  }

  private write(): void {
    for (const [k, el] of Object.entries(this.fields)) {
      const v = (this.settings as unknown as Record<string, unknown>)[k];
      if (el instanceof HTMLInputElement && el.type === "checkbox") el.checked = Boolean(v);
      else el.value = String(v);
      const out = el.parentElement?.querySelector("output");
      if (out) out.textContent = Number(v).toFixed(2);
    }
  }
}
