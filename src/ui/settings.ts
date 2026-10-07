import type { GrassQuality } from "../world/grass";

export type Preset = "low" | "medium" | "high" | "ultra";

export const TIMES = ["dawn", "day", "golden", "night"] as const;
export type TimeMode = "cycle" | (typeof TIMES)[number];

export const FILTERS = ["none", "painterly", "watercolor", "film", "miniature", "ink"] as const;
export type Filter = (typeof FILTERS)[number];

export interface Settings {
  preset: Preset | "custom";
  renderScale: number;
  grass: GrassQuality;
  drawDistance: number;
  bloom: boolean;
  showStats: boolean;
  /** Time of day: the day turns (about 10 minutes), or one time is held. */
  time: TimeMode;
  /** 0 = follow the display refresh rate. */
  fpsTarget: number;
  autoResolution: boolean;
  /** "cycle" turns the seasons; otherwise one season is held (the default: summer). */
  seasonMode: "cycle" | "summer" | "autumn" | "winter" | "spring";
  /** Weather: showers come and go (auto), or always clear / always raining. */
  weather: "auto" | "clear" | "rain";
  /** Generative background score. */
  music: boolean;
  /** Who you are: the wind itself, or a swallow riding it. */
  avatar: "wind" | "swallow";
  /** Phones: steer by dragging a finger, or by tilting the phone. */
  steering: "touch" | "tilt";
  /** Screen style: a painted or photographic look over the final image. */
  filter: Filter;
  /** World seed: 0 is the original landscape; any other value generates a different one. */
  seed: number;
}

export const PRESETS: Record<Preset, Omit<Settings, "preset" | "showStats" | "time" | "fpsTarget" | "autoResolution" | "seed" | "seasonMode" | "music" | "weather" | "filter" | "steering" | "avatar">> = {
  low: { renderScale: 0.6, grass: "low", drawDistance: 0.7, bloom: false },
  medium: { renderScale: 0.8, grass: "medium", drawDistance: 0.9, bloom: true },
  high: { renderScale: 1, grass: "high", drawDistance: 1, bloom: true },
  ultra: { renderScale: 1, grass: "ultra", drawDistance: 1.35, bloom: true },
};

const KEY = "endless-field-settings";

function defaults(): Settings {
  // Phones and small integrated GPUs start on medium.
  const mobile = matchMedia("(pointer: coarse)").matches;
  const preset: Preset = mobile ? "low" : "high";
  return { preset, ...PRESETS[preset], showStats: true, time: "cycle", fpsTarget: 60, autoResolution: true, seed: 0, seasonMode: "summer", music: false, weather: "auto", filter: "miniature", steering: "touch", avatar: "wind", ...(mobile ? { fpsTarget: 30, showStats: false } : {}) };
}

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const saved = JSON.parse(raw) as Partial<Settings> & { v?: number };
      // v2: music became opt-in; older saves had it on by default.
      if ((saved.v ?? 1) < 2) {
        saved.music = false;
        saved.filter = "miniature";
      }
      // v3: phones start light (earlier saves on phones could hold desktop settings).
      if ((saved.v ?? 1) < 3 && matchMedia("(pointer: coarse)").matches) {
        Object.assign(saved, { preset: "low", ...PRESETS.low, fpsTarget: 30 });
      }
      return { ...defaults(), ...saved };
    }
  } catch {
    // Storage can be unavailable (private mode); fall back to defaults.
  }
  return defaults();
}

function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...s, v: 3 }));
  } catch {
    // Non-essential.
  }
}

type Row =
  | { k: keyof Settings; label: string; type: "choice"; options: [string | number, string][] }
  | { k: keyof Settings; label: string; type: "toggle"; on?: string; off?: string }
  | { k: keyof Settings; label: string; type: "range"; min: number; max: number; step: number }
  | { k: keyof Settings; label: string; type: "number" }
  | { label: string; type: "action"; run: (p: SettingsPanel) => void }
  | { label: string; type: "keys"; keys: string };

interface Page {
  id: string;
  label: string;
  icon: string;
  rows: Row[];
}

const PAGES: Page[] = [
  {
    id: "look",
    label: "Look",
    icon: "◐",
    rows: [
      { k: "filter", label: "Film simulation", type: "choice", options: [["none", "Natural"], ["painterly", "Painterly"], ["watercolor", "Watercolor"], ["film", "Film"], ["miniature", "Miniature"], ["ink", "Ink wash"]] },
      { k: "bloom", label: "Bloom", type: "toggle" },
    ],
  },
  {
    id: "world",
    label: "World",
    icon: "△",
    rows: [
      { k: "avatar", label: "Play as", type: "choice", options: [["wind", "The wind"], ["swallow", "A swallow"]] },
      { k: "seasonMode", label: "Season", type: "choice", options: [["summer", "Summer"], ["autumn", "Autumn"], ["winter", "Winter"], ["spring", "Spring"], ["cycle", "Turning"]] },
      { k: "weather", label: "Weather", type: "choice", options: [["auto", "Showers"], ["clear", "Clear"], ["rain", "Rain"]] },
      { k: "time", label: "Time of day", type: "choice", options: [["cycle", "Turning"], ["dawn", "Dawn"], ["day", "Day"], ["golden", "Golden hour"], ["night", "Night"]] },
      { k: "seed", label: "World seed", type: "number" },
      { label: "New world", type: "action", run: (p) => p.set({ seed: 1 + Math.floor(Math.random() * 999998) }) },
    ],
  },
  {
    id: "sound",
    label: "Sound",
    icon: "♪",
    rows: [{ k: "music", label: "Music", type: "toggle" }],
  },
  {
    id: "system",
    label: "System",
    icon: "▤",
    rows: [
      { k: "preset", label: "Quality", type: "choice", options: [["low", "Low"], ["medium", "Medium"], ["high", "High"], ["ultra", "Ultra"], ["custom", "Custom"]] },
      { k: "renderScale", label: "Render scale", type: "range", min: 0.5, max: 1, step: 0.05 },
      { k: "autoResolution", label: "Auto resolution", type: "toggle" },
      { k: "grass", label: "Grass density", type: "choice", options: [["low", "Low"], ["medium", "Medium"], ["high", "High"], ["ultra", "Ultra"]] },
      { k: "drawDistance", label: "Draw distance", type: "range", min: 0.5, max: 1.5, step: 0.05 },
      { k: "fpsTarget", label: "Frame rate", type: "choice", options: [[0, "Display"], [60, "60"], [30, "30"]] },
      { k: "showStats", label: "Frame counter", type: "toggle" },
      { k: "steering", label: "Phone steering", type: "choice", options: [["touch", "Drag"], ["tilt", "Tilt"]] },
    ],
  },
  {
    id: "keys",
    label: "Keys",
    icon: "⌘",
    rows: [
      { label: "Steer", type: "keys", keys: "WASD · arrows · mouse" },
      { label: "Gust", type: "keys", keys: "Space · hold click" },
      { label: "Rise / dive", type: "keys", keys: "↑ ↓" },
      { label: "Vertical loop", type: "keys", keys: "E" },
      { label: "Let go", type: "keys", keys: "X" },
      { label: "Pause", type: "keys", keys: "P" },
      { label: "Free roam", type: "keys", keys: "M" },
      { label: "Map", type: "keys", keys: "K" },
      { label: "Film simulation", type: "keys", keys: "L" },
      { label: "Next time of day", type: "keys", keys: "N" },
      { label: "Next season", type: "keys", keys: "Y" },
      { label: "Rain", type: "keys", keys: "R" },
      { label: "Frame counter", type: "keys", keys: "F" },
      { label: "Menu", type: "keys", keys: "O" },
    ],
  },
];

/** Keys whose change does not turn the quality preset into "custom". */
const FREE_KEYS = ["showStats", "time", "fpsTarget", "autoResolution", "seed", "seasonMode", "music", "weather", "filter", "steering", "avatar"];

/**
 * Settings, styled as a camera menu: tabbed pages of rows, each value stepped with ‹ ›.
 * Mouse or keyboard (↑↓ select, ←→ change, Q/E or Tab page, Enter toggle, Esc close).
 */
export class SettingsPanel {
  private readonly root: HTMLElement;
  private readonly tabs: HTMLElement;
  private readonly body: HTMLElement;
  private readonly hint: HTMLElement;
  private page = 0;
  private row = 0;

  constructor(private settings: Settings, private readonly onChange: (s: Settings) => void) {
    const gear = document.createElement("button");
    gear.id = "gear";
    gear.title = "Menu (O)";
    gear.innerHTML = `<span class="dot"></span>MENU`;
    this.root = document.createElement("div");
    this.root.id = "settings";
    this.root.className = "osd";
    this.root.hidden = true;
    this.root.innerHTML = `
      <div class="osd-head"><span class="osd-mode">MENU</span><nav class="osd-tabs"></nav></div>
      <div class="osd-body"></div>
      <div class="osd-foot"></div>`;
    this.tabs = this.root.querySelector(".osd-tabs")!;
    this.body = this.root.querySelector(".osd-body")!;
    this.hint = this.root.querySelector(".osd-foot")!;
    document.getElementById("hud")!.append(gear, this.root);
    gear.addEventListener("click", () => this.toggle());
    // Keep clicks on the panel from reaching the game.
    for (const el of [gear, this.root]) el.addEventListener("pointerdown", (e) => e.stopPropagation());
    window.addEventListener("keydown", (e) => this.key(e), { capture: true });
    this.render();
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
    this.commit();
  }

  private commit(): void {
    saveSettings(this.settings);
    this.render();
    this.onChange(this.settings);
  }

  /** Applies one value the way the old form did: presets fill in their values. */
  private apply(k: keyof Settings, value: unknown): void {
    if (k === "preset") {
      const p = value as Preset | "custom";
      this.settings = p === "custom" ? { ...this.settings, preset: p } : { ...this.settings, preset: p, ...PRESETS[p] };
    } else {
      this.settings = { ...this.settings, [k]: value };
      if (!FREE_KEYS.includes(k)) this.settings.preset = "custom";
    }
    this.commit();
  }

  /** Steps a row's value by `dir` (-1 / +1); toggles flip, actions run. */
  private step(r: Row, dir: number): void {
    if (r.type === "action") return r.run(this);
    if (r.type === "keys" || r.type === "number") return;
    const v = this.settings[r.k];
    if (r.type === "toggle") this.apply(r.k, !v);
    else if (r.type === "range") {
      const next = Math.round((Number(v) + dir * r.step) / r.step) * r.step;
      this.apply(r.k, Math.min(r.max, Math.max(r.min, Number(next.toFixed(3)))));
    } else {
      const i = r.options.findIndex(([o]) => o === v);
      this.apply(r.k, r.options[(i + dir + r.options.length) % r.options.length][0]);
    }
  }

  private key(e: KeyboardEvent): void {
    if (!this.open) return;
    const typing = e.target instanceof HTMLInputElement;
    if (typing && e.key !== "Escape") return;
    const rows = PAGES[this.page].rows;
    const go = (fn: () => void) => {
      e.preventDefault();
      e.stopImmediatePropagation();
      fn();
      this.render();
    };
    switch (e.key) {
      case "ArrowUp":
        return go(() => (this.row = (this.row - 1 + rows.length) % rows.length));
      case "ArrowDown":
        return go(() => (this.row = (this.row + 1) % rows.length));
      case "ArrowLeft":
        return go(() => this.step(rows[this.row], -1));
      case "ArrowRight":
      case "Enter":
        return go(() => this.step(rows[this.row], 1));
      case "Tab":
      case "e":
      case "E":
        return go(() => this.selectPage(this.page + (e.shiftKey ? -1 : 1)));
      case "q":
      case "Q":
        return go(() => this.selectPage(this.page - 1));
      case "Escape":
        return go(() => {
          (document.activeElement as HTMLElement | null)?.blur();
          this.toggle(false);
        });
    }
  }

  private selectPage(i: number): void {
    this.page = (i + PAGES.length) % PAGES.length;
    this.row = 0;
  }

  private valueText(r: Row): string {
    if (r.type === "action") return "▸";
    if (r.type === "keys") return r.keys;
    const v = this.settings[r.k];
    if (r.type === "toggle") return v ? (r.on ?? "On") : (r.off ?? "Off");
    if (r.type === "range") return Number(v).toFixed(2);
    if (r.type === "choice") return r.options.find(([o]) => o === v)?.[1] ?? String(v);
    return String(v);
  }

  private render(): void {
    this.tabs.replaceChildren(
      ...PAGES.map((p, i) => {
        const t = document.createElement("button");
        t.className = "osd-tab" + (i === this.page ? " on" : "");
        t.innerHTML = `<i>${p.icon}</i><span>${p.label}</span>`;
        t.addEventListener("click", () => {
          this.selectPage(i);
          this.render();
        });
        return t;
      }),
    );
    const page = PAGES[this.page];
    this.body.replaceChildren(
      ...page.rows.map((r, i) => {
        const el = document.createElement("div");
        el.className = `osd-row ${r.type}` + (i === this.row ? " sel" : "");
        const label = document.createElement("span");
        label.className = "lbl";
        label.textContent = r.label;
        const val = document.createElement("span");
        val.className = "val";
        if (r.type === "number") {
          const input = document.createElement("input");
          input.type = "number";
          input.min = "0";
          input.max = "999999";
          input.value = String(this.settings[r.k]);
          // The seed regenerates the world (a reload), so it applies on commit.
          input.addEventListener("change", () => this.apply(r.k, Math.max(0, Math.floor(Number(input.value) || 0))));
          val.append(input);
        } else if (r.type === "choice" || r.type === "toggle" || r.type === "range") {
          const prev = document.createElement("button");
          prev.textContent = "‹";
          prev.addEventListener("click", (e) => {
            e.stopPropagation();
            this.row = i;
            this.step(r, -1);
          });
          const next = document.createElement("button");
          next.textContent = "›";
          next.addEventListener("click", (e) => {
            e.stopPropagation();
            this.row = i;
            this.step(r, 1);
          });
          const text = document.createElement("b");
          text.textContent = this.valueText(r);
          if (r.type === "toggle" && this.settings[r.k]) text.className = "is-on";
          if (r.type === "range") {
            const meter = document.createElement("span");
            meter.className = "meter";
            const k = (Number(this.settings[r.k]) - r.min) / (r.max - r.min);
            meter.style.setProperty("--k", String(k));
            val.append(prev, meter, text, next);
          } else val.append(prev, text, next);
        } else {
          val.textContent = this.valueText(r);
        }
        el.append(label, val);
        el.addEventListener("click", () => {
          const was = this.row === i;
          this.row = i;
          if (r.type === "toggle" || r.type === "action" || (was && r.type === "choice")) this.step(r, 1);
          else this.render();
        });
        return el;
      }),
    );
    this.hint.innerHTML = `<span><kbd>↑</kbd><kbd>↓</kbd> select</span><span><kbd>←</kbd><kbd>→</kbd> change</span><span><kbd>Q</kbd><kbd>E</kbd> page</span><span><kbd>O</kbd> close</span>`;
  }
}
