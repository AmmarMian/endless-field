/** Pointer + keyboard + touch steering. Steer values are in [-1, 1] relative to screen center. */
export class Input {
  steerX = 0;
  steerY = 0;
  gust = false;
  rise = false;
  dive = false;
  active = false;
  /** True while the pointer is over UI or a menu is open: steering eases back to center. */
  suspended = false;
  private pointerDown = false;
  private keys = new Set<string>();
  /** Presses latched on keydown so a quick tap between two frames is never lost. */
  private pressed = new Set<string>();
  private readonly listeners: (() => void)[] = [];

  constructor(private readonly el: HTMLElement) {
    const on = <K extends keyof WindowEventMap>(target: Window | HTMLElement, type: K, fn: (e: WindowEventMap[K]) => void, opts?: AddEventListenerOptions) => {
      target.addEventListener(type, fn as EventListener, opts);
      this.listeners.push(() => target.removeEventListener(type, fn as EventListener));
    };
    on(window, "pointermove", (e) => {
      // Moving toward the HUD (settings gear, panel) must not steer the wind.
      const overUi = (e.target as HTMLElement | null)?.closest?.("#gear, #settings") != null;
      if (overUi || this.suspended) return;
      this.aim(e.clientX, e.clientY);
    });
    on(el, "pointerdown", (e) => {
      this.pointerDown = true;
      this.active = true;
      this.aim(e.clientX, e.clientY);
    });
    on(window, "pointerup", () => (this.pointerDown = false));
    on(window, "pointercancel", () => (this.pointerDown = false));
    on(window, "blur", () => {
      this.pointerDown = false;
      this.keys.clear();
    });
    on(window, "keydown", (e) => {
      this.keys.add(e.code);
      // Commands follow the typed letter (layout-independent: AZERTY's M is not KeyM);
      // movement keeps physical positions (WASD on QWERTY = ZQSD on AZERTY).
      if (!e.repeat && e.key.length === 1) this.pressed.add(e.key.toLowerCase());
      if (e.code === "Space") e.preventDefault();
    });
    on(window, "keyup", (e) => this.keys.delete(e.code));
    on(el, "contextmenu", (e) => e.preventDefault());
  }

  private aim(x: number, y: number): void {
    const r = this.el.getBoundingClientRect();
    this.steerX = Math.max(-1, Math.min(1, ((x - r.left) / r.width) * 2 - 1));
    this.steerY = Math.max(-1, Math.min(1, ((y - r.top) / r.height) * 2 - 1));
  }

  update(): void {
    const k = this.keys;
    if (this.suspended) {
      this.steerX *= 0.85;
      this.steerY *= 0.85;
    }
    this.gust = this.pointerDown || k.has("Space");
    this.rise = k.has("ShiftLeft") || k.has("ShiftRight") || k.has("KeyW") || k.has("ArrowUp");
    this.dive = k.has("ControlLeft") || k.has("ControlRight") || k.has("KeyS") || k.has("ArrowDown");
    // Keyboard steering overrides the pointer while held.
    if (k.has("KeyA") || k.has("ArrowLeft")) this.steerX = -0.7;
    else if (k.has("KeyD") || k.has("ArrowRight")) this.steerX = 0.7;
  }

  /** Drops presses nobody asked about this frame. */
  endFrame(): void {
    this.pressed.clear();
  }

  /** True once per press of the typed character `letter` (e.g. "m"). */
  wasPressed(letter: string): boolean {
    return this.pressed.delete(letter.toLowerCase());
  }

  dispose(): void {
    for (const off of this.listeners) off();
  }
}
