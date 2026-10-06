/** Pointer + keyboard + touch steering. Steer values are in [-1, 1] relative to screen center. */
export class Input {
  steerX = 0;
  steerY = 0;
  gust = false;
  rise = false;
  dive = false;
  active = false;
  private pointerDown = false;
  private keys = new Set<string>();
  private readonly listeners: (() => void)[] = [];

  constructor(private readonly el: HTMLElement) {
    const on = <K extends keyof WindowEventMap>(target: Window | HTMLElement, type: K, fn: (e: WindowEventMap[K]) => void, opts?: AddEventListenerOptions) => {
      target.addEventListener(type, fn as EventListener, opts);
      this.listeners.push(() => target.removeEventListener(type, fn as EventListener));
    };
    on(window, "pointermove", (e) => this.aim(e.clientX, e.clientY));
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
    this.gust = this.pointerDown || k.has("Space");
    this.rise = k.has("ShiftLeft") || k.has("ShiftRight") || k.has("KeyW") || k.has("ArrowUp");
    this.dive = k.has("ControlLeft") || k.has("ControlRight") || k.has("KeyS") || k.has("ArrowDown");
    // Keyboard steering overrides the pointer while held.
    if (k.has("KeyA") || k.has("ArrowLeft")) this.steerX = -0.7;
    else if (k.has("KeyD") || k.has("ArrowRight")) this.steerX = 0.7;
  }

  wasPressed(code: string): boolean {
    if (this.keys.has(code)) {
      this.keys.delete(code);
      return true;
    }
    return false;
  }

  dispose(): void {
    for (const off of this.listeners) off();
  }
}
