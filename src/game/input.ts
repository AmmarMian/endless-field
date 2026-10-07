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
  /** The pointer left the window: steering eases back to straight flight. */
  private outside = false;
  /** Keyboard steering, eased toward the held direction; owns steering until the mouse moves. */
  private keySteer = 0;
  private keyboardMode = false;
  private lastPointer: [number, number] | null = null;
  private keys = new Set<string>();
  /** Presses latched on keydown so a quick tap between two frames is never lost. */
  private pressed = new Set<string>();
  private readonly listeners: (() => void)[] = [];
  /** Touch: a finger dragged from where it landed steers like a joystick. */
  touchMode = false;
  private touch: { id: number; x: number; y: number } | null = null;
  /** Held by the on-screen gust button (or a second finger). */
  touchGust = false;
  private fingers = 0;
  /** Tilt steering (phone orientation), in [-1, 1]; null when not in use. */
  tilt: [number, number] | null = null;

  constructor(private readonly el: HTMLElement) {
    const on = <K extends keyof WindowEventMap>(target: Window | HTMLElement, type: K, fn: (e: WindowEventMap[K]) => void, opts?: AddEventListenerOptions) => {
      target.addEventListener(type, fn as EventListener, opts);
      this.listeners.push(() => target.removeEventListener(type, fn as EventListener));
    };
    on(document.documentElement, "pointerleave", () => (this.outside = true));
    on(document.documentElement, "pointerenter", () => (this.outside = false));
    on(window, "pointermove", (e) => {
      if (e.pointerType === "touch") {
        if (this.touch && e.pointerId === this.touch.id && !this.suspended) {
          // About 7% of the screen's short side for full lock.
          const reach = Math.min(innerWidth, innerHeight) * 0.07;
          this.steerX = Math.max(-1, Math.min(1, (e.clientX - this.touch.x) / reach));
          this.steerY = Math.max(-1, Math.min(1, (e.clientY - this.touch.y) / reach));
        }
        return;
      }
      this.outside = false;
      // Moving toward the HUD (settings gear, panel) must not steer the wind.
      const overUi = (e.target as HTMLElement | null)?.closest?.("#gear, #settings") != null;
      if (overUi || this.suspended) return;
      // A deliberate mouse move hands steering back to the pointer.
      if (this.keyboardMode && this.lastPointer && Math.hypot(e.clientX - this.lastPointer[0], e.clientY - this.lastPointer[1]) < 40) return;
      this.keyboardMode = false;
      this.lastPointer = [e.clientX, e.clientY];
      this.aim(e.clientX, e.clientY);
    });
    on(el, "pointerdown", (e) => {
      this.active = true;
      if (e.pointerType === "touch") {
        this.touchMode = true;
        this.fingers++;
        if (!this.touch) this.touch = { id: e.pointerId, x: e.clientX, y: e.clientY };
        return;
      }
      this.pointerDown = true;
      this.aim(e.clientX, e.clientY);
    });
    const release = (e: PointerEvent) => {
      if (e.pointerType === "touch") {
        this.fingers = Math.max(0, this.fingers - 1);
        if (this.touch && e.pointerId === this.touch.id) this.touch = null;
        return;
      }
      this.pointerDown = false;
    };
    on(window, "pointerup", release);
    on(window, "pointercancel", release);
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

  update(dt = 1 / 60): void {
    const k = this.keys;
    if (this.suspended || (this.outside && !this.keyboardMode)) {
      this.steerX *= 0.85;
      this.steerY *= 0.85;
    }
    if (this.touchMode && !this.touch) {
      // No finger on the glass: tilt steers, or the wind straightens out.
      if (this.tilt) {
        this.steerX += (this.tilt[0] - this.steerX) * Math.min(1, dt * 8);
        this.steerY += (this.tilt[1] - this.steerY) * Math.min(1, dt * 8);
      } else {
        this.steerX *= Math.exp(-dt * 6);
        this.steerY *= Math.exp(-dt * 6);
      }
    }
    this.gust = this.pointerDown || k.has("Space") || this.touchGust || this.fingers >= 2;
    this.rise = k.has("ShiftLeft") || k.has("ShiftRight") || k.has("KeyW") || k.has("ArrowUp");
    this.dive = k.has("ControlLeft") || k.has("ControlRight") || k.has("KeyS") || k.has("ArrowDown");
    // Keyboard steering (WASD on QWERTY = ZQSD on AZERTY, or arrows): eases in while held and
    // back to straight on release, like leaning into the wind.
    const left = k.has("KeyA") || k.has("ArrowLeft");
    const right = k.has("KeyD") || k.has("ArrowRight");
    const want = (right ? 1 : 0) - (left ? 1 : 0);
    if (want !== 0 || this.rise || this.dive) {
      if (!this.keyboardMode) this.keySteer = this.steerX;
      this.keyboardMode = true;
    }
    if (this.keyboardMode) {
      const rate = want !== 0 ? 3.5 : 5;
      this.keySteer += (want * 0.85 - this.keySteer) * Math.min(1, dt * rate);
      this.steerX = this.keySteer;
      this.steerY = 0;
    }
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
