import { feltPiano } from "./piano";

/**
 * Generative ambient score for a felt piano, kept far in the background: in D major
 * pentatonic at 75 bpm (the same key and grid as the lantern notes, so they stay in harmony).
 *
 *   left hand   the chord root low, then a slow broken chord over I - vi - IV - V
 *               (D, Bm, G, A), one chord per bar
 *   right hand  a sparse melody, phrased in 4-bar sentences that rise and settle, resting
 *               often, sometimes echoing an octave up
 * At night it thins out and sinks lower. Notes are scheduled ~0.4 s ahead (Web Audio clock).
 */
const BPM = 75;
const BEAT = 60 / BPM;
/** The score sits well under the wind: present only if you listen for it. */
const LEVEL = 0.16;
const SCALE = [0, 2, 4, 7, 9];
const CHORDS = [
  [50, 54, 57, 62], // D
  [47, 50, 54, 59], // Bm
  [43, 47, 50, 55], // G
  [45, 49, 52, 57], // A
];

function hz(n: number): number {
  return 440 * Math.pow(2, (n - 69) / 12);
}

export class Music {
  private readonly bus: GainNode;
  private next = 0;
  private step = 0;
  private timer = 0;
  private enabled = true;
  private night = 0;
  private degree = 7;

  constructor(private readonly ctx: AudioContext, out: AudioNode, reverb: AudioNode) {
    this.bus = ctx.createGain();
    this.bus.gain.value = 0.0;
    this.bus.connect(out);
    this.bus.connect(reverb);
    // Fade in over a few seconds after the start.
    this.bus.gain.setTargetAtTime(LEVEL, ctx.currentTime + 1.5, 2.5);
    this.next = Math.ceil(ctx.currentTime / BEAT) * BEAT + BEAT;
    this.timer = window.setInterval(() => this.schedule(), 100);
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
    this.bus.gain.setTargetAtTime(on ? LEVEL : 0, this.ctx.currentTime, 1.2);
  }

  setNight(amount: number): void {
    this.night = amount;
  }

  dispose(): void {
    clearInterval(this.timer);
  }

  /** Schedules every eighth-note slot that falls within the lookahead window. */
  private schedule(): void {
    const ahead = this.ctx.currentTime + 0.4;
    while (this.next < ahead) {
      if (this.enabled) this.play(this.step, this.next);
      this.next += BEAT / 2;
      this.step++;
    }
  }

  private play(step: number, t: number): void {
    const eighth = step % 8;
    const bar = Math.floor(step / 8);
    const chord = CHORDS[bar % 4];
    const sparse = this.night;
    const shift = this.night > 0.5 ? -12 : 0;
    const human = () => (Math.random() - 0.5) * 0.025;
    // Left hand: the root low on beat one, then a slow broken chord (fewer notes at night).
    if (eighth === 0) this.piano(chord[0] - 12, t + human(), 0.35);
    const broken: Record<number, number> = { 2: 1, 4: 2, 6: 3 };
    if (eighth in broken && Math.random() > 0.25 + sparse * 0.4) {
      this.piano(chord[broken[eighth]] + shift, t + human(), 0.22 + Math.random() * 0.08);
    }
    // Right hand: a phrase of 4 bars; walks the scale by small steps, rising through bars
    // 1-3 and settling home in bar 4.
    const phraseBar = bar % 4;
    const density = [0.28, 0.32, 0.36, 0.24][phraseBar] * (1 - sparse * 0.6);
    if (eighth % 2 === 0 && Math.random() < density) {
      const pull = phraseBar < 3 ? 0.6 : -0.8;
      const move = Math.random() < 0.5 + pull * 0.3 ? 1 : -1;
      this.degree = Math.max(4, Math.min(12, this.degree + move * (Math.random() < 0.8 ? 1 : 2)));
      if (phraseBar === 3 && eighth === 6) this.degree = 5;
      const note = 62 + Math.floor(this.degree / 5) * 12 + SCALE[this.degree % 5] + shift;
      this.piano(note, t + human(), 0.3 + Math.random() * 0.15);
      if (Math.random() < 0.15) this.piano(note + 12, t + BEAT * 0.75 + human(), 0.15);
    }
  }

  private piano(n: number, t: number, velocity: number): void {
    feltPiano(this.ctx, this.bus, n, Math.max(t, this.ctx.currentTime), velocity);
  }
}
