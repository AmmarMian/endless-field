import { feltPiano } from "./piano";

/**
 * Generative exploration score for a felt piano, kept far in the background: sparse fragments
 * in D (on the same 75 bpm grid as the lantern notes) between long silences. At night the
 * fragments sink an octave and come less often. Notes are scheduled ~0.4 s ahead.
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

  /** Notes still to play in the current fragment: [eighth offset, MIDI note, velocity]. */
  private phrase: [number, number, number][] = [];
  private phraseStep = 0;
  /** Eighths of silence left before the next fragment. */
  private rest = 12;

  /**
   * Like a wanderer's piano: short fragments of a few notes, freely timed, separated by long
   * silences in which only the wind is heard. Each fragment draws on a small motif bank,
   * transposed within the key, sometimes over a single low note.
   */
  private play(step: number, t: number): void {
    const human = () => (Math.random() - 0.5) * 0.03;
    if (this.phrase.length === 0) {
      if (--this.rest > 0) return;
      this.compose();
      this.phraseStep = step;
    }
    const at = step - this.phraseStep;
    while (this.phrase.length && this.phrase[0][0] <= at) {
      const [, note, vel] = this.phrase.shift()!;
      this.piano(note, t + human(), vel);
    }
    if (this.phrase.length === 0) {
      // 10-30 s of quiet (longer at night).
      this.rest = Math.round((24 + Math.random() * 48) * (1 + this.night * 0.6));
    }
  }

  private compose(): void {
    // Motifs as scale degrees (0 = D in the D major scale with a lydian colour) and eighth
    // offsets: a rising question, a falling answer, a hovering figure, an open fifth.
    const MOTIFS: [number, number][][] = [
      [[0, 4], [2, 3], [4, 3], [7, 6]],
      [[9, 0], [7, 2], [4, 3], [2, 5]],
      [[4, 0], [5, 2], [4, 1], [2, 3], [4, 6]],
      [[0, 0], [4, 3], [7, 2], [11, 4], [9, 6]],
      [[7, 0], [4, 2], [9, 3], [7, 5]],
    ];
    const MAJOR = [0, 2, 4, 5, 7, 9, 11];
    const motif = MOTIFS[Math.floor(Math.random() * MOTIFS.length)];
    const shift = Math.floor(Math.random() * 3) * 2;
    const octave = this.night > 0.5 ? 62 : 74;
    const out: [number, number, number][] = [];
    // Sometimes a low note first, left to ring under the fragment.
    if (Math.random() < 0.6) out.push([0, [38, 43, 45, 47][Math.floor(Math.random() * 4)], 0.3]);
    let at = 0;
    for (const [degree, gap] of motif) {
      at += gap === 0 && out.length ? 1 : gap;
      const d = degree + shift;
      const semis = Math.floor(d / 7) * 12 + MAJOR[((d % 7) + 7) % 7] + (d % 7 === 3 && Math.random() < 0.4 ? 1 : 0);
      out.push([at, octave + semis - 12, 0.28 + Math.random() * 0.14]);
      // An occasional sixth below, softly.
      if (Math.random() < 0.2) out.push([at, octave + semis - 21, 0.16]);
    }
    out.sort((a, b) => a[0] - b[0]);
    this.phrase = out;
  }

  private piano(n: number, t: number, velocity: number): void {
    feltPiano(this.ctx, this.bus, n, Math.max(t, this.ctx.currentTime), velocity);
  }
}
