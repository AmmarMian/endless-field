/**
 * Generative ambient score: gentle and a little cute, in D major pentatonic at 75 bpm (the
 * same key and grid as the lantern chimes, so everything played on top stays in harmony).
 *
 *   kalimba   arpeggios over a slow I - vi - IV - V progression (D, Bm, G, A), one chord per bar
 *   music box a sparse melody, phrased in 4-bar sentences that rise and settle, resting often,
 *             sometimes echoing its last note an octave up
 *   bass      soft sine on the chord roots
 * At night it thins out and sinks lower. Notes are scheduled ~0.4 s ahead (Web Audio clock).
 */
const BPM = 75;
const BEAT = 60 / BPM;
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
    this.bus.gain.setTargetAtTime(0.32, ctx.currentTime + 1.5, 2.5);
    this.next = Math.ceil(ctx.currentTime / BEAT) * BEAT + BEAT;
    this.timer = window.setInterval(() => this.schedule(), 100);
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
    this.bus.gain.setTargetAtTime(on ? 0.32 : 0, this.ctx.currentTime, 1.2);
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
    // Bass on beat one of each bar.
    if (eighth === 0) this.bass(chord[0] - 12 + shift * 0, t, 0.22);
    // Kalimba: up-and-back arpeggio on the beats, with a lilting pickup on the 4th eighth.
    const arp = [0, 2, 1, 3, 2, 1, 3, 2];
    if ((eighth % 2 === 0 || eighth === 3) && Math.random() > sparse * 0.5) {
      this.kalimba(chord[arp[eighth]] + 12 + shift, t, eighth === 0 ? 0.16 : 0.1);
    }
    // Music box: a phrase of 4 bars; plays on some eighths, walks the scale by small steps,
    // rising through bars 1-3 and settling home in bar 4.
    const phraseBar = bar % 4;
    const density = [0.35, 0.4, 0.45, 0.3][phraseBar] * (1 - sparse * 0.6);
    if (eighth % 2 === 0 && Math.random() < density) {
      const pull = phraseBar < 3 ? 0.6 : -0.8;
      const move = Math.random() < 0.5 + pull * 0.3 ? 1 : -1;
      this.degree = Math.max(4, Math.min(12, this.degree + move * (Math.random() < 0.8 ? 1 : 2)));
      if (phraseBar === 3 && eighth === 6) this.degree = 5;
      const note = 62 + Math.floor(this.degree / 5) * 12 + SCALE[this.degree % 5] + shift;
      this.musicBox(note, t, 0.07);
      if (Math.random() < 0.18) this.musicBox(note + 12, t + BEAT * 0.75, 0.03);
    }
  }

  private voice(t: number, attack: number, decay: number, level: number): GainNode {
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(level, t + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t + attack + decay);
    g.connect(this.bus);
    return g;
  }

  private osc(type: OscillatorType, f: number, t: number, dur: number, out: AudioNode, level = 1): void {
    const o = this.ctx.createOscillator();
    o.type = type;
    o.frequency.value = f;
    const g = this.ctx.createGain();
    g.gain.value = level;
    o.connect(g).connect(out);
    o.start(t);
    o.stop(t + dur);
  }

  /** Kalimba: a plucked tine (fast attack, a woody inharmonic partial that dies quickly). */
  private kalimba(n: number, t: number, level: number): void {
    const body = this.voice(t, 0.004, 1.6, level);
    this.osc("sine", hz(n), t, 1.8, body);
    const click = this.voice(t, 0.002, 0.12, level * 0.6);
    this.osc("sine", hz(n) * 5.4, t, 0.2, click, 0.5);
  }

  /** Music box: a bright, small comb tooth (fundamental plus a glassy upper partial). */
  private musicBox(n: number, t: number, level: number): void {
    const v = this.voice(t, 0.003, 2.2, level);
    this.osc("sine", hz(n), t, 2.4, v);
    this.osc("sine", hz(n) * 3.01, t, 0.8, v, 0.25);
    this.osc("triangle", hz(n) * 2, t, 0.5, v, 0.08);
  }

  /** Soft round bass. */
  private bass(n: number, t: number, level: number): void {
    const v = this.voice(t, 0.08, BEAT * 3.5, level);
    this.osc("sine", hz(n), t, BEAT * 4, v);
  }
}
