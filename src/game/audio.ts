import { Music } from "./music";
/**
 * Procedural soundscape: wind that follows speed, pentatonic chimes for each bloom, a chord
 * when a whole cluster blooms, and a slow evolving drone. No audio assets needed.
 */
export class Audio {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private reverb!: ConvolverNode;
  private windGain!: GainNode;
  private windFilter!: BiquadFilterNode;
  private noteIndex = 0;
  private crickets!: GainNode;
  private nightLevel = 0;
  // D major pentatonic across three octaves.
  private readonly scale = [62, 64, 66, 69, 71, 74, 76, 78, 81, 83, 86, 88];

  start(): void {
    if (this.ctx) {
      void this.ctx.resume();
      return;
    }
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0.8;
    this.master.connect(ctx.destination);

    this.reverb = ctx.createConvolver();
    this.reverb.buffer = this.impulse(3.2);
    const wet = ctx.createGain();
    wet.gain.value = 0.55;
    this.reverb.connect(wet).connect(this.master);

    // Wind: looping pink-ish noise through a moving bandpass.
    const noise = ctx.createBufferSource();
    noise.buffer = this.noiseBuffer(4);
    noise.loop = true;
    this.windFilter = ctx.createBiquadFilter();
    this.windFilter.type = "bandpass";
    this.windFilter.frequency.value = 500;
    this.windFilter.Q.value = 0.7;
    this.windGain = ctx.createGain();
    this.windGain.gain.value = 0;
    noise.connect(this.windFilter).connect(this.windGain).connect(this.master);
    noise.start();

    this.drone([50, 57, 62], 0.022);
    this.crickets = ctx.createGain();
    this.crickets.gain.value = 0;
    this.crickets.connect(this.master);
    this.crickets.connect(this.reverb);
    this.scheduleCrickets();
    this.music = new Music(ctx, this.master, this.reverb);
    this.music.setEnabled(this.musicOn);
  }

  private music: Music | null = null;
  private rainGain: GainNode | null = null;
  private patterGain: GainNode | null = null;

  /** Rain on the meadow: a soft hiss plus pattering drops (0 dry .. 1 downpour). */
  setRain(amount: number): void {
    if (!this.ctx) return;
    if (!this.rainGain) {
      const ctx = this.ctx;
      const hiss = ctx.createBufferSource();
      hiss.buffer = this.noiseBuffer(5);
      hiss.loop = true;
      const lp = ctx.createBiquadFilter();
      lp.type = "lowpass";
      lp.frequency.value = 2400;
      this.rainGain = ctx.createGain();
      this.rainGain.gain.value = 0;
      hiss.connect(lp).connect(this.rainGain).connect(this.master);
      hiss.start();
      // Patter: brighter noise, amplitude-modulated by a fast random flutter.
      const patter = ctx.createBufferSource();
      patter.buffer = this.noiseBuffer(3);
      patter.loop = true;
      const hp = ctx.createBiquadFilter();
      hp.type = "bandpass";
      hp.frequency.value = 3800;
      hp.Q.value = 0.9;
      this.patterGain = ctx.createGain();
      this.patterGain.gain.value = 0;
      const flutter = ctx.createGain();
      flutter.gain.value = 0.5;
      const lfo = ctx.createOscillator();
      lfo.type = "sawtooth";
      lfo.frequency.value = 13;
      const lfoAmt = ctx.createGain();
      lfoAmt.gain.value = 0.5;
      lfo.connect(lfoAmt).connect(flutter.gain);
      lfo.start();
      patter.connect(hp).connect(flutter).connect(this.patterGain).connect(this.master);
      patter.start();
    }
    const t = this.ctx.currentTime;
    this.rainGain.gain.setTargetAtTime(amount * 0.16, t, 1.0);
    this.patterGain!.gain.setTargetAtTime(amount * 0.07, t, 1.0);
  }
  private musicOn = true;

  /** Background score on / off. */
  setMusic(on: boolean): void {
    this.musicOn = on;
    this.music?.setEnabled(on);
  }

  /** Cricket chirps: short trains of high sine pulses at random stereo positions. */
  private scheduleCrickets(): void {
    const ctx = this.ctx!;
    const tick = () => {
      if (this.nightLevel > 0.05) {
        const t = ctx.currentTime + Math.random() * 0.4;
        const pan = ctx.createStereoPanner();
        pan.pan.value = Math.random() * 2 - 1;
        pan.connect(this.crickets);
        const freq = 4200 + Math.random() * 900;
        const pulses = 3 + Math.floor(Math.random() * 4);
        for (let i = 0; i < pulses; i++) {
          const osc = ctx.createOscillator();
          osc.frequency.value = freq;
          const g = ctx.createGain();
          const s = t + i * 0.055;
          g.gain.setValueAtTime(0, s);
          g.gain.linearRampToValueAtTime(0.05, s + 0.008);
          g.gain.exponentialRampToValueAtTime(0.0001, s + 0.04);
          osc.connect(g).connect(pan);
          osc.start(s);
          osc.stop(s + 0.05);
        }
      }
      setTimeout(tick, 250 + Math.random() * 600);
    };
    tick();
  }

  setNight(amount: number): void {
    this.nightLevel = amount;
    this.music?.setNight(amount);
    if (this.ctx) this.crickets.gain.setTargetAtTime(amount * 0.9, this.ctx.currentTime, 0.5);
  }

  private noiseBuffer(seconds: number): AudioBuffer {
    const ctx = this.ctx!;
    const buf = ctx.createBuffer(2, ctx.sampleRate * seconds, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      let b0 = 0;
      let b1 = 0;
      let b2 = 0;
      for (let i = 0; i < d.length; i++) {
        const w = Math.random() * 2 - 1;
        b0 = 0.997 * b0 + w * 0.029;
        b1 = 0.985 * b1 + w * 0.032;
        b2 = 0.95 * b2 + w * 0.048;
        d[i] = (b0 + b1 + b2) * 0.5;
      }
    }
    return buf;
  }

  private impulse(seconds: number): AudioBuffer {
    const ctx = this.ctx!;
    const len = ctx.sampleRate * seconds;
    const buf = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2.6);
    }
    return buf;
  }

  private drone(notes: number[], level: number): void {
    const ctx = this.ctx!;
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 600;
    const gain = ctx.createGain();
    gain.gain.value = 0;
    gain.gain.linearRampToValueAtTime(level, ctx.currentTime + 6);
    filter.connect(gain);
    gain.connect(this.master);
    gain.connect(this.reverb);
    for (const n of notes) {
      for (const detune of [-6, 6]) {
        const osc = ctx.createOscillator();
        osc.type = "sawtooth";
        osc.frequency.value = midi(n);
        osc.detune.value = detune;
        osc.connect(filter);
        osc.start();
      }
    }
    // Slow breathing on the filter.
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.05;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = 250;
    lfo.connect(lfoGain).connect(filter.frequency);
    lfo.start();
  }

  private bell(note: number, when: number, level: number): void {
    const ctx = this.ctx!;
    const out = ctx.createGain();
    out.gain.setValueAtTime(0, when);
    out.gain.linearRampToValueAtTime(level, when + 0.01);
    out.gain.exponentialRampToValueAtTime(0.0001, when + 3.2);
    out.connect(this.master);
    out.connect(this.reverb);
    // Slightly inharmonic partials give a glassy bell.
    const partials: [number, number][] = [[1, 1], [2.01, 0.35], [3.98, 0.12], [5.4, 0.05]];
    for (const [ratio, amp] of partials) {
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.value = midi(note) * ratio;
      const g = ctx.createGain();
      g.gain.value = amp;
      osc.connect(g).connect(out);
      osc.start(when);
      osc.stop(when + 3.3);
    }
  }

  /** A note per bloom, walking up and down the pentatonic scale. */
  bloom(): void {
    if (!this.ctx) return;
    const step = Math.random() < 0.7 ? 1 : -1;
    this.noteIndex = (this.noteIndex + step + this.scale.length) % this.scale.length;
    this.bell(this.scale[this.noteIndex], this.ctx.currentTime, 0.16);
  }

  /**
   * A lantern lit. The chain plays a melody in D major pentatonic, phrased in fours like the
   * lanterns, lifting a step each time it comes round; harmony grows with the chain (a fifth
   * from 4, an octave shimmer from 8). Notes are snapped to a 75 bpm eighth-note grid, so a
   * slightly uneven flight still sounds in time. A fresh chain starts with a soft low note.
   */
  lantern(chain: number): void {
    if (!this.ctx) return;
    const steps = [0, 2, 4, 7, 9];
    // Scale degrees: rise, answer, climb, resolve.
    const melody = [0, 2, 4, 2, 1, 3, 5, 4, 2, 4, 6, 5, 4, 3, 2, 5];
    const i = chain - 1;
    const degree = melody[i % melody.length] + Math.min(Math.floor(i / melody.length), 3);
    const note = 62 + Math.floor(degree / 5) * 12 + steps[degree % 5];
    const grid = 60 / 75 / 2;
    const now = this.ctx.currentTime;
    const t = Math.ceil(now / grid - 0.25) * grid;
    const when = Math.max(now, t);
    const level = chain === 1 ? 0.08 : 0.1 + Math.min(chain, 16) * 0.006;
    this.bell(note, when, level);
    if (chain >= 4) this.bell(note - 5, when, level * 0.4);
    if (chain >= 8) this.bell(note + 12, when + grid * 0.5, level * 0.3);
    // Every completed phrase of four lands with a soft low root.
    if (chain >= 4 && chain % 4 === 0) this.bell(50 + Math.min(Math.floor(i / 16), 2) * 2, when, level * 0.6);
  }

  /** Every lantern lit in one run: a long rising arpeggio over a warm chord. */
  lanternsComplete(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.3;
    const run = [62, 64, 66, 69, 71, 74, 76, 78, 81, 83, 86];
    run.forEach((n, i) => this.bell(n, t + i * 0.13, 0.13));
    [50, 57, 62, 66].forEach((n) => this.bell(n, t + run.length * 0.13, 0.12));
  }

  /** Arpeggiated chord when an entire cluster has bloomed. */
  cluster(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.15;
    [62, 66, 69, 74, 78].forEach((n, i) => this.bell(n + 12, t + i * 0.11, 0.12));
  }

  /** Called each frame with speed in [0, 1]. */
  update(speed: number, altitude: number): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const level = 0.08 + speed * 0.32 - Math.min(altitude / 120, 0.06);
    this.windGain.gain.setTargetAtTime(Math.max(0.02, level), t, 0.3);
    this.windFilter.frequency.setTargetAtTime(380 + speed * 1300, t, 0.4);
  }
}

function midi(n: number): number {
  return 440 * Math.pow(2, (n - 69) / 12);
}
