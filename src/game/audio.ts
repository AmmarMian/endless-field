import { Music } from "./music";
import { feltPiano } from "./piano";
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
    // Everything passes a low-pass that closes under water (muffled, as heard when diving).
    this.muffle = ctx.createBiquadFilter();
    this.muffle.type = "lowpass";
    this.muffle.frequency.value = 20000;
    this.master.connect(this.muffle).connect(ctx.destination);

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
  private muffle!: BiquadFilterNode;

  /** Under water (0..1): sounds dull and close. */
  setUnderwater(k: number): void {
    if (!this.ctx) return;
    this.muffle.frequency.setTargetAtTime(20000 * Math.pow(0.02, k), this.ctx.currentTime, 0.08);
  }
  private rainGain: GainNode | null = null;
  private patterGain: GainNode | null = null;
  private patterFilter: BiquadFilterNode | null = null;

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
      this.patterFilter = hp;
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

  /** A felt piano note (blooms, lanterns), into the dry mix and the room. */
  private bell(note: number, when: number, level: number): void {
    const out = this.ctx!.createGain();
    out.gain.value = Math.min(1, level * 2.2);
    out.connect(this.master);
    out.connect(this.reverb);
    feltPiano(this.ctx!, out, note, when, Math.min(1, 0.35 + level * 2.5));
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

  /**
   * A whole bed of flowers opened and its petals swirl up: a rising sweep of soft piano over a
   * warm chord, with a glassy shimmer that blooms on top and a breath of air.
   */
  petalBank(): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t = ctx.currentTime + 0.05;
    const run = [62, 66, 69, 74, 78, 81, 86, 90];
    run.forEach((n, i) => {
      this.bell(n, t + i * 0.07, 0.08 + i * 0.006);
      this.glass(n + 12, t + i * 0.07 + 0.02, 0.008, 1.4);
    });
    [50, 57, 62, 66, 69].forEach((n) => this.bell(n, t + 0.05, 0.07));
    this.glass(93, t + run.length * 0.07, 0.016, 3);
    // The whoosh of petals lifting.
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuffer(2);
    const f = ctx.createBiquadFilter();
    f.type = "bandpass";
    f.frequency.setValueAtTime(500, t);
    f.frequency.exponentialRampToValueAtTime(3200, t + 0.9);
    f.Q.value = 0.7;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.06, t + 0.25);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 1.4);
    src.connect(f).connect(g).connect(this.master);
    g.connect(this.reverb);
    src.start(t);
    src.stop(t + 1.5);
  }

  /** Arpeggiated chord when an entire cluster has bloomed. */
  cluster(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.15;
    [62, 66, 69, 74, 78].forEach((n, i) => this.bell(n + 12, t + i * 0.11, 0.12));
  }

  private surface: { grass: GainNode; leaves: GainNode; water: GainNode } | null = null;

  /** A looping noise bed through a filter, flickered by a random-ish amplitude flutter. */
  private bed(type: BiquadFilterType, freq: number, q: number, flutterHz: number, flutterDepth: number): GainNode {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuffer(3 + Math.random() * 2);
    src.loop = true;
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    const flutter = ctx.createGain();
    flutter.gain.value = 1 - flutterDepth;
    for (const hz of [flutterHz, flutterHz * 1.37, flutterHz * 0.61]) {
      const lfo = ctx.createOscillator();
      lfo.frequency.value = hz;
      const amt = ctx.createGain();
      amt.gain.value = flutterDepth / 3;
      lfo.connect(amt).connect(flutter.gain);
      lfo.start();
    }
    const out = ctx.createGain();
    out.gain.value = 0;
    src.connect(f).connect(flutter).connect(out).connect(this.master);
    out.connect(this.reverb);
    src.start();
    return out;
  }

  /**
   * The land answers the wind: grass hisses as it is combed, leaves rustle when the wind runs
   * through a canopy, water chatters under it. Each in [0, 1], scaled by speed and closeness.
   */
  setSurroundings(grass: number, leaves: number, water: number, speed: number): void {
    if (!this.ctx) return;
    if (!this.surface) {
      this.surface = {
        grass: this.bed("bandpass", 2600, 0.6, 3.1, 0.6),
        leaves: this.bed("highpass", 4200, 0.5, 11, 0.9),
        water: this.bed("bandpass", 700, 1.2, 5.3, 0.8),
      };
    }
    const t = this.ctx.currentTime;
    const k = 0.35 + speed * 0.65;
    this.surface.grass.gain.setTargetAtTime(grass * k * 0.09, t, 0.25);
    this.surface.leaves.gain.setTargetAtTime(leaves * k * 0.08, t, 0.2);
    this.surface.water.gain.setTargetAtTime(water * k * 0.1, t, 0.4);
  }

  /** A flock bursting up: a flurry of wingbeats and a few alarmed chirps, panned to the flock. */
  birds(n: number, pan: number, distance: number): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t0 = ctx.currentTime;
    const near = Math.max(0.15, Math.min(1, 12 / Math.max(distance, 1)));
    const panner = ctx.createStereoPanner();
    panner.pan.value = Math.max(-1, Math.min(1, pan));
    panner.connect(this.master);
    // Wingbeats: short filtered noise bursts at ~14 Hz per bird, staggered.
    const buf = this.noiseBuffer(1);
    for (let i = 0; i < n; i++) {
      const start = t0 + Math.random() * 0.35;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const f = ctx.createBiquadFilter();
      f.type = "bandpass";
      f.frequency.value = 900 + Math.random() * 700;
      f.Q.value = 0.8;
      const g = ctx.createGain();
      g.gain.value = 0;
      const beats = 10 + Math.floor(Math.random() * 6);
      for (let k = 0; k < beats; k++) {
        const bt = start + k / (13 + Math.random() * 3);
        const amp = 0.09 * near * (1 - k / beats);
        g.gain.setValueAtTime(0, bt);
        g.gain.linearRampToValueAtTime(amp, bt + 0.012);
        g.gain.exponentialRampToValueAtTime(0.0001, bt + 0.06);
      }
      src.connect(f).connect(g).connect(panner);
      src.start(start, Math.random() * 0.2);
      src.stop(start + 1.3);
    }
    // A few chirps: quick downward sine sweeps.
    for (let i = 0; i < 3; i++) {
      const ct = t0 + 0.05 + Math.random() * 0.5;
      const osc = ctx.createOscillator();
      const base = 3800 + Math.random() * 1500;
      osc.frequency.setValueAtTime(base, ct);
      osc.frequency.exponentialRampToValueAtTime(base * 0.7, ct + 0.07);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, ct);
      g.gain.linearRampToValueAtTime(0.03 * near, ct + 0.008);
      g.gain.exponentialRampToValueAtTime(0.0001, ct + 0.09);
      osc.connect(g).connect(panner);
      g.connect(this.reverb);
      osc.start(ct);
      osc.stop(ct + 0.1);
    }
  }

  /** A sparrow's chirrup from a feeding flock, panned and distanced, so it can be found by ear. */
  chirp(pan: number, distance: number): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const near = Math.max(0, Math.min(1, 14 / Math.max(distance, 1))) ** 0.8;
    if (near < 0.05) return;
    const p = ctx.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, pan)) * 0.85;
    // Farther birds sound duller.
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 3000 + near * 6000;
    p.connect(lp).connect(this.master);
    lp.connect(this.reverb);
    const notes = 2 + Math.floor(Math.random() * 3);
    let t = ctx.currentTime + Math.random() * 0.1;
    const base = 3600 + Math.random() * 900;
    for (let i = 0; i < notes; i++) {
      const o = ctx.createOscillator();
      // "Chirrup": a quick rise then a fall, wobbling fast.
      const f = base * (1 + (Math.random() - 0.5) * 0.12);
      o.frequency.setValueAtTime(f * 0.8, t);
      o.frequency.linearRampToValueAtTime(f * 1.15, t + 0.025);
      o.frequency.exponentialRampToValueAtTime(f * 0.7, t + 0.07);
      const fm = ctx.createOscillator();
      fm.frequency.value = 160 + Math.random() * 80;
      const fmAmt = ctx.createGain();
      fmAmt.gain.value = f * 0.05;
      fm.connect(fmAmt).connect(o.frequency);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.045 * near, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.08);
      o.connect(g).connect(p);
      o.start(t);
      fm.start(t);
      o.stop(t + 0.09);
      fm.stop(t + 0.09);
      t += 0.09 + Math.random() * 0.06;
    }
  }

  /** A soft, glassy sine shimmering above a piano note (the "magic" in the discovery sounds). */
  private glass(note: number, t: number, level: number, length = 2.5): void {
    const ctx = this.ctx!;
    const o = ctx.createOscillator();
    o.frequency.value = 440 * Math.pow(2, (note - 69) / 12);
    const vib = ctx.createOscillator();
    vib.frequency.value = 5.5;
    const va = ctx.createGain();
    va.gain.value = 3;
    vib.connect(va).connect(o.frequency);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(level, t + 0.04);
    g.gain.exponentialRampToValueAtTime(0.0001, t + length);
    o.connect(g);
    g.connect(this.reverb);
    g.connect(this.master);
    o.start(t);
    vib.start(t);
    o.stop(t + length + 0.1);
    vib.stop(t + length + 0.1);
  }

  /** Something new found: a quick rising arpeggio with a glassy shimmer on top. */
  /** An animal comes to trust the wind: a soft rising pair of bells. */
  friend(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.03;
    this.bell(74, t, 0.1);
    this.bell(81, t + 0.13, 0.1);
    this.glass(93, t + 0.13, 0.01, 1.6);
  }

  /** A fox springing at the swallow in play: a quick falling pluck. */
  pounce(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.02;
    this.bell(86, t, 0.06);
    this.bell(81, t + 0.07, 0.05);
  }

  discovery(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.05;
    const run = [69, 74, 78, 81, 86];
    run.forEach((n, i) => {
      this.bell(n, t + i * 0.085, 0.12);
      this.glass(n + 12, t + i * 0.085, 0.012, 1.2);
    });
    const end = t + run.length * 0.085 + 0.05;
    [62, 69, 78].forEach((n) => this.bell(n, end, 0.08));
    this.glass(90, end, 0.018, 3);
  }

  /** Entering a named place: a calm open chord, low, with a far-off sparkle. */
  region(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.1;
    [50, 57, 66, 69].forEach((n, i) => this.bell(n, t + i * 0.18, 0.07));
    this.glass(81, t + 0.8, 0.008, 3.5);
    this.glass(88, t + 1.1, 0.006, 3.5);
  }

  /** A glass wind bell (furin): a bright, slightly inharmonic ring, panned and distanced. */
  windBell(bell: number, strength: number, pan: number, distance: number): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const near = Math.max(0, Math.min(1, 10 / Math.max(distance, 1)));
    if (near < 0.04) return;
    const t = ctx.currentTime + Math.random() * 0.03;
    const p = ctx.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, pan)) * 0.8;
    p.connect(this.master);
    p.connect(this.reverb);
    const f0 = 440 * Math.pow(2, ([86, 90, 93][bell % 3] - 69) / 12);
    const level = 0.05 * near * (0.4 + 0.6 * strength);
    for (const [ratio, amp, decay] of [[1, 1, 2.2], [2.32, 0.5, 1.2], [4.25, 0.25, 0.6], [6.63, 0.12, 0.35]] as const) {
      const o = ctx.createOscillator();
      o.frequency.value = f0 * ratio * (1 + (Math.random() - 0.5) * 0.004);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(level * amp, t + 0.003);
      g.gain.exponentialRampToValueAtTime(0.0001, t + decay);
      o.connect(g).connect(p);
      o.start(t);
      o.stop(t + decay + 0.05);
    }
  }

  private whirr: { gain: GainNode; filter: BiquadFilterNode } | null = null;

  /** Paper pinwheels spinning nearby: a soft fluttering whirr (0 silent .. 1 spinning hard). */
  setWhirr(level: number): void {
    if (!this.ctx) return;
    if (!this.whirr) {
      const ctx = this.ctx;
      const src = ctx.createBufferSource();
      src.buffer = this.noiseBuffer(2);
      src.loop = true;
      const filter = ctx.createBiquadFilter();
      filter.type = "bandpass";
      filter.frequency.value = 1800;
      filter.Q.value = 1.2;
      const flutter = ctx.createGain();
      flutter.gain.value = 0.5;
      const lfo = ctx.createOscillator();
      lfo.type = "triangle";
      lfo.frequency.value = 24;
      const amt = ctx.createGain();
      amt.gain.value = 0.5;
      lfo.connect(amt).connect(flutter.gain);
      lfo.start();
      const gain = ctx.createGain();
      gain.gain.value = 0;
      src.connect(filter).connect(flutter).connect(gain).connect(this.master);
      src.start();
      this.whirr = { gain, filter };
    }
    const t = this.ctx.currentTime;
    this.whirr.gain.gain.setTargetAtTime(level * 0.06, t, 0.3);
    this.whirr.filter.frequency.setTargetAtTime(1200 + level * 1600, t, 0.3);
  }

  /** The kingfisher's call: a high, piercing double whistle. */
  kingfisher(pan: number, distance: number): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const near = Math.max(0.1, Math.min(1, 12 / Math.max(distance, 1)));
    const p = ctx.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, pan)) * 0.8;
    p.connect(this.master);
    p.connect(this.reverb);
    for (let i = 0; i < 2; i++) {
      const t = ctx.currentTime + i * 0.16;
      const o = ctx.createOscillator();
      o.frequency.setValueAtTime(6200, t);
      o.frequency.linearRampToValueAtTime(7000, t + 0.04);
      o.frequency.linearRampToValueAtTime(5600, t + 0.11);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.05 * near, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
      o.connect(g).connect(p);
      o.start(t);
      o.stop(t + 0.13);
    }
  }

  /** The kingfisher lost: three soft falling notes. */
  raceLost(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.05;
    [69, 66, 62].forEach((n, i) => this.bell(n, t + i * 0.22, 0.07));
  }

  /** A fish breaking the surface: a wet slap and a short spray hiss. */
  splash(size: number, pan: number, distance: number): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const near = Math.max(0, Math.min(1, 10 / Math.max(distance, 1)));
    if (near < 0.05) return;
    const t = ctx.currentTime;
    const p = ctx.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, pan)) * 0.8;
    p.connect(this.master);
    p.connect(this.reverb);
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuffer(1);
    const f = ctx.createBiquadFilter();
    f.type = "bandpass";
    f.frequency.setValueAtTime(900 + size * 400, t);
    f.frequency.exponentialRampToValueAtTime(2600, t + 0.25);
    f.Q.value = 0.8;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.16 * near * (0.5 + size), t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.45);
    src.connect(f).connect(g).connect(p);
    src.start(t);
    src.stop(t + 0.5);
    // The "plop": a quick falling tone as the water closes.
    const o = ctx.createOscillator();
    o.frequency.setValueAtTime(520 - size * 120, t + 0.02);
    o.frequency.exponentialRampToValueAtTime(180, t + 0.12);
    const og = ctx.createGain();
    og.gain.setValueAtTime(0, t + 0.02);
    og.gain.linearRampToValueAtTime(0.06 * near, t + 0.03);
    og.gain.exponentialRampToValueAtTime(0.0001, t + 0.15);
    o.connect(og).connect(p);
    o.start(t + 0.02);
    o.stop(t + 0.16);
  }

  /**
   * An insect caught: a tiny click and a bright note. Catches in quick succession climb the
   * pentatonic scale (`combo` 1, 2, 3...), so a good run through a swarm plays a phrase.
   */
  catchInsect(combo: number): void {
    if (!this.ctx) return;
    const steps = [0, 2, 4, 7, 9];
    const i = Math.min(combo - 1, 14);
    const note = 74 + Math.floor(i / 5) * 12 + steps[i % 5];
    const t = this.ctx.currentTime;
    this.glass(note, t, 0.03 + Math.min(combo, 10) * 0.002, 0.7);
    this.bell(note, t, 0.06);
  }

  /** Skimming the river (a soft swish and patter). */
  skim(): void {
    if (!this.ctx) return;
    this.splash(0.1, 0, 3);
  }

  /**
   * Riding a thermal: a slow rising phrase over a soft, swelling chord, as the air lifts you
   * (about as long as the climb).
   */
  thermal(): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t0 = ctx.currentTime + 0.05;
    // A warm pad: three soft sines swelling in and fading.
    const pad = ctx.createGain();
    pad.gain.setValueAtTime(0, t0);
    pad.gain.linearRampToValueAtTime(0.035, t0 + 2.5);
    pad.gain.linearRampToValueAtTime(0.03, t0 + 7);
    pad.gain.exponentialRampToValueAtTime(0.0001, t0 + 11);
    pad.connect(this.master);
    pad.connect(this.reverb);
    for (const n of [50, 57, 62, 66]) {
      const o = ctx.createOscillator();
      o.frequency.value = 440 * Math.pow(2, (n - 69) / 12);
      o.detune.value = (Math.random() - 0.5) * 8;
      o.connect(pad);
      o.start(t0);
      o.stop(t0 + 11.5);
    }
    // The phrase: up the scale, unhurried, glass on top of the piano.
    const rise = [62, 64, 66, 69, 71, 74, 76, 78, 81, 83, 86];
    rise.forEach((n, i) => {
      const t = t0 + 0.4 + i * 0.62;
      this.bell(n, t, 0.07);
      if (i % 2 === 0) this.glass(n + 12, t, 0.01, 1.6);
    });
  }

  /** Settling on a perch to rest: a slow, gentle phrase that lands softly. */
  rest(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.3;
    const phrase: [number, number, number][] = [[74, 0, 0.09], [69, 0.55, 0.07], [71, 1.0, 0.07], [66, 1.6, 0.08], [62, 2.4, 0.09]];
    for (const [n, dt, v] of phrase) this.bell(n, t + dt, v);
    [50, 57].forEach((n) => this.bell(n, t + 2.4, 0.06));
    this.glass(86, t + 2.5, 0.006, 3);
  }

  /** A rainbow appears: a soft, wide open chord with a glassy sparkle running up. */
  rainbow(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.2;
    [50, 57, 62, 66, 69, 74].forEach((n, i) => this.bell(n, t + i * 0.12, 0.06));
    [86, 90, 93, 98].forEach((n, i) => this.glass(n, t + 0.8 + i * 0.18, 0.007, 2.5));
  }

  /** A frog by the river at night: a short, throaty double croak, placed in the stereo field. */
  frog(pan: number, distance: number): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const near = Math.max(0, Math.min(1, 14 / Math.max(distance, 1)));
    if (near < 0.05) return;
    const p = ctx.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, pan)) * 0.85;
    const lp = ctx.createBiquadFilter();
    lp.type = "bandpass";
    lp.frequency.value = 600 + Math.random() * 300;
    lp.Q.value = 2.5;
    p.connect(this.master);
    p.connect(this.reverb);
    lp.connect(p);
    const f0 = 140 + Math.random() * 60;
    const croaks = Math.random() < 0.5 ? 2 : 3;
    for (let i = 0; i < croaks; i++) {
      const t = ctx.currentTime + 0.05 + i * 0.16;
      const o = ctx.createOscillator();
      o.type = "sawtooth";
      o.frequency.setValueAtTime(f0, t);
      o.frequency.linearRampToValueAtTime(f0 * 0.85, t + 0.1);
      // Rapid pulses within each croak (the "rrr").
      const am = ctx.createGain();
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 38;
      const amt = ctx.createGain();
      amt.gain.value = 0.5;
      lfo.connect(amt).connect(am.gain);
      am.gain.value = 0.5;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.05 * near, t + 0.015);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
      o.connect(am).connect(g).connect(lp);
      o.start(t);
      lfo.start(t);
      o.stop(t + 0.13);
      lfo.stop(t + 0.13);
    }
  }

  /** Rain falling on water nearby: a little bright plink (call often while raining). */
  plink(amount: number): void {
    if (!this.ctx || amount < 0.05) return;
    const ctx = this.ctx;
    const t = ctx.currentTime + Math.random() * 0.05;
    const o = ctx.createOscillator();
    const f = 1400 + Math.random() * 1800;
    o.frequency.setValueAtTime(f, t);
    o.frequency.exponentialRampToValueAtTime(f * 1.6, t + 0.04);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.012 * amount, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.07);
    const p = ctx.createStereoPanner();
    p.pan.value = Math.random() * 2 - 1;
    o.connect(g).connect(p).connect(this.master);
    o.start(t);
    o.stop(t + 0.08);
  }

  /** Under trees the rain is a softer, duller patter (0 open .. 1 under a canopy). */
  setRainShelter(leafy: number): void {
    if (!this.ctx || !this.patterGain) return;
    // The patter's filter: dull it under the leaves.
    const f = this.patterFilter;
    if (f) f.frequency.setTargetAtTime(3800 - leafy * 2600, this.ctx.currentTime, 0.5);
  }

  /** The swallow starts slicing the water: a little rising piano figure. */
  skimMotif(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + 0.05;
    [74, 78, 81, 86].forEach((n, i) => this.bell(n, t + i * 0.16, 0.06 - i * 0.008));
  }

  /** Holds every sound (pause). */
  setPaused(on: boolean): void {
    if (!this.ctx) return;
    void (on ? this.ctx.suspend() : this.ctx.resume());
  }

  /** Called each frame with speed in [0, 1]. */
  update(speed: number, altitude: number): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    // Resting or slow flight can report below 0: keep the filter in its range.
    speed = Math.min(1, Math.max(0, speed));
    const level = 0.08 + speed * 0.32 - Math.min(altitude / 120, 0.06);
    this.windGain.gain.setTargetAtTime(Math.max(0.02, level), t, 0.3);
    this.windFilter.frequency.setTargetAtTime(380 + speed * 1300, t, 0.4);
  }
}

function midi(n: number): number {
  return 440 * Math.pow(2, (n - 69) / 12);
}
