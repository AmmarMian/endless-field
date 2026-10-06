/**
 * A soft felt piano, synthesized: a few stretched (inharmonic) string partials, each pair of
 * strings slightly detuned so they beat, upper partials dying faster than the fundamental, a
 * muffled hammer thump, and a felt-damped tone that dulls as the note rings.
 */
function hz(n: number): number {
  return 440 * Math.pow(2, (n - 69) / 12);
}

/** Inharmonicity of the strings (stretched upper partials, the "piano" colour). */
const B = 0.00035;
const PARTIALS = 6;

let noise: AudioBuffer | null = null;

function hammerNoise(ctx: BaseAudioContext): AudioBuffer {
  if (noise && noise.sampleRate === ctx.sampleRate) return noise;
  const len = Math.floor(ctx.sampleRate * 0.06);
  noise = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = noise.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
  return noise;
}

/**
 * Plays MIDI note `n` at time `t` into `out`. `velocity` 0..1 sets loudness and brightness.
 * Returns when the note will have died away (seconds, absolute).
 */
export function feltPiano(ctx: BaseAudioContext, out: AudioNode, n: number, t: number, velocity: number, level = 1): number {
  const f0 = hz(n);
  // Low notes ring for long, high ones are short.
  const sustain = Math.min(7, Math.max(1.2, 4.2 * Math.pow(262 / f0, 0.55)));
  const end = t + sustain * 1.6;

  // Felt: a low-pass that starts a little open and closes as the note rings.
  const tone = ctx.createBiquadFilter();
  tone.type = "lowpass";
  tone.Q.value = 0.4;
  const open = Math.min(5000, f0 * 3 + 500 + velocity * 1400);
  tone.frequency.setValueAtTime(open, t);
  tone.frequency.exponentialRampToValueAtTime(Math.max(f0 * 1.5, 350), t + sustain * 0.6);
  const amp = ctx.createGain();
  amp.gain.value = level * (0.25 + 0.75 * velocity) * 0.5;
  tone.connect(amp).connect(out);

  for (let k = 1; k <= PARTIALS; k++) {
    const fk = k * f0 * Math.sqrt(1 + B * k * k);
    if (fk > 9000) break;
    const a = Math.pow(k, -1.4) * (k === 1 ? 1 : 0.6 + velocity * 0.4);
    const tau = sustain / (1 + 0.7 * (k - 1));
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    // Felt hammers: a few milliseconds of attack, not a click.
    g.gain.linearRampToValueAtTime(a, t + 0.006 + 0.004 * (1 - velocity));
    // Two-stage decay: a quicker initial drop, then the long ring (the piano "prompt/after" sound).
    g.gain.setTargetAtTime(a * 0.45, t + 0.01, tau * 0.12);
    g.gain.setTargetAtTime(0, t + tau * 0.3, tau * 0.45);
    g.connect(tone);
    for (const cents of [-1.1, 1.3]) {
      const o = ctx.createOscillator();
      o.frequency.value = fk;
      o.detune.value = cents;
      const half = ctx.createGain();
      half.gain.value = 0.5;
      o.connect(half).connect(g);
      o.start(t);
      o.stop(end);
    }
  }

  // The hammer and the wooden body: a soft, low thump.
  const thump = ctx.createBufferSource();
  thump.buffer = hammerNoise(ctx);
  const lp = ctx.createBiquadFilter();
  lp.type = "lowpass";
  lp.frequency.value = 500 + f0 * 0.5;
  const tg = ctx.createGain();
  tg.gain.value = 0.18 * velocity;
  thump.connect(lp).connect(tg).connect(tone);
  thump.start(t);
  return end;
}
