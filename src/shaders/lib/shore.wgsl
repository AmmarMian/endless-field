// The shore: waves coming in to the beach (shared by the sea surface and the wet sand) and
// the sand's look. Distances are coastInfo()'s: signed, inland positive, so `-s` is how far
// out to sea a point lies.
import { SEA_Y, gnoise } from "./terrain.wgsl";

// Seconds between breakers, and the swash's highest run-up above the still level (m).
export const SURF_PERIOD: f32 = 8.5;
export const RUNUP: f32 = 0.42;

fn hash1(n: f32) -> f32 {
  return fract(sin(n * 127.1 + 311.7) * 43758.5453);
}

// Phase of the incoming waves at signed distance `s`: crests are lines of equal phase, so
// they lie along the shore. Waves slow as the water shallows (c = sqrt(g h) over a sloping
// bed), so crests bunch up toward the beach; on the sand the swash runs up at a few m/s.
export fn surfPhase(s: f32, xz: vec2f, t: f32) -> f32 {
  let d = max(-s, 0.0);
  let travel = 2.9 * sqrt(d) + d * 0.012 - max(s, 0.0) * 0.28;
  // Crests are not ruler-straight: they bend a little along the shore and drift.
  let bend = gnoise(vec2f(xz.x / 140.0 + 3.0, xz.y / 140.0 - 7.0)) * 0.55 + gnoise(xz / 45.0) * 0.12;
  return (t + travel) / SURF_PERIOD + bend;
}

// The size of wave number `k` (sets of larger and smaller waves).
export fn surfSet(k: f32) -> f32 {
  return 0.55 + 0.45 * hash1(k) * (0.6 + 0.4 * sin(k * 0.9));
}

// Water height above SEA_Y from the surf at signed distance `s` with still-water depth
// `depth`; x = height, y = breaking (0..1: the crest is spilling), z = phase in the cycle
// (0 = crest just passed), w = bore foam (the turbulent water left behind a broken wave).
export fn surf(s: f32, depth: f32, xz: vec2f, t: f32, open: f32) -> vec4f {
  let ph = surfPhase(s, xz, t);
  let k = floor(ph);
  let u = ph - k;
  // The next wave's size takes over as it arrives, so nothing jumps where cycles meet.
  let size = mix(surfSet(k), surfSet(k + 1.0), smoothstep(0.75, 1.0, u)) * open;
  let d = max(-s, 0.0);
  // Shoaling: waves grow as they reach shallower water, until the depth limits them
  // (they break at about 0.8 of the depth) and run on as a foaming bore.
  let a0 = 0.32 * size * (1.0 + 0.8 * (1.0 - smoothstep(10.0, 90.0, d)));
  let limit = 0.55 * max(depth, 0.0) + 0.04;
  let a = min(a0, limit);
  let breaking = smoothstep(0.75, 1.0, a0 / max(limit, 1e-3)) * (1.0 - smoothstep(60.0, 110.0, d));
  // Profile over the cycle: a steep face ahead of the crest (u -> 1), a long back behind it.
  let front = smoothstep(0.82, 1.0, u);
  let back = exp(-u * 4.5);
  let eta = a * (max(front * front, back) - 0.28);
  // The swash: at the beach the water runs up and drains back once per wave, at a height
  // independent of the depth limit.
  let wash = RUNUP * size * (max(smoothstep(0.86, 1.0, u), exp(-u * 3.2) * (1.0 - u)) - 0.12);
  let nearShore = 1.0 - smoothstep(2.0, 22.0, d);
  let h = mix(eta, wash, nearShore);
  let bore = exp(-u * mix(4.5, 7.0, nearShore)) * smoothstep(0.2, 1.0, breaking + nearShore * 0.6) * open;
  return vec4f(h, breaking * open, u, bore);
}

// Highest the swash has recently reached above SEA_Y (the wet sand's upper edge).
export fn swashReach(xz: vec2f, t: f32, open: f32) -> f32 {
  let ph = surfPhase(0.0, xz, t);
  let k = floor(ph);
  // The last two waves' run-ups, the older one drying (blended so the edge never jumps).
  let r = mix(max(surfSet(k), surfSet(k - 1.0) * 0.85), max(surfSet(k + 1.0), surfSet(k) * 0.85), smoothstep(0.75, 1.0, ph - k));
  return (RUNUP * r * 0.88 + 0.03) * open + 0.05;
}

// How far inland the sand reaches (m from the waterline): far into the dunes, little on the
// rocky stretches; ragged where the grass takes over.
export fn sandEdge(xz: vec2f, dune: f32) -> f32 {
  return mix(32.0, 165.0, dune) + gnoise(xz / 23.0 + vec2f(9.0, 3.0)) * 16.0 + gnoise(xz / 70.0) * 22.0;
}

// Sand albedo: pale and fine above the tide, ribbed by ripples low on the beach and under
// the water, darkened where wet, with a wrack line of weed and shell bits at the high mark.
export fn sandColor(xz: vec2f, y: f32, s: f32, wet: f32, dist: f32) -> vec3f {
  let above = y - SEA_Y;
  let tone = gnoise(xz / 9.0 + vec2f(2.0, 4.0)) * 0.5 + 0.5;
  let grain = gnoise(xz * 3.7) * 0.5 + 0.5;
  var dry = mix(vec3f(0.5, 0.45, 0.35), vec3f(0.6, 0.54, 0.43), tone) * mix(0.94, 1.04, grain);
  // Dunes are paler (wind-sorted), and greyer where the field's soil mixes in.
  dry = mix(dry, dry * vec3f(1.06, 1.04, 1.0), smoothstep(40.0, 90.0, s));
  // Ripples low on the beach and under the water, running along the shore; faded with
  // distance before they could alias.
  let w = gnoise(xz / 30.0) * 6.0;
  let rip = sin((s + w) * 9.0 + gnoise(xz * 0.6) * 2.5) * 0.5 + 0.5;
  let ripAmt = (1.0 - smoothstep(0.2, 0.9, above)) * (1.0 - smoothstep(8.0, 30.0, dist));
  dry = dry * mix(1.0, mix(0.92, 1.04, rip), ripAmt);
  // Wrack: a band of dark weed and pale shell grit along the highest tide line.
  let wrackY = RUNUP + 0.32 + gnoise(xz / 18.0) * 0.08;
  let band = 1.0 - smoothstep(0.0, 0.1, abs(above - wrackY));
  let weed = smoothstep(0.62, 0.8, gnoise(xz * 3.1 + vec2f(-3.0, 1.0)) * 0.5 + 0.5) * smoothstep(0.3, 0.6, gnoise(xz * 0.4) * 0.5 + 0.5) * band;
  let shell = smoothstep(0.82, 0.92, gnoise(xz * 2.6 + vec2f(7.0, 5.0)) * 0.5 + 0.5) * smoothstep(-0.1, 0.4, above) * (1.0 - smoothstep(1.4, 2.4, above));
  dry = mix(dry, vec3f(0.09, 0.075, 0.04), weed * 0.7 * (1.0 - smoothstep(15.0, 50.0, dist)));
  dry = mix(dry, vec3f(0.72, 0.68, 0.6), shell * 0.7 * (1.0 - smoothstep(10.0, 40.0, dist)));
  let wetC = dry * vec3f(0.58, 0.58, 0.6);
  return mix(dry, wetC, wet);
}
