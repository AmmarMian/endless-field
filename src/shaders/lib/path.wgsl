// The lantern path (mirrored in src/world/lantern-path.ts): a gravel path from the spawn
// meadow to the forest edge, lined with stone lanterns that light it warmly at night.

// Placement comes from the world seed (set by the host; defaults are seed 0).
override PATH_X0: f32 = 30.0;
override PATH_X1: f32 = 470.0;
override PATH_ZB: f32 = 70.0;
override PATH_P1: f32 = 0.4;
override PATH_P2: f32 = 1.7;
export const PATH_WIDTH = 1.3;
export fn pathZ(x: f32) -> f32 {
  return PATH_ZB + 16.0 * sin(x / 68.0 + PATH_P1) + 7.0 * sin(x / 29.0 + PATH_P2);
}

fn pathSlope(x: f32) -> f32 {
  return (16.0 / 68.0) * cos(x / 68.0 + PATH_P1) + (7.0 / 29.0) * cos(x / 29.0 + PATH_P2);
}

fn pathNormal(x: f32) -> vec2f {
  let s = pathSlope(x);
  return vec2f(-s, 1.0) / sqrt(1.0 + s * s);
}

// Distance from the path centerline (meters; large away from the path).
export fn pathDistance(xz: vec2f) -> f32 {
  if (xz.x < PATH_X0 - 6.0 || xz.x > PATH_X1 + 6.0) {
    return 1e9;
  }
  let s = pathSlope(xz.x);
  let d = abs(xz.y - pathZ(xz.x)) / sqrt(1.0 + s * s);
  let over = max(max(PATH_X0 - xz.x, xz.x - PATH_X1), 0.0);
  return length(vec2f(d, over));
}

// Lantern light is gathered by each shader (it needs the Globals arrays, which a library
// cannot bind, and passing them by value costs a copy per vertex):
//   let k0 = lanternFirst(p, G.lampPos[1].w, G.lampPos[2].w, G.night);
//   for k in k0 .. k0 + 4 (within count = G.lampPos[0].w):
//     sum += lanternTerm(p, G.lampPos[k].xyz, lamp brightness k, k, G.time);
//   light = LANTERN_COLOR * sum * G.night

export const LANTERN_COLOR = vec3f(1.0, 0.58, 0.24);

// First of the five lanterns to check around `p` (sorted by x along the path), or -100 when
// nothing can be lit here (daytime, or far from the path).
export fn lanternFirst(p: vec3f, firstX: f32, meanDx: f32, night: f32) -> i32 {
  if (night < 0.01 || p.x < PATH_X0 - 15.0 || p.x > PATH_X1 + 15.0 || abs(p.z - pathZ(p.x)) > 16.0) {
    return -100;
  }
  return i32(round((p.x - firstX) / max(meanDx, 1.0))) - 2;
}

// One lantern's light at `p`: soft falloff over a few meters, a candle's flicker.
export fn lanternTerm(p: vec3f, lp: vec3f, lamp: f32, k: f32, t: f32) -> f32 {
  if (lamp <= 0.001) {
    return 0.0;
  }
  let d = distance(p.xz, lp.xz);
  let flicker = 0.86 + 0.08 * sin(t * 7.3 + k * 1.7) + 0.06 * sin(t * 13.1 + k * 4.3);
  return lamp * flicker * 2.2 / (1.0 + d * d * 0.8) * (1.0 - smoothstep(5.0, 10.0, d));
}
