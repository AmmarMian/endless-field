// The lantern path (mirrored in src/world/lantern-path.ts): a gravel path from the spawn
// meadow to the forest edge, lined with stone lanterns that light it warmly at night.

const PATH_X0 = 30.0;
const PATH_X1 = 470.0;
export const PATH_WIDTH = 1.3;
const LANTERN_X0 = 40.0;
const LANTERN_STEP = 10.0;
const LANTERN_OFFSET = 2.3;
const LANTERN_LAST = 43.0;

export fn pathZ(x: f32) -> f32 {
  return 70.0 + 16.0 * sin(x / 68.0 + 0.4) + 7.0 * sin(x / 29.0 + 1.7);
}

fn pathSlope(x: f32) -> f32 {
  return (16.0 / 68.0) * cos(x / 68.0 + 0.4) + (7.0 / 29.0) * cos(x / 29.0 + 1.7);
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

// Position of lantern k (alternating sides of the path).
fn lanternPos(k: f32) -> vec2f {
  let x = LANTERN_X0 + k * LANTERN_STEP;
  let side = select(-1.0, 1.0, (i32(k) % 2) == 0);
  return vec2f(x, pathZ(x)) + pathNormal(x) * side * LANTERN_OFFSET;
}

// Warm light from the nearest lanterns at a point (only at night). Each lantern flickers a
// little like a candle behind paper.
// `lamps` holds each lantern's brightness (Globals.lamps): only lit lanterns light the path.
export fn lanternLight(p: vec3f, night: f32, t: f32, lamps: array<vec4f, 12>) -> vec3f {
  if (p.x < PATH_X0 - 15.0 || p.x > PATH_X1 + 15.0 || abs(p.z - pathZ(p.x)) > 16.0) {
    return vec3f(0.0);
  }
  let k0 = round((p.x - LANTERN_X0) / LANTERN_STEP);
  var light = vec3f(0.0);
  for (var i = -1; i <= 1; i = i + 1) {
    let k = k0 + f32(i);
    if (k < 0.0 || k > LANTERN_LAST) {
      continue;
    }
    let ki = u32(k);
    let lamp = lamps[ki / 4u][ki % 4u];
    if (lamp <= 0.001) {
      continue;
    }
    let d = distance(p.xz, lanternPos(k));
    let flicker = 0.86 + 0.08 * sin(t * 7.3 + k * 1.7) + 0.06 * sin(t * 13.1 + k * 4.3);
    light = light + lamp * flicker * 2.4 / (1.0 + d * d * 0.8) * (1.0 - smoothstep(5.0, 10.0, d));
  }
  // The pool of light only shows at night.
  return vec3f(1.0, 0.58, 0.24) * light * night;
}
