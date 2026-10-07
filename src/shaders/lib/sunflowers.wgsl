// The sunflower field beyond the river (mirrored in src/world/sunflower-field.ts): a large
// planted field on graded farmland, rows running away from the river, with a ragged edge.

// Placement comes from the world seed (set by the host; defaults are seed 0).
override SF_CX: f32 = -62.0;
override SF_CZ: f32 = -400.0;
override SF_DX: f32 = 0.3902;
override SF_DZ: f32 = -0.9207;
const SF_HALF = vec2f(80.0, 50.0);
// Ground level the field is graded to (depends on the world seed; set by the host).
override SF_LEVEL: f32 = -22.1;
export const SF_ROW = 0.76;
// Across-row offset of the first drill row (plants sit on the ridges).
export const SF_ROW_PHASE = 55.0;

// (along the rows, across the rows) in meters from the field center.
export fn sunflowerLocal(xz: vec2f) -> vec2f {
  let d = xz - vec2f(SF_CX, SF_CZ);
  let dir = vec2f(SF_DX, SF_DZ);
  return vec2f(dot(d, dir), dot(d, vec2f(-dir.y, dir.x)));
}

fn sfEdgeDistance(l: vec2f) -> f32 {
  let q = abs(l) / SF_HALF;
  return pow(pow(q.x, 6.0) + pow(q.y, 6.0), 1.0 / 6.0) + 0.03 * sin(l.x * 0.11 + 1.3) + 0.03 * sin(l.y * 0.17);
}

// 1 inside the field, easing to 0 over its ragged edge.
export fn sunflowerField(xz: vec2f) -> f32 {
  let l = sunflowerLocal(xz);
  if (abs(l.x) > SF_HALF.x * 1.2 || abs(l.y) > SF_HALF.y * 1.2) {
    return 0.0;
  }
  return 1.0 - smoothstep(0.95, 1.0, sfEdgeDistance(l));
}

// Farmland is graded: pulls the ground toward the field level, easing out beyond the edge.
export fn sunflowerLevel(xz: vec2f, h: f32) -> f32 {
  let l = sunflowerLocal(xz);
  if (abs(l.x) > SF_HALF.x * 1.6 || abs(l.y) > SF_HALF.y * 1.8) {
    return h;
  }
  let w = (1.0 - smoothstep(0.9, 1.45, sfEdgeDistance(l))) * 0.85;
  return mix(h, SF_LEVEL, w);
}

// True within `margin` meters of the field's bounding box (keeps beds, trees and props out).
export fn sunflowerNear(xz: vec2f, margin: f32) -> bool {
  let l = abs(sunflowerLocal(xz));
  return l.x < SF_HALF.x + margin && l.y < SF_HALF.y + margin;
}
