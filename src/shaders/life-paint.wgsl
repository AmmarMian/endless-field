// Paints restored "life" into the toroidal world map around a bloomed flower. Cells carry
// the world key they belong to, so stale data from wrapped-around regions reads as zero.
import { LifeCell, lifeIndex, lifeKey } from "./lib/field.wgsl";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";

struct Splat {
  center: vec2f,
  radius: f32,
  strength: f32,
}

@group(0) @binding(0) var<uniform> splat: Splat;
@group(0) @binding(1) var<storage, read_write> life: array<LifeCell>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  let span = i32(ceil(splat.radius * 1.8)) * 2 + 1;
  if (i32(id.x) >= span || i32(id.y) >= span) {
    return;
  }
  let origin = vec2i(floor(splat.center)) - vec2i(span / 2);
  let cell = origin + vec2i(id.xy);
  let p = vec2f(cell) + 0.5;
  // Organic edge: domain-warp the point, then wobble the radius at two scales so the
  // restored area grows as ragged tongues rather than an oval.
  let warp = vec2f(simplex2d(p * 0.045 + vec2f(13.0, 7.0)), simplex2d(p * 0.045 + vec2f(-5.0, 31.0))) * splat.radius * 0.45;
  let d = distance(p + warp, splat.center);
  let wobble = 1.0 + 0.3 * simplex2d(p * 0.07) + 0.15 * simplex2d(p * 0.23);
  let r = splat.radius * wobble;
  let v = splat.strength * (1.0 - smoothstep(r * 0.25, r, d));
  if (v <= 0.0) {
    return;
  }
  let i = lifeIndex(cell);
  let k = lifeKey(cell);
  var c = life[i];
  if (c.key != k) {
    c.life = 0.0;
    c.key = k;
  }
  c.life = max(c.life, min(v, 1.0));
  life[i] = c;
}
