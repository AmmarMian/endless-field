// Shared grass-field data: blade instance layout, the "life" map that flowers restore,
// and the field palette used by both the blades and the distant ground so they match.

export struct Blade {
  // xyz = root position, w = height
  root: vec4f,
  // xy = facing (cos, sin), z = width, w = life (0 dry .. 1 restored)
  shape: vec4f,
  // xy = tip displacement in units of height, z = per-blade random,
  // w = meadow patch value in [0, 1) + 2 * kind (0 = blade, 1 = seed head) + 4 * hue variant
  bend: vec4f,
  // xyz = ground normal at the root (computed once per blade in the cull pass)
  ground: vec4f,
}

export struct LifeCell {
  life: f32,
  key: u32,
}

export const LIFE_SIZE: i32 = 512;

export fn lifeKey(cell: vec2i) -> u32 {
  return ((bitcast<u32>(cell.x) & 0xFFFFu) << 16u) | (bitcast<u32>(cell.y) & 0xFFFFu);
}

export fn lifeIndex(cell: vec2i) -> u32 {
  let w = vec2u(bitcast<vec2u>(cell) & vec2u(511u));
  return w.y * 512u + w.x;
}

// Palette: dry golden meadow (unrestored) to lush, saturated green (restored). `hue` in
// [-1, 1] pushes patches toward rust/red (negative) or cool sage green (positive).
export fn fieldColor(meadow: f32, life: f32, t: f32, hue: f32) -> vec3f {
  let dryBase = vec3f(0.030, 0.020, 0.008);
  var dryTip = mix(vec3f(0.40, 0.24, 0.07), vec3f(0.46, 0.34, 0.11), meadow);
  dryTip = mix(dryTip, vec3f(0.42, 0.16, 0.06), max(-hue, 0.0) * 0.5);
  dryTip = mix(dryTip, vec3f(0.30, 0.31, 0.14), max(hue, 0.0) * 0.7);
  let lushBase = vec3f(0.006, 0.024, 0.006);
  var lushTip = mix(vec3f(0.10, 0.32, 0.03), vec3f(0.26, 0.42, 0.05), meadow);
  lushTip = mix(lushTip, vec3f(0.30, 0.30, 0.04), max(-hue, 0.0) * 0.5);
  lushTip = mix(lushTip, vec3f(0.05, 0.30, 0.10), max(hue, 0.0) * 0.6);
  let base = mix(dryBase, lushBase, life);
  let tip = mix(dryTip, lushTip, life);
  return mix(base, tip, smoothstep(0.0, 1.0, pow(t, 0.75)));
}

// Field structure shared by placement and the distant ground: x = tall meadow amount,
// y = short lawn amount, z = color hue variant in [-1, 1].
export fn fieldKind(xz: vec2f, n0: f32, n1: f32) -> vec3f {
  let tall = smoothstep(0.18, 0.5, n0);
  let lawn = smoothstep(0.25, 0.55, -n0);
  let hue = clamp(n1 * 1.6, -1.0, 1.0) * smoothstep(0.15, 0.45, abs(n1));
  return vec3f(tall, lawn, hue);
}
