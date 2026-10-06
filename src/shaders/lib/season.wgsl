// The season loop: Globals.season in [0, 4) runs summer (0) -> autumn (1) -> winter (2) ->
// spring (3) -> summer. Everything blends between neighbouring seasons by these weights.

// (summer, autumn, winter, spring), summing to 1.
export fn seasonWeights(s: f32) -> vec4f {
  var w = vec4f(0.0);
  for (var i = 0; i < 4; i = i + 1) {
    // Circular distance from season i, in [0, 2].
    let d = abs(fract((s - f32(i)) / 4.0 + 0.5) * 4.0 - 2.0);
    w[i] = max(0.0, 1.0 - d);
  }
  return w;
}

export const SNOW = vec3f(0.86, 0.89, 0.95);

// Meadow grass through the year: summer gold (as authored), autumn russet, winter frosted
// straw with white tips, spring fresh green. `t` is the height along the blade (0 root, 1 tip).
export fn seasonGrass(col: vec3f, w: vec4f, t: f32) -> vec3f {
  let lum = dot(col, vec3f(0.3, 0.59, 0.11));
  let autumn = mix(col, vec3f(0.62, 0.32, 0.12) * lum * 2.3, 0.6);
  let straw = vec3f(0.5, 0.47, 0.4) * lum * 1.7 + vec3f(0.015);
  let winter = mix(straw, SNOW * 0.9, smoothstep(0.55, 1.0, t) * 0.7);
  let spring = mix(col, vec3f(0.3, 0.6, 0.12) * lum * 2.4, 0.8);
  return col * w.x + autumn * w.y + winter * w.z + spring * w.w;
}

// Broadleaf foliage in spring: fresh, light green.
export fn springLeaf(albedo: vec3f) -> vec3f {
  let lum = dot(albedo, vec3f(0.3, 0.59, 0.11));
  return vec3f(0.42, 0.68, 0.18) * lum * 2.2;
}

// Stable per-feature random in [0, 1) (leaf thinning in winter, petals in spring).
export fn seasonHash(x: f32) -> f32 {
  return fract(sin(x * 12.9898 + 4.1414) * 43758.5453);
}
