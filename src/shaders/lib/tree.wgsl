// Shared tree instance layout and wind deformation (meshes and impostors must sway alike).

export struct TreeInstance {
  // xyz = root position, w = uniform scale
  root: vec4f,
  // xy = (cos, sin) of yaw, z = seed, w = LOD crossfade (see lodKeep)
  rot: vec4f,
  // x = forest amount (autumn color), y = light under the canopy; computed once on the CPU
  env: vec4f,
}

export fn rotateYaw(v: vec3f, cs: vec2f) -> vec3f {
  return vec3f(cs.x * v.x + cs.y * v.z, v.y, -cs.y * v.x + cs.x * v.z);
}

// Whole-tree sway: grows with height (flex) and follows the wind with slow gusts.
export fn treeSway(flex: f32, heightN: f32, windDir: vec2f, strength: f32, time: f32, seed: f32) -> vec3f {
  let slow = sin(time * 0.55 + seed * 13.0) * 0.6 + sin(time * 1.13 + seed * 5.0) * 0.4;
  let amount = strength * (0.35 + 0.25 * slow) * heightN * heightN;
  let branch = sin(time * 2.1 + seed * 29.0 + flex * 6.0) * 0.06 * flex * strength;
  return vec3f(windDir.x, 0.0, windDir.y) * (amount * 0.6 + branch) + vec3f(0.0, -amount * 0.05, 0.0);
}

// Autumn foliage: recolors leaf texels by luminance into gold / orange / crimson, chosen per
// tree by `seed`; `amount` is how deep into autumn the tree is (0 keeps it green).
export fn autumnLeaf(albedo: vec3f, seed: f32, amount: f32) -> vec3f {
  let lum = dot(albedo, vec3f(0.3, 0.59, 0.11));
  let pick = fract(seed * 7.31);
  var hue = vec3f(1.0, 0.62, 0.12);
  if (pick > 0.33) { hue = vec3f(1.0, 0.36, 0.06); }
  if (pick > 0.66) { hue = vec3f(0.78, 0.12, 0.05); }
  if (pick > 0.9) { hue = vec3f(0.85, 0.75, 0.2); }
  // A few trees in each stand stay green, and color varies a little within a crown.
  let keep = step(0.86, fract(seed * 3.17));
  let autumn = hue * lum * 2.4;
  return mix(albedo, autumn, amount * (1.0 - keep));
}

// 4x4 Bayer threshold in [0, 1) from the pixel position (LOD crossfades).
export fn bayer4(p: vec2f) -> f32 {
  var m = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
  let i = u32(p.x) % 4u + (u32(p.y) % 4u) * 4u;
  return (m[i] + 0.5) / 16.0;
}

export fn lodKeep(w: f32, p: vec2f) -> bool {
  let d = bayer4(p);
  return select((d >= -w), (d < w), (w > 0.0));
}
