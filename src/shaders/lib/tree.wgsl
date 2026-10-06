// Shared tree instance layout and wind deformation (meshes and impostors must sway alike).

export struct TreeInstance {
  // xyz = root position, w = uniform scale
  root: vec4f,
  // xy = (cos, sin) of yaw, z = seed, w = unused
  rot: vec4f,
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
