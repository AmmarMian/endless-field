// Screen-space light shafts: radial blur of the bright-pass toward the sun's screen position.
// Bright sky seen through gaps in the canopy streaks into god rays; occluders stay dark.
struct Params {
  sunUv: vec2f,
  strength: f32,
  decay: f32,
}

@group(0) @binding(0) var bright: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  if (params.strength <= 0.0) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let steps = 48;
  let delta = (params.sunUv - uv) / f32(steps) * 0.9;
  var p = uv;
  var weight = 1.0;
  var acc = vec3f(0.0);
  for (var i = 0; i < steps; i = i + 1) {
    p = p + delta;
    let c = textureSampleLevel(bright, samp, clamp(p, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
    acc = acc + min(c, vec3f(4.0)) * weight;
    weight = weight * params.decay;
  }
  return vec4f(acc / f32(steps) * params.strength, 1.0);
}
