// Tent-filtered upsample of the coarser level, added to the current level.
struct Params {
  texel: vec2f,
  radius: f32,
  pad: f32,
}

@group(0) @binding(0) var coarse: texture_2d<f32>;
@group(0) @binding(1) var fine: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> params: Params;

fn tap(uv: vec2f, o: vec2f) -> vec3f {
  return textureSampleLevel(coarse, samp, uv + o * params.texel * params.radius, 0.0).rgb;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  var col = tap(uv, vec2f(0.0, 0.0)) * 4.0;
  col = col + (tap(uv, vec2f(-1.0, 0.0)) + tap(uv, vec2f(1.0, 0.0)) + tap(uv, vec2f(0.0, -1.0)) + tap(uv, vec2f(0.0, 1.0))) * 2.0;
  col = col + tap(uv, vec2f(-1.0, -1.0)) + tap(uv, vec2f(1.0, -1.0)) + tap(uv, vec2f(-1.0, 1.0)) + tap(uv, vec2f(1.0, 1.0));
  col = col / 16.0;
  let base = textureSampleLevel(fine, samp, uv, 0.0).rgb;
  return vec4f(base + col, 1.0);
}
