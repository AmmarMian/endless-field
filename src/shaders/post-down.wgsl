// 13-tap downsample (Jimenez, "Next Generation Post Processing in Call of Duty").
// With `threshold > 0` it also extracts the bright part of the HDR scene (first level).
struct Params {
  texel: vec2f,
  threshold: f32,
  knee: f32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

fn tap(uv: vec2f, o: vec2f) -> vec3f {
  return textureSampleLevel(src, samp, uv + o * params.texel, 0.0).rgb;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let a = tap(uv, vec2f(-2.0, -2.0));
  let b = tap(uv, vec2f(0.0, -2.0));
  let c = tap(uv, vec2f(2.0, -2.0));
  let d = tap(uv, vec2f(-2.0, 0.0));
  let e = tap(uv, vec2f(0.0, 0.0));
  let f = tap(uv, vec2f(2.0, 0.0));
  let g = tap(uv, vec2f(-2.0, 2.0));
  let h = tap(uv, vec2f(0.0, 2.0));
  let i = tap(uv, vec2f(2.0, 2.0));
  let j = tap(uv, vec2f(-1.0, -1.0));
  let k = tap(uv, vec2f(1.0, -1.0));
  let l = tap(uv, vec2f(-1.0, 1.0));
  let m = tap(uv, vec2f(1.0, 1.0));
  var col = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  if (params.threshold > 0.0) {
    // Soft-knee threshold so highlights bloom smoothly instead of popping.
    let br = max(col.r, max(col.g, col.b));
    var soft = clamp(br - params.threshold + params.knee, 0.0, 2.0 * params.knee);
    soft = soft * soft / (4.0 * params.knee + 1e-4);
    let contrib = max(soft, br - params.threshold) / max(br, 1e-4);
    col = min(col * contrib, vec3f(64.0));
  }
  return vec4f(col, 1.0);
}
