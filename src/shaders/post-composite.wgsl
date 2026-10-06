// Final image: bloom, exposure, filmic tone map, grade, vignette, sRGB encode + dither.
struct Params {
  bloomStrength: f32,
  exposure: f32,
  vignette: f32,
  time: f32,
  grade: vec4f,
}

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var shafts: texture_2d<f32>;

// Stephen Hill's fitted ACES.
fn aces(c: vec3f) -> vec3f {
  let m1 = mat3x3f(
    vec3f(0.59719, 0.07600, 0.02840),
    vec3f(0.35458, 0.90834, 0.13383),
    vec3f(0.04823, 0.01566, 0.83777));
  let m2 = mat3x3f(
    vec3f(1.60475, -0.10208, -0.00327),
    vec3f(-0.53108, 1.10813, -0.07276),
    vec3f(-0.07367, -0.00605, 1.07602));
  let v = m1 * c;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp(m2 * (a / b), vec3f(0.0), vec3f(1.0));
}

fn srgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(@location(0) uv: vec2f, @builtin(position) frag: vec4f) -> @location(0) vec4f {
  var col = textureSampleLevel(scene, samp, uv, 0.0).rgb;
  let b = textureSampleLevel(bloom, samp, uv, 0.0).rgb;
  col = col + b * params.bloomStrength + textureSampleLevel(shafts, samp, uv, 0.0).rgb;
  col = col * params.exposure;
  // Gentle warm lift in the shadows, Flower-like pastel response.
  let lum = dot(col, vec3f(0.2126, 0.7152, 0.0722));
  col = mix(vec3f(lum), col, params.grade.w);
  col = col + params.grade.xyz * (1.0 - smoothstep(0.0, 0.6, lum)) * 0.04;
  col = aces(col);
  let d = uv - 0.5;
  col = col * (1.0 - params.vignette * dot(d, d) * 1.6);
  var outc = srgb(col);
  outc = outc + (hash12(frag.xy + params.time * 61.0) - 0.5) / 255.0;
  return vec4f(outc, 1.0);
}
