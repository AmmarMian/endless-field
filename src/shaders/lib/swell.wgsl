// Ocean swell: long Gerstner wave trains rolling in toward the coast (from the south, +z,
// travelling north), under the FFT chop. Analytic, so the CPU mirror (src/world/swell.ts)
// knows the same surface: the wind's wingtip, floating gulls and whales ride it.

const SWELL_N: u32 = 4u;
// Wavelength (m), amplitude (m), heading (radians from travelling north, -z), phase.
const SW_L = array<f32, 4>(110.0, 71.0, 43.0, 29.0);
const SW_A = array<f32, 4>(0.62, 0.38, 0.2, 0.11);
const SW_H = array<f32, 4>(0.12, -0.28, 0.45, -0.6);
const SW_P = array<f32, 4>(0.0, 1.7, 4.1, 2.6);
const STEEP: f32 = 0.55;

// x, z = horizontal shift, y = height; w = unused. `k` scales the whole swell (calm water).
export fn swell(xz: vec2f, t: f32, k: f32) -> vec3f {
  var d = vec3f(0.0);
  for (var i = 0u; i < SWELL_N; i = i + 1u) {
    let dir = vec2f(sin(SW_H[i]), -cos(SW_H[i]));
    let kw = 6.2831853 / SW_L[i];
    let w = sqrt(9.81 * kw);
    let th = kw * dot(dir, xz) - w * t + SW_P[i];
    let a = SW_A[i] * k;
    let q = STEEP / (kw * SW_A[i] * f32(SWELL_N));
    d += vec3f(dir.x * q * a * cos(th), a * sin(th), dir.y * q * a * cos(th));
  }
  return d;
}

// The swell's slope (d height / d xz), for normals.
export fn swellGrad(xz: vec2f, t: f32, k: f32) -> vec2f {
  var g = vec2f(0.0);
  for (var i = 0u; i < SWELL_N; i = i + 1u) {
    let dir = vec2f(sin(SW_H[i]), -cos(SW_H[i]));
    let kw = 6.2831853 / SW_L[i];
    let w = sqrt(9.81 * kw);
    let th = kw * dot(dir, xz) - w * t + SW_P[i];
    g += dir * (SW_A[i] * k * kw * cos(th));
  }
  return g;
}
