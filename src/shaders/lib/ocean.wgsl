// The open sea's swell: a Phillips spectrum evolved in frequency space and brought back to
// space by an inverse FFT (adapted from the vgpu fft-ocean-surface example). Shared by the
// compute passes (ocean-*.wgsl) that fill the displacement texture the sea surface samples.
export const TWO_PI: f32 = 6.28318530718;
export const N: u32 = 256u;
export const LOG2N: u32 = 8u;
export const GRAVITY: f32 = 9.81;

export struct SimParams {
  windDir: vec2f,
  windSpeed: f32,
  amplitude: f32,
  patchSize: f32,
  time: f32,
  _pad: vec2f,
}

export fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

export fn bitrev(i: u32) -> u32 {
  return reverseBits(i) >> (32u - LOG2N);
}

// One 256-point inverse FFT (radix 2, in workgroup memory) over three fields at once.
export fn fftStages3(
  a: ptr<workgroup, array<vec2f, 256>>,
  b: ptr<workgroup, array<vec2f, 256>>,
  c: ptr<workgroup, array<vec2f, 256>>,
  lid: u32,
) {
  for (var s: u32 = 0u; s < LOG2N; s = s + 1u) {
    let half = 1u << s;
    let m = half << 1u;
    if (lid < 128u) {
      let k = lid & (half - 1u);
      let base = (lid >> s) << (s + 1u);
      let i0 = base + k;
      let i1 = i0 + half;
      let ang = TWO_PI * f32(k) / f32(m);
      let w = vec2f(cos(ang), sin(ang));
      let a0 = (*a)[i0];
      let a1 = cmul(w, (*a)[i1]);
      (*a)[i0] = a0 + a1;
      (*a)[i1] = a0 - a1;
      let b0 = (*b)[i0];
      let b1 = cmul(w, (*b)[i1]);
      (*b)[i0] = b0 + b1;
      (*b)[i1] = b0 - b1;
      let c0 = (*c)[i0];
      let c1 = cmul(w, (*c)[i1]);
      (*c)[i0] = c0 + c1;
      (*c)[i1] = c0 - c1;
    }
    workgroupBarrier();
  }
}
