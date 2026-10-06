// Depth of field: a wide-aperture lens focused on the middle of the frame. Gather blur where
// each tap spreads by its own circle of confusion (so blurred foreground bleeds over sharp
// background), but background never bleeds over sharper things in front of it.
struct Params {
  near: f32,
  /** Circle of confusion (px) per diopter of defocus. */
  aperture: f32,
  /** Largest circle of confusion (px radius). */
  maxCoc: f32,
  pad: f32,
}

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var depthTex: texture_2d<f32>;
@group(0) @binding(2) var focusTex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<uniform> params: Params;

const TAPS = 40;

fn cocAt(d: f32, invFocus: f32) -> f32 {
  // Reversed-Z depth is near / distance, so 1 / distance is linear in it.
  let invDist = d / params.near;
  return min(params.aperture * abs(invDist - invFocus), params.maxCoc);
}

@fragment
fn fs_main(@location(0) uv: vec2f, @builtin(position) frag: vec4f) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(scene));
  let px = vec2i(frag.xy);
  let focus = exp(textureLoad(focusTex, vec2i(0), 0).r);
  let invFocus = 1.0 / max(focus, 0.3);
  let dC = textureLoad(depthTex, px, 0).r;
  let cocC = cocAt(dC, invFocus);
  var acc = textureLoad(scene, px, 0).rgb;
  var wsum = 1.0;
  // Per-pixel rotation of the spiral turns banding into fine noise.
  let rot = fract(sin(dot(frag.xy, vec2f(12.9898, 78.233))) * 43758.5453) * 6.2831;
  for (var i = 0; i < TAPS; i++) {
    let fi = f32(i) + 0.5;
    let r = sqrt(fi / f32(TAPS)) * params.maxCoc;
    let a = fi * 2.39996 + rot;
    let q = uv + vec2f(cos(a), sin(a)) * r / size;
    let col = textureSampleLevel(scene, samp, q, 0.0).rgb;
    let d = textureSampleLevel(depthTex, samp, q, 0.0).r;
    let cocT = cocAt(d, invFocus);
    // The tap reaches this pixel if its own blur disc covers the distance.
    var w = clamp(cocT - r + 1.0, 0.0, 1.0);
    // Something behind this pixel cannot spread over it beyond this pixel's own blur.
    if (d < dC) {
      w = min(w, clamp(cocC - r + 1.0, 0.0, 1.0));
    }
    acc += col * w;
    wsum += w;
  }
  return vec4f(acc / wsum, 1.0);
}
