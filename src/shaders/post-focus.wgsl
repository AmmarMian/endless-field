// Autofocus: measures what is in the middle of the frame and eases the focus distance toward
// it. Drawn into a 1x1 target with alpha blending, so the target itself is the smoothed state.
struct Params {
  near: f32,
  rate: f32,
  pad0: f32,
  pad1: f32,
}

@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(depthTex));
  // A 9x9 grid over the central fifth of the frame, weighted toward the very middle. Samples
  // closer than ~2.5 m (a blade of grass across the lens) count little, so focus does not
  // twitch onto every stalk the wind skims past.
  var sum = 0.0;
  var wsum = 0.0;
  for (var j = -4; j <= 4; j++) {
    for (var i = -4; i <= 4; i++) {
      let o = vec2f(f32(i), f32(j)) / 4.0;
      let p = (vec2f(0.5) + o * vec2f(0.1, 0.12)) * size;
      let d = textureLoad(depthTex, vec2i(p), 0).r;
      let dist = clamp(params.near / max(d, 1e-5), 0.5, 400.0);
      let w = exp(-dot(o, o) * 1.5) * mix(0.08, 1.0, smoothstep(1.5, 3.0, dist));
      sum += log(dist) * w;
      wsum += w;
    }
  }
  return vec4f(sum / wsum, 0.0, 0.0, params.rate);
}
