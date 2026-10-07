// Final image: bloom, exposure, filmic tone map, grade, vignette, sRGB encode + dither, and an
// optional screen style (0 none, 1 painterly, 2 watercolor, 3 film, 4 miniature, 5 ink,
// 6 cozy).
struct Params {
  bloomStrength: f32,
  exposure: f32,
  vignette: f32,
  time: f32,
  grade: vec4f,
  texel: vec2f,
  style: f32,
  aspect: f32,
  // Camera under water (0..1): the image wavers and takes the water's tint.
  underwater: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
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

fn noise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2f(1.0, 0.0)), u.x), mix(hash12(i + vec2f(0.0, 1.0)), hash12(i + vec2f(1.0, 1.0)), u.x), u.y);
}

fn fbm(p: vec2f) -> f32 {
  return noise2(p) * 0.5 + noise2(p * 2.03 + 7.1) * 0.3 + noise2(p * 4.1 + 2.3) * 0.2;
}

fn luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

/** Scene radiance with bloom and shafts, before exposure. */
fn hdrAt(uv: vec2f) -> vec3f {
  return textureSampleLevel(scene, samp, uv, 0.0).rgb;
}

/** Exposed and tone-mapped scene (display-referred, linear) at uv: what the filters work on. */
fn ldrAt(uv: vec2f) -> vec3f {
  return aces(hdrAt(uv) * params.exposure);
}

/**
 * Kuwahara: of the four quadrants around the pixel, keep the mean of the flattest one. Detail
 * flattens into strokes while edges stay crisp. `r` is the stroke radius in texels.
 */
fn kuwahara(uv: vec2f, r: i32, step: vec2f) -> vec3f {
  var m: array<vec3f, 4>;
  var s: array<vec3f, 4>;
  for (var k = 0; k < 4; k++) {
    m[k] = vec3f(0.0);
    s[k] = vec3f(0.0);
  }
  for (var j = -r; j <= r; j++) {
    for (var i = -r; i <= r; i++) {
      let c = ldrAt(uv + vec2f(f32(i), f32(j)) * step);
      let cc = c * c;
      if (i <= 0 && j <= 0) { m[0] += c; s[0] += cc; }
      if (i >= 0 && j <= 0) { m[1] += c; s[1] += cc; }
      if (i <= 0 && j >= 0) { m[2] += c; s[2] += cc; }
      if (i >= 0 && j >= 0) { m[3] += c; s[3] += cc; }
    }
  }
  let n = f32((r + 1) * (r + 1));
  var best = vec3f(0.0);
  var bestVar = 1e9;
  for (var k = 0; k < 4; k++) {
    let mean = m[k] / n;
    let v = s[k] / n - mean * mean;
    let tv = v.r + v.g + v.b;
    if (tv < bestVar) {
      bestVar = tv;
      best = mean;
    }
  }
  return best;
}

/** Luminance edge strength (Sobel) of the tone-mapped image, `d` texels apart. */
fn edges(uv: vec2f, d: vec2f) -> f32 {
  let tl = luma(ldrAt(uv + vec2f(-d.x, -d.y)));
  let t = luma(ldrAt(uv + vec2f(0.0, -d.y)));
  let tr = luma(ldrAt(uv + vec2f(d.x, -d.y)));
  let l = luma(ldrAt(uv + vec2f(-d.x, 0.0)));
  let r = luma(ldrAt(uv + vec2f(d.x, 0.0)));
  let bl = luma(ldrAt(uv + vec2f(-d.x, d.y)));
  let b = luma(ldrAt(uv + vec2f(0.0, d.y)));
  let br = luma(ldrAt(uv + vec2f(d.x, d.y)));
  let gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
  let gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
  return sqrt(gx * gx + gy * gy);
}

/** Paper fibre and tooth, in screen space (aspect-correct), around 1. */
fn paper(uv: vec2f) -> f32 {
  let p = vec2f(uv.x * params.aspect, uv.y);
  let fibre = noise2(p * vec2f(900.0, 260.0)) * 0.5 + noise2(p * vec2f(260.0, 900.0)) * 0.5;
  let tooth = fbm(p * 140.0);
  return 0.94 + fibre * 0.06 + tooth * 0.05;
}

/** Adds HDR light (bloom, shafts) over an already tone-mapped colour without clipping. */
fn screen(base: vec3f, light: vec3f) -> vec3f {
  return base + (vec3f(1.0) - base) * aces(light);
}

/** The standard grade: saturation, warm shadow lift, vignette (on a tone-mapped colour). */
fn finish(c: vec3f, uv: vec2f, vignette: f32) -> vec3f {
  let lum = luma(c);
  var col = mix(vec3f(lum), c, params.grade.w);
  col = col + params.grade.xyz * (1.0 - smoothstep(0.0, 0.4, lum)) * 0.03;
  let d = uv - 0.5;
  return col * (1.0 - vignette * dot(d, d) * 1.6);
}

@fragment
fn fs_main(@location(0) uvIn: vec2f, @builtin(position) frag: vec4f) -> @location(0) vec4f {
  let style = i32(params.style + 0.5);
  // Under water the view wavers, as through moving water.
  let uw = params.underwater;
  let uv = uvIn + uw * 0.0035 * vec2f(sin(uvIn.y * 38.0 + params.time * 2.1), cos(uvIn.x * 31.0 + params.time * 1.7));
  let glow = (textureSampleLevel(bloom, samp, uv, 0.0).rgb * params.bloomStrength + textureSampleLevel(shafts, samp, uv, 0.0).rgb) * params.exposure;
  let tx = params.texel;
  var col: vec3f;
  if (style == 1) {
    // Painterly: oil-like strokes; light still blooms over the paint, on faint canvas.
    var c = screen(kuwahara(uv, 3, tx * 1.5), glow * 0.8);
    let canvas = noise2(vec2f(uv.x * params.aspect, uv.y) * vec2f(420.0, 420.0));
    c = c * (0.97 + canvas * 0.05);
    col = finish(c, uv, params.vignette);
    let l = luma(col);
    col = mix(vec3f(l), col, 1.12);
  } else if (style == 2) {
    // Watercolor: pigment bleeds past edges (wobbled lookup), washes flatten, pigment pools at
    // the borders of each wash, granulates in the darks and leaves white paper in the lights.
    let wob = (vec2f(fbm(uv * 9.0 + 3.0), fbm(uv * 9.0 + 11.0)) - 0.5) * tx * 9.0;
    var c = screen(kuwahara(uv + wob, 2, tx * 2.0), glow * 0.5);
    let e = edges(uv + wob, tx * 2.0);
    let lum = luma(c);
    // Thin the paint: lift toward paper, more in the lights.
    let paperCol = vec3f(0.97, 0.95, 0.9);
    c = mix(c, paperCol, 0.08 + 0.3 * smoothstep(0.6, 0.95, lum));
    c = mix(vec3f(luma(c)), c, 1.25);
    let pool = smoothstep(0.1, 0.35, e);
    c = c * (1.0 - pool * 0.2);
    let gran = fbm(vec2f(uv.x * params.aspect, uv.y) * 220.0);
    c = c * (1.0 - (gran - 0.5) * 0.22 * (1.0 - lum));
    c = c * paper(uv);
    col = finish(c, uv, params.vignette * 0.5);
  } else if (style == 3) {
    // Film: lateral chromatic aberration, red halation around highlights, toe, grain.
    let d = uv - 0.5;
    let ca = d * dot(d, d) * 0.012;
    let h = vec3f(hdrAt(uv + ca).r, hdrAt(uv).g, hdrAt(uv - ca).b);
    var c = h * params.exposure + glow;
    let halo = textureSampleLevel(bloom, samp, uv, 0.0).rgb * params.exposure;
    c = c + vec3f(1.0, 0.32, 0.12) * luma(halo) * 0.35;
    c = aces(c * 1.05);
    // Faded blacks and a warm-green split, like a print stock.
    c = c * 0.94 + vec3f(0.03, 0.028, 0.035);
    let lum = luma(c);
    c = c + vec3f(0.02, 0.01, -0.02) * smoothstep(0.4, 0.9, lum) + vec3f(-0.01, 0.005, 0.015) * (1.0 - smoothstep(0.0, 0.35, lum));
    col = finish(c, uv, params.vignette * 1.4);
    let fy = floor(params.time * 24.0);
    let g = hash12(frag.xy * 0.71 + fy * 13.7) + hash12(frag.xy * 1.37 + fy * 7.3) - 1.0;
    col = col + g * 0.06 * (0.3 + sqrt(max(luma(col), 0.0))) * (1.0 - luma(col) * 0.6);
  } else if (style == 4) {
    // Miniature: tilt-shift blur away from a band below center, punchier colour.
    let band = smoothstep(0.08, 0.42, abs(uv.y - 0.56));
    var c = hdrAt(uv);
    if (band > 0.01) {
      var acc = c;
      var wsum = 1.0;
      let rad = band * 9.0;
      for (var i = 1; i < 24; i++) {
        let fi = f32(i);
        let a = fi * 2.39996;
        let r = sqrt(fi / 24.0) * rad;
        acc += hdrAt(uv + vec2f(cos(a), sin(a)) * r * tx);
        wsum += 1.0;
      }
      c = acc / wsum;
    }
    c = aces(c * params.exposure + glow);
    c = finish(c, uv, params.vignette);
    let l = luma(c);
    col = clamp(mix(vec3f(l), c, 1.3), vec3f(0.0), vec3f(1.0));
    col = col * col * (3.0 - 2.0 * col) * 0.35 + col * 0.65;
  } else if (style == 6) {
    // Cozy: vibrant but kind. Vibrance (dull colours gain more saturation than vivid ones),
    // a warm golden grade, softly lifted warm shadows, a little extra glow, a warm vignette.
    var c = hdrAt(uv) * params.exposure * 1.1 + glow * 1.35;
    c = aces(c);
    let lum = luma(c);
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let sat = (mx - mn) / max(mx, 1e-4);
    c = mix(vec3f(lum), c, 1.18 + 0.75 * (1.0 - sat));
    c = max(c, vec3f(0.0));
    // Warm: golden highlights, peach midtones, shadows lifted toward a warm umber.
    c = c * vec3f(1.04, 1.0, 0.9);
    c = c + vec3f(0.035, 0.022, 0.01) * (1.0 - smoothstep(0.0, 0.35, lum));
    c = c + vec3f(0.03, 0.018, -0.01) * smoothstep(0.45, 0.95, lum);
    // A soft S-curve for a little depth without crushing.
    c = mix(c, c * c * (3.0 - 2.0 * c), 0.25);
    let d = uv - 0.5;
    col = c * (1.0 - params.vignette * 0.8 * dot(d, d) * 1.6) + vec3f(0.012, 0.006, 0.0) * dot(d, d) * 4.0;
  } else if (style == 5) {
    // Ink wash (sumi-e): tone in three soft washes of ink on warm paper, linework at edges, a
    // breath of the original colour kept.
    let base = screen(kuwahara(uv, 2, tx * 1.5), glow * 0.4);
    let lum = luma(base);
    let e = edges(uv, tx * 1.5);
    let paperCol = vec3f(0.95, 0.91, 0.82);
    let ink = vec3f(0.08, 0.08, 0.1);
    let tone = pow(lum, 0.8);
    let washes = (smoothstep(0.12, 0.2, tone) + smoothstep(0.32, 0.42, tone) + smoothstep(0.6, 0.72, tone)) / 3.0;
    var c = mix(ink, paperCol, mix(washes, tone, 0.25));
    c = mix(c, c * (base / max(lum, 0.02)), 0.18);
    let line = smoothstep(0.08, 0.3, e * (0.8 + fbm(uv * 60.0) * 0.6));
    c = mix(c, ink, line * 0.75);
    c = c * paper(uv);
    col = finish(c, uv, params.vignette * 0.6);
  } else {
    var c = hdrAt(uv) * params.exposure + glow;
    // Gentle warm lift in the shadows, Flower-like pastel response.
    let lum = luma(c);
    c = mix(vec3f(lum), c, params.grade.w);
    c = c + params.grade.xyz * (1.0 - smoothstep(0.0, 0.6, lum)) * 0.04;
    c = aces(c);
    let d = uv - 0.5;
    col = c * (1.0 - params.vignette * dot(d, d) * 1.6);
  }
  if (uw > 0.0) {
    // A cool green-blue cast and darker edges, as if seen through a mask of water.
    let lum = luma(col);
    col = mix(col, mix(vec3f(lum) * vec3f(0.55, 0.95, 1.0), col * vec3f(0.7, 0.95, 1.0), 0.5), uw * 0.6);
    let d = uvIn - 0.5;
    col *= 1.0 - uw * dot(d, d) * 1.4;
  }
  var outc = srgb(clamp(col, vec3f(0.0), vec3f(1.0)));
  outc = outc + (hash12(frag.xy + params.time * 61.0) - 0.5) / 255.0;
  return vec4f(outc, 1.0);
}
