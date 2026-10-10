// Birds (models, rigs and animation: tools/blender/model_birds.py). Each species is skinned on
// the GPU from animation baked in Blender: per clip frame and bone, a skinning matrix (3 rows,
// float16). An instance plays two frames of its clips at once and blends them (a crossfade
// from the clip it left to the one it plays), between baked frames as well.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Bird {
  // xyz = position (the body's centre), w = heading (yaw)
  pos: vec4f,
  // x = pitch (+ dips the head), y = bank (roll), z = size (times life size), w = glow (0..1)
  pose: vec4f,
  // x = frame now (absolute, fractional), y = frame of the clip faded from, z = its weight,
  // w = a per-bird random in [0, 1)
  anim: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> birds: array<Bird>;
@group(0) @binding(2) var<storage, read> skins: array<u32>;

override BONES: u32 = 14u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) ao: f32,
  @location(4) @interpolate(flat) part: u32,
  @location(5) @interpolate(flat) glow: f32,
  @location(6) gloss: f32,
  // Wing: span 0..1, chord 0..1; tail: across -1..1, along 0..1; z = 1 on an upper surface.
  @location(7) feather: vec3f,
  @location(8) local: vec3f,
}

struct Rows {
  r0: vec4f,
  r1: vec4f,
  r2: vec4f,
}

fn rowsAt(frame: u32, bone: u32) -> Rows {
  let b = (frame * BONES + bone) * 6u;
  return Rows(
    vec4f(unpack2x16float(skins[b]), unpack2x16float(skins[b + 1u])),
    vec4f(unpack2x16float(skins[b + 2u]), unpack2x16float(skins[b + 3u])),
    vec4f(unpack2x16float(skins[b + 4u]), unpack2x16float(skins[b + 5u])),
  );
}

fn mixRows(a: Rows, b: Rows, t: f32) -> Rows {
  return Rows(mix(a.r0, b.r0, t), mix(a.r1, b.r1, t), mix(a.r2, b.r2, t));
}

/** A bone's matrix at a fractional frame (the next baked frame is always in the same clip). */
fn boneAt(frame: f32, bone: u32) -> Rows {
  let f0 = floor(frame);
  let i = u32(max(f0, 0.0));
  return mixRows(rowsAt(i, bone), rowsAt(i + 1u, bone), frame - f0);
}

fn bonePose(anim: vec4f, bone: u32) -> Rows {
  var m = boneAt(anim.x, bone);
  if (anim.z > 0.001) {
    m = mixRows(m, boneAt(anim.y, bone), anim.z);
  }
  return m;
}

fn apply(m: Rows, v: vec4f) -> vec3f {
  return vec3f(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

/** Model space -> world: size, bank about the body's long axis, pitch, then heading. */
fn place(b: Bird, v: vec3f, isDir: bool) -> vec3f {
  var lp = v * select(b.pose.z, 1.0, isDir);
  let cr = cos(b.pose.y);
  let sr = sin(b.pose.y);
  lp = vec3f(lp.x * cr - lp.y * sr, lp.x * sr + lp.y * cr, lp.z);
  let cp = cos(-b.pose.x);
  let sp = sin(-b.pose.x);
  lp = vec3f(lp.x, lp.y * cp + lp.z * sp, -lp.y * sp + lp.z * cp);
  let cy = cos(b.pos.w);
  let sy = sin(b.pos.w);
  return vec3f(lp.x * cy + lp.z * sy, lp.y, -lp.x * sy + lp.z * cy);
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @location(4) j: vec4u, @builtin(instance_index) ii: u32) -> VOut {
  let b = birds[ii];
  let w0 = f32(j.z) / 255.0;
  var m = bonePose(b.anim, j.x);
  if (w0 < 0.999) {
    m = mixRows(bonePose(b.anim, j.y), m, w0);
  }
  let lp = apply(m, vec4f(p.xyz, 1.0));
  let ln = apply(m, vec4f(n.xyz, 0.0));
  let world = b.pos.xyz + place(b, lp, false);
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = place(b, ln, true);
  // Painted colours are sRGB; a little per-bird variation in warmth.
  let warm = 0.94 + 0.12 * b.anim.w;
  out.albedo = pow(e.rgb, vec3f(2.2)) * vec3f(warm, 1.0, 2.0 - warm);
  out.ao = e.a;
  out.part = u32(p.w + 0.5);
  out.glow = b.pose.w;
  out.gloss = f32(j.w) / 255.0;
  out.feather = vec3f(t, n.w);
  out.local = p.xyz;
  return out;
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  let v = normalize(G.camPos - frag.world);
  let l = G.sunDir;
  // Soft wrapped light for down and feathers; thin wing and tail feathers glow when backlit.
  let wrap = clamp((dot(n, l) + 0.35) / 1.35, 0.0, 1.0);
  let thin = select(0.0, 0.5, frag.part == 3u || frag.part == 4u);
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 4.0) * thin;
  var albedo = frag.albedo;
  // Feathers drawn from the wing's and tail's own coordinates: (along, across) a feather.
  var fe = vec2f(0.0, 0.5);
  var feathered = false;
  if (frag.part == 3u) {
    let c = frag.feather.y;
    if (c > 0.42) {
      // Flight feathers: one per 1/18 of the span, running back to the trailing edge.
      fe = vec2f((c - 0.42) / 0.58, fract(frag.feather.x * 18.0) * 2.0 - 1.0);
    } else {
      // Coverts: overlapping rows toward the leading edge.
      fe = vec2f(fract(c / 0.42 * 3.0), fract(frag.feather.x * 30.0 + floor(c / 0.42 * 3.0) * 0.5) * 2.0 - 1.0);
    }
    feathered = true;
  } else if (frag.part == 4u) {
    fe = vec2f(frag.feather.y, fract((frag.feather.x * 0.5 + 0.5) * 6.0) * 2.0 - 1.0);
    feathered = true;
  }
  // Pattern frequencies on screen (derivatives in uniform control flow, before branching).
  let fineF = 1.0 - smoothstep(0.3, 1.0, length(fwidth(frag.feather.xy * vec2f(18.0, 6.0))));
  let q = frag.local.xz * vec2f(320.0, 240.0) + vec2f(0.0, frag.local.y * 200.0);
  let fineB = 1.0 - smoothstep(0.3, 1.0, length(fwidth(q)));
  if (feathered) {
    // A feather: darker toward the shaft with a fine pale shaft line, barbs as faint slanting
    // stripes, slightly lighter worn edges.
    let u = fe.x;
    let a = abs(fe.y);
    albedo *= 0.84 + 0.16 * smoothstep(0.0, 0.7, a) * fineF + 0.16 * (1.0 - fineF);
    let shaft = (1.0 - smoothstep(0.03, 0.12, a)) * smoothstep(0.02, 0.1, u) * (1.0 - smoothstep(0.85, 1.0, u));
    albedo = mix(albedo, albedo * 1.4 + vec3f(0.02), shaft * 0.5 * fineF);
    albedo *= 1.0 + 0.06 * sin(u * 50.0 + a * 12.0) * fineF;
    albedo *= 1.0 - 0.18 * smoothstep(0.85, 1.0, a) * fineF;
  } else if (frag.part == 0u) {
    // Body plumage: small overlapping feathers, crescent-edged, fading out at a distance.
    let cell = fract(q) - vec2f(0.5, 0.15);
    let crescent = smoothstep(0.32, 0.46, length(cell));
    albedo *= 1.0 - 0.12 * crescent * fineB;
  }
  var col = albedo * (ambientSky(n, s) * 0.75 * frag.ao + G.sunColor * (wrap * 0.85 + back));
  if (frag.part == 2u || frag.part == 1u) {
    // Eyes (and a little on the bill): a bright wet glint.
    let h = normalize(l + v);
    col += G.sunColor * pow(max(dot(n, h), 0.0), select(30.0, 80.0, frag.part == 2u)) * select(0.2, 2.0, frag.part == 2u);
  }
  if (frag.gloss > 0.01) {
    // Structural colour of glossy plumage: not a gloss, a hue that shifts with the angle
    // (steel-blue facing you, violet to green-blue toward the edges), only on dark feathers.
    let edge = 1.0 - abs(dot(n, v));
    let dark = 1.0 - smoothstep(0.08, 0.3, dot(albedo, vec3f(0.33)));
    let hue = mix(vec3f(0.12, 0.28, 0.75), mix(vec3f(0.42, 0.2, 0.7), vec3f(0.1, 0.5, 0.55), smoothstep(0.5, 0.9, edge)), smoothstep(0.2, 0.7, edge));
    col += hue * dark * frag.gloss * (ambientSky(n, s) * 0.1 + G.sunColor * wrap * 0.035);
  }
  // Feathered and matte: soft sky fill (dark plumage still reads), a faint velvet rim.
  col += albedo * ambientSky(n, s) * 0.2;
  col += albedo * (G.zenithColor * 0.4 + G.sunColor * 0.12) * pow(1.0 - abs(dot(n, v)), 3.0);
  if (frag.glow > 0.0) {
    // A spirit bird: lit from within, with a warm rim.
    let rim = pow(1.0 - abs(dot(n, v)), 2.0);
    col += (albedo * 0.5 + vec3f(1.0, 0.85, 0.6) * rim * 0.4) * frag.glow * 0.45;
  }
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}

// ---- Halo: the light a glowing bird gives off (additive, around its body) ----

struct HOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) k: f32,
}

@vertex
fn vs_halo(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> HOut {
  let b = birds[ii];
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let toCam = G.camPos - b.pos.xyz;
  let d = length(toCam);
  let view = toCam / d;
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), view));
  let up = cross(view, right);
  // Grows a little with distance, so the light still reads from far off.
  let size = (0.7 + d * 0.008) * 0.75;
  var out: HOut;
  out.pos = G.viewProj * vec4f(b.pos.xyz + (right * c.x + up * c.y) * size, 1.0);
  out.uv = c;
  // A faint, warm presence, not a lamp.
  out.k = b.pose.w * (0.85 + 0.15 * sin(G.time * 3.0)) * 0.4;
  return out;
}

@fragment
fn fs_halo(frag: HOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let light = exp(-r2 * 5.0) * 0.9 + exp(-r2 * 28.0) * 1.6;
  let col = mix(vec3f(1.0, 0.75, 0.35), vec3f(1.0, 0.86, 0.6), exp(-r2 * 6.0));
  return vec4f(col * light * frag.k * mix(0.35, 0.5, G.night), 0.0);
}
