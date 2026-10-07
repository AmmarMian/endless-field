// Meadow birds (model: tools/blender/model_bird.py): painted plumage from vertex colours.
// Wings flap about the shoulder; the outer hand bends further than the arm.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Bird {
  // xyz = position, w = heading (yaw)
  pos: vec4f,
  // x = flap angle, y = wing spread, z = pitch, w = seed (in [0, 1); 1 + k: glowing with k)
  pose: vec4f,
  // x = bank (roll, rad), y = size multiplier (0 = 1), z = elbow (span m; 0 = sparrow),
  // w = wing sweep (0 spread .. 1 swept back on the upstroke)
  extra: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> birds: array<Bird>;

override SHOULDER: f32 = 0.022;
const SCALE = 1.35;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) ao: f32,
  @location(4) @interpolate(flat) part: u32,
  @location(5) @interpolate(flat) glow: f32,
  @location(6) gloss: f32,
  // Feathers: x = along (0 root .. 1 tip), y = across (-1 .. 1, 0 on the shaft); z = 1 on
  // a feather. Body: the model-space position, for plumage texture.
  @location(7) feather: vec3f,
  @location(8) local: vec3f,
}

fn rotZ(v: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let b = birds[ii];
  // p.w: the part, plus (feathers) how far along the feather in the fraction.
  let part = u32(floor(p.w + 0.01));
  let along = fract(p.w + 0.01) / 0.9;
  var lp = p.xyz;
  let modelPos = p.xyz;
  var ln = n.xyz;
  if (part == 3u && t.x != 0.0) {
    // Wing: rotate about the shoulder (an axis along the body); the hand bends further.
    let side = sign(t.x);
    let d = abs(t.x);
    let elbow = select(0.035, b.extra.z, b.extra.z > 0.0);
    // The hand: a further bend at the wrist. The sparrows bend it with the arm; the swallow
    // (elbow given) keyframes it on its own (pose.y), lagging the arm through the stroke.
    let hand = select(b.pose.x * 0.6, b.pose.y, b.extra.z > 0.0);
    // The swallow's wing body grows out of the flank: its root stays in the body and the arm's
    // turn eases in over the first 3 cm (a smooth shoulder, no kink); the hand bends on top.
    let armW = select(smoothstep(0.0, 0.012, d), smoothstep(0.0, 0.03, d), b.extra.z > 0.0);
    let a = (b.pose.x * armW + hand * smoothstep(elbow, elbow * 2.1, d)) * side;
    // On the upstroke the hand sweeps back (the wing half folds), about the elbow.
    let sweep = b.extra.w * smoothstep(elbow * 0.6, elbow * 1.4, d) * 0.9;
    if (sweep > 0.0) {
      let ex = side * (SHOULDER + elbow);
      let rel = vec2f(lp.x - ex, lp.z);
      let cs = cos(sweep * side);
      let sn = sin(sweep * side);
      let r2 = vec2f(rel.x * cs - rel.y * sn, rel.x * sn + rel.y * cs);
      lp = vec3f(r2.x + ex, lp.y, r2.y);
    }
    let pivot = vec2f(side * SHOULDER, 0.008);
    let r = rotZ(lp.xy - pivot, a) + pivot;
    lp = vec3f(r, lp.z);
    ln = vec3f(rotZ(ln.xy, a), ln.z);
  }
  lp *= SCALE * select(1.0, b.extra.y, b.extra.y > 0.0);
  // Bank about the body's long axis, then pitch about x (positive dips the head), then heading.
  let cr = cos(b.extra.x);
  let sr = sin(b.extra.x);
  lp = vec3f(lp.x * cr - lp.y * sr, lp.x * sr + lp.y * cr, lp.z);
  ln = vec3f(ln.x * cr - ln.y * sr, ln.x * sr + ln.y * cr, ln.z);
  let cp = cos(-b.pose.z);
  let sp = sin(-b.pose.z);
  lp = vec3f(lp.x, lp.y * cp + lp.z * sp, -lp.y * sp + lp.z * cp);
  ln = vec3f(ln.x, ln.y * cp + ln.z * sp, -ln.y * sp + ln.z * cp);
  let cy = cos(b.pos.w);
  let sy = sin(b.pos.w);
  lp = vec3f(lp.x * cy + lp.z * sy, lp.y, -lp.x * sy + lp.z * cy);
  ln = vec3f(ln.x * cy + ln.z * sy, ln.y, -ln.x * sy + ln.z * cy);
  let world = b.pos.xyz + lp;
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = ln;
  // Painted colours are sRGB; a little per-bird variation in warmth.
  let warm = 0.92 + 0.16 * fract(b.pose.w * 7.3);
  out.albedo = pow(e.rgb, vec3f(2.2)) * vec3f(warm, 1.0, 2.0 - warm);
  out.ao = e.a;
  out.part = part;
  out.glow = max(b.pose.w - 1.0, 0.0);
  out.gloss = t.y;
  // z: 1 on the swallow's wing and tail feathers, 2 on its body, 0 for other birds.
  out.feather = vec3f(along, n.w, select(0.0, select(2.0, 1.0, part == 3u || part == 4u), b.extra.z > 0.0));
  out.local = modelPos;
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
  // Pattern frequencies on screen (derivatives in uniform control flow, before branching).
  let fineF = 1.0 - smoothstep(0.3, 1.0, length(fwidth(vec2f(frag.feather.x * 60.0, abs(frag.feather.y) * 14.0))));
  let q = frag.local.xz * vec2f(320.0, 240.0) + vec2f(0.0, frag.local.y * 200.0);
  let fineB = 1.0 - smoothstep(0.3, 1.0, length(fwidth(q)));
  if (frag.feather.z > 0.5 && frag.feather.z < 1.5) {
    // A feather: darker toward the shaft with a fine pale shaft line, barbs as faint slanting
    // stripes, slightly lighter worn edges and a soft grey-brown fringe at the tip.
    let u = frag.feather.x;
    let a = abs(frag.feather.y);
    let fine = fineF;
    albedo *= 0.82 + 0.18 * smoothstep(0.0, 0.7, a);
    let shaft = (1.0 - smoothstep(0.03, 0.1, a)) * smoothstep(0.02, 0.1, u) * (1.0 - smoothstep(0.85, 1.0, u));
    albedo = mix(albedo, albedo * 1.5 + vec3f(0.025), shaft * 0.6 * fine);
    albedo *= 1.0 + 0.07 * sin(u * 60.0 + a * 14.0) * fine;
    albedo += vec3f(0.02, 0.02, 0.022) * smoothstep(0.8, 1.0, a);
    albedo = mix(albedo, vec3f(0.16, 0.15, 0.14), smoothstep(0.88, 1.0, u) * 0.35);
  } else if (frag.feather.z > 1.5) {
    // Body plumage: small overlapping feathers, crescent-edged, fading out at a distance.
    let cell = fract(q) - vec2f(0.5, 0.15);
    let crescent = smoothstep(0.32, 0.46, length(cell));
    let fine = fineB;
    albedo *= 1.0 - 0.12 * crescent * fine;
  }
  var col = albedo * (ambientSky(n, s) * 0.75 * frag.ao + G.sunColor * (wrap * 0.85 + back));
  if (frag.part == 2u) {
    // Eyes: a bright wet glint.
    let h = normalize(l + v);
    col += G.sunColor * pow(max(dot(n, h), 0.0), 80.0) * 2.0;
  }
  if (frag.feather.z > 0.5) {
    // Structural colour of the swallow's dark plumage: not a gloss, a hue that shifts with
    // the angle (steel-blue facing you, violet to green-blue toward the edges), only on the
    // dark upper feathers.
    let edge = 1.0 - abs(dot(n, v));
    let dark = 1.0 - smoothstep(0.08, 0.25, dot(albedo, vec3f(0.33)));
    let hue = mix(vec3f(0.12, 0.28, 0.75), mix(vec3f(0.42, 0.2, 0.7), vec3f(0.1, 0.5, 0.55), smoothstep(0.5, 0.9, edge)), smoothstep(0.2, 0.7, edge));
    col += hue * dark * (ambientSky(n, s) * 0.1 + G.sunColor * wrap * 0.035);
    // Feathered and matte: soft sky fill (dark plumage still reads), a faint velvet rim.
    col += albedo * ambientSky(n, s) * 0.35;
    col += albedo * (G.zenithColor * 0.4 + G.sunColor * 0.12) * pow(1.0 - abs(dot(n, v)), 3.0);
  }
  if (frag.glow > 0.0) {
    // A spirit bird: lit from within, with a bright turquoise rim.
    let rim = pow(1.0 - abs(dot(n, v)), 2.0);
    col += frag.albedo * frag.glow * 0.35 + vec3f(0.45, 0.95, 1.0) * rim * frag.glow * 0.35;
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
  let size = 0.7 + d * 0.008;
  var out: HOut;
  out.pos = G.viewProj * vec4f(b.pos.xyz + vec3f(0.0, 0.05, 0.0) + (right * c.x + up * c.y) * size, 1.0);
  out.uv = c;
  out.k = max(b.pose.w - 1.0, 0.0) * (0.85 + 0.15 * sin(G.time * 3.0));
  return out;
}

@fragment
fn fs_halo(frag: HOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let light = exp(-r2 * 5.0) * 0.9 + exp(-r2 * 28.0) * 1.6;
  let col = mix(vec3f(1.0, 0.75, 0.35), vec3f(0.45, 0.9, 1.0), exp(-r2 * 6.0));
  return vec4f(col * light * frag.k * mix(0.35, 0.5, G.night), 0.0);
}
