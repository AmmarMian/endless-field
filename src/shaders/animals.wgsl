// Field animals (models, rigs and animation: tools/blender/model_animals.py), skinned on the
// GPU from animation baked in Blender, exactly as the birds are (see birds.wgsl): an instance
// plays a frame of one clip and crossfades from another.
//
// KIND picks the surface: 0 fur (hares, deer, foxes), 1 butterfly (wing patterns painted here
// by species, chosen per instance), 2 dragonfly (glassy veined wings, metallic body), 3 frog
// (wet skin), 4 duck (plumage).
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Animal {
  // xyz = position (ground under it, or the body for fliers), w = heading
  pos: vec4f,
  // x = pitch, y = roll, z = size (times life size), w = glow (0..1)
  pose: vec4f,
  // x = frame, y = frame faded from, z = its weight, w = random per animal in [0, 1)
  anim: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> animals: array<Animal>;
@group(0) @binding(2) var<storage, read> skins: array<u32>;

override BONES: u32 = 14u;
override KIND: u32 = 0u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) ao: f32,
  @location(4) @interpolate(flat) part: u32,
  @location(5) @interpolate(flat) glow: f32,
  @location(6) gloss: f32,
  // Wings: span 0..1, around 0..1 (+2 on hind wings); z = 1 on an upper / outer surface.
  @location(7) coord: vec3f,
  @location(8) local: vec3f,
  @location(9) @interpolate(flat) seed: f32,
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

fn place(a: Animal, v: vec3f, isDir: bool) -> vec3f {
  var lp = v * select(a.pose.z, 1.0, isDir);
  let cr = cos(a.pose.y);
  let sr = sin(a.pose.y);
  lp = vec3f(lp.x * cr - lp.y * sr, lp.x * sr + lp.y * cr, lp.z);
  let cp = cos(-a.pose.x);
  let sp = sin(-a.pose.x);
  lp = vec3f(lp.x, lp.y * cp + lp.z * sp, -lp.y * sp + lp.z * cp);
  let cy = cos(a.pos.w);
  let sy = sin(a.pos.w);
  return vec3f(lp.x * cy + lp.z * sy, lp.y, -lp.x * sy + lp.z * cy);
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @location(4) j: vec4u, @builtin(instance_index) ii: u32) -> VOut {
  let a = animals[ii];
  let w0 = f32(j.z) / 255.0;
  var m = bonePose(a.anim, j.x);
  if (w0 < 0.999) {
    m = mixRows(bonePose(a.anim, j.y), m, w0);
  }
  let lp = apply(m, vec4f(p.xyz, 1.0));
  let ln = apply(m, vec4f(n.xyz, 0.0));
  let world = a.pos.xyz + place(a, lp, false);
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = place(a, ln, true);
  // Painted colours are sRGB; each animal a little warmer or cooler, lighter or darker.
  let s = a.anim.w;
  out.albedo = pow(e.rgb, vec3f(2.2)) * (0.9 + 0.2 * fract(s * 13.7)) * vec3f(0.96 + 0.08 * s, 1.0, 1.04 - 0.08 * s);
  out.ao = e.a;
  out.part = u32(p.w + 0.5);
  out.glow = a.pose.w;
  out.gloss = f32(j.w) / 255.0;
  out.coord = vec3f(t, n.w);
  out.local = p.xyz;
  out.seed = s;
  return out;
}

fn hash1(x: f32) -> f32 {
  return fract(sin(x * 127.1) * 43758.5453);
}

fn noise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let n = i.x + i.y * 57.0 + i.z * 113.0;
  return mix(
    mix(mix(hash1(n), hash1(n + 1.0), u.x), mix(hash1(n + 57.0), hash1(n + 58.0), u.x), u.y),
    mix(mix(hash1(n + 113.0), hash1(n + 114.0), u.x), mix(hash1(n + 170.0), hash1(n + 171.0), u.x), u.y),
    u.z,
  );
}

fn ring(d: f32, r0: f32, r1: f32) -> f32 {
  return smoothstep(r0 - 0.02, r0, d) * (1.0 - smoothstep(r1, r1 + 0.02, d));
}

/**
 * Butterfly wings by species (s: out from the root 0..1, c: around the wing 0..1, hind: on the
 * hind wing, top: upper side). Species: 0 peacock, 1 brimstone, 2 small white, 3 common blue,
 * 4 orange-tip, 5 small tortoiseshell.
 */
fn butterflyWing(species: u32, s: f32, c: f32, hind: bool, top: bool) -> vec3f {
  let margin = smoothstep(0.86, 0.95, s);
  let root = 1.0 - smoothstep(0.0, 0.3, s);
  // Position in a rough wing plane: apex of the forewing near (s 1, c 0.35).
  let ang = (c - 0.35) * 3.0;
  let q = vec2f(s * cos(ang), s * sin(ang));
  var col = vec3f(0.5);
  if (species == 0u) {
    // Peacock: rust red, a big eyespot at each wing's outer corner, dark margins; below, ash.
    col = vec3f(0.55, 0.1, 0.05);
    let eyeAt = select(vec2f(0.82, 0.05), vec2f(0.72, -0.25), hind);
    let d = distance(q, eyeAt);
    col = mix(col, vec3f(0.95, 0.8, 0.45), ring(d, 0.0, 0.2));
    col = mix(col, vec3f(0.05, 0.05, 0.08), ring(d, 0.0, 0.14));
    col = mix(col, vec3f(0.25, 0.45, 0.85), ring(d, 0.04, 0.1));
    col = mix(col, vec3f(0.12, 0.06, 0.05), margin * 0.8);
    if (!top) {
      col = vec3f(0.09, 0.07, 0.06) * (0.8 + 0.4 * fract(sin(s * 40.0 + c * 17.0) * 3.0));
    }
  } else if (species == 1u) {
    // Brimstone: sulphur yellow (hind paler), a small orange dot mid-wing.
    col = select(vec3f(0.95, 0.85, 0.22), vec3f(0.9, 0.86, 0.45), hind);
    col = mix(col, vec3f(0.95, 0.5, 0.1), 1.0 - smoothstep(0.04, 0.07, distance(q, vec2f(0.5, -0.05))));
    col = select(col * vec3f(0.85, 0.95, 0.7), col, top);
  } else if (species == 2u) {
    // Small white: chalk white, a grey-black apex and one spot; creamy below.
    col = vec3f(0.94, 0.94, 0.9);
    if (!hind) {
      col = mix(col, vec3f(0.18, 0.18, 0.2), smoothstep(0.75, 0.85, s) * (1.0 - smoothstep(0.15, 0.3, abs(c - 0.3))) * select(0.2, 1.0, top));
      col = mix(col, vec3f(0.18, 0.18, 0.2), 1.0 - smoothstep(0.05, 0.075, distance(q, vec2f(0.55, -0.12))));
    }
    col = select(col * vec3f(1.0, 0.98, 0.78), col, top);
  } else if (species == 3u) {
    // Common blue: violet blue with a white fringe; grey-brown dotted below.
    if (top) {
      col = mix(vec3f(0.3, 0.42, 0.95), vec3f(0.08, 0.1, 0.2), smoothstep(0.82, 0.9, s));
      col = mix(col, vec3f(0.95), smoothstep(0.93, 0.98, s));
    } else {
      col = vec3f(0.6, 0.58, 0.55);
      let spots = 1.0 - smoothstep(0.02, 0.035, distance(fract(q * 6.0), vec2f(0.5)));
      col = mix(col, vec3f(0.08), spots * smoothstep(0.3, 0.5, s) * (1.0 - margin));
      col = mix(col, vec3f(0.95, 0.55, 0.15), smoothstep(0.75, 0.8, s) * (1.0 - smoothstep(0.86, 0.9, s)) * select(0.0, 1.0, hind));
    }
  } else if (species == 4u) {
    // Orange-tip: white, the forewing's outer half orange with a dark tip; mossy green
    // marbling below the hind wing.
    col = vec3f(0.95, 0.95, 0.92);
    if (!hind) {
      col = mix(col, vec3f(1.0, 0.45, 0.08), smoothstep(0.5, 0.56, s) * select(0.4, 1.0, top));
      col = mix(col, vec3f(0.15, 0.15, 0.15), smoothstep(0.85, 0.9, s) * (1.0 - smoothstep(0.15, 0.25, abs(c - 0.3))));
    } else if (!top) {
      let marble = noise3(vec3f(q * 9.0, 1.0));
      col = mix(col, vec3f(0.4, 0.5, 0.25), smoothstep(0.45, 0.6, marble));
    }
  } else {
    // Small tortoiseshell: orange with black and yellow blotches on the leading edge, a
    // dark base and blue crescents along the margin.
    col = vec3f(0.92, 0.4, 0.07);
    col = mix(col, vec3f(0.08, 0.06, 0.05), root * 0.9);
    if (!hind) {
      let lead = smoothstep(0.08, 0.0, abs(c - 0.12)) + smoothstep(0.08, 0.0, abs(c - 0.24));
      col = mix(col, vec3f(0.06), clamp(lead, 0.0, 1.0) * smoothstep(0.3, 0.4, s) * (1.0 - smoothstep(0.8, 0.85, s)));
      col = mix(col, vec3f(0.95, 0.85, 0.5), smoothstep(0.06, 0.0, abs(c - 0.18)) * smoothstep(0.5, 0.6, s) * (1.0 - smoothstep(0.72, 0.78, s)));
    }
    let edge = smoothstep(0.86, 0.9, s);
    col = mix(col, vec3f(0.07), edge);
    col = mix(col, vec3f(0.2, 0.35, 0.85), edge * (1.0 - smoothstep(0.92, 0.95, s)) * step(0.5, fract(c * 14.0)));
    if (!top) {
      col = vec3f(0.18, 0.13, 0.08) * (0.8 + 0.4 * noise3(vec3f(q * 14.0, 2.0)));
    }
  }
  // Every wing darkens a little at the very root (the furry body side).
  return col * (1.0 - 0.35 * root);
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let sky = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  let v = normalize(G.camPos - frag.world);
  let l = G.sunDir;
  var albedo = frag.albedo;
  var alpha = 1.0;
  let thin = frag.part == 3u;
  // Fur and skin detail, faded out where it would alias.
  let fp = frag.local * select(260.0, 2400.0, KIND == 1u || KIND == 2u);
  let fineK = 1.0 - smoothstep(0.4, 1.2, length(fwidth(fp)));
  let hair = noise3(fp * vec3f(1.0, 1.0, 3.0));
  if (KIND == 1u && thin) {
    let species = u32(frag.seed * 6.0) % 6u;
    let hind = frag.coord.y >= 1.5;
    let c = select(frag.coord.y, frag.coord.y - 2.0, hind);
    albedo = pow(butterflyWing(species, frag.coord.x, c, hind, (frag.coord.z > 0.5) == front), vec3f(2.2));
    // Scales: a faint sheen across the wing.
    albedo *= 0.92 + 0.08 * sin(frag.coord.x * 90.0) * fineK;
  } else if (KIND == 2u && thin) {
    // Dragonfly wing: clear, a fine net of veins, a dark pterostigma near the tip.
    let s = frag.coord.x;
    let c = fract(frag.coord.y);
    let veins = max(1.0 - smoothstep(0.0, 0.08, abs(fract(s * 22.0) - 0.5) * 2.0 - 0.9), 1.0 - smoothstep(0.0, 0.1, abs(fract(c * 9.0) - 0.5) * 2.0 - 0.85));
    let stigma = smoothstep(0.8, 0.83, s) * (1.0 - smoothstep(0.88, 0.9, s)) * (1.0 - smoothstep(0.1, 0.25, abs(c - 0.25)));
    albedo = mix(vec3f(0.75, 0.82, 0.85), vec3f(0.1, 0.1, 0.1), max(veins * 0.6 * fineK, stigma));
    alpha = 0.3 + 0.5 * max(veins * fineK, stigma);
  } else if (frag.part == 0u) {
    if (KIND == 0u) {
      albedo *= mix(1.0, 0.75 + 0.5 * hair, fineK);
    } else {
      albedo *= mix(1.0, 0.88 + 0.24 * hair, fineK);
    }
  }
  let wrap = clamp((dot(n, l) + select(0.3, 0.6, thin)) / select(1.3, 1.6, thin), 0.0, 1.0);
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0) * select(0.0, 0.8, thin);
  var col = albedo * (ambientSky(n, sky) * 0.8 * frag.ao + G.sunColor * (wrap * 0.85 + back));
  // Fur: a soft sheen on the outline (light caught in the hairs).
  if (KIND == 0u && frag.part == 0u) {
    col += albedo * (G.sunColor * 0.25 + G.zenithColor * 0.2) * pow(1.0 - abs(dot(n, v)), 2.5);
  }
  if (frag.gloss > 0.01 || frag.part == 2u) {
    // Wet noses, eyes, frog skin, metallic dragonfly bodies: a sharp highlight.
    let h = normalize(l + v);
    let k = select(frag.gloss, 1.0, frag.part == 2u);
    col += G.sunColor * pow(max(dot(n, h), 0.0), select(40.0, 90.0, frag.part == 2u)) * k * select(0.5, 2.0, frag.part == 2u);
    if (KIND == 2u && frag.part == 0u) {
      let edge = 1.0 - abs(dot(n, v));
      col += albedo * 0.6 * vec3f(0.4, 0.9, 1.0) * edge;
    }
  }
  if (frag.glow > 0.0) {
    let rim = pow(1.0 - abs(dot(n, v)), 2.0);
    col += (albedo * 0.6 + vec3f(1.0, 0.85, 0.55) * rim * 0.5) * frag.glow;
  }
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, sky, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, alpha);
}

// ---- Halo: night butterflies glow softly (additive) ----

struct HOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) k: f32,
}

@vertex
fn vs_halo(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> HOut {
  let a = animals[ii];
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let toCam = G.camPos - a.pos.xyz;
  let view = toCam / length(toCam);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), view));
  let up = cross(view, right);
  let size = 0.35 + length(toCam) * 0.004;
  var out: HOut;
  out.pos = G.viewProj * vec4f(a.pos.xyz + (right * c.x + up * c.y) * size, 1.0);
  out.uv = c;
  out.k = a.pose.w * (0.8 + 0.2 * sin(G.time * 2.0 + a.anim.w * 30.0));
  return out;
}

@fragment
fn fs_halo(frag: HOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let light = exp(-r2 * 6.0) * 0.6 + exp(-r2 * 30.0);
  return vec4f(vec3f(0.85, 0.95, 1.0) * light * frag.k * 0.35, 0.0);
}
