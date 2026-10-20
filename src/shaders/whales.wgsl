// Humpback whales (model, rig and animation: tools/blender/model_whale.py), skinned on the GPU
// from baked animation like the birds (birds.wgsl). Wet, dark, rubbery skin with a broad
// sheen; what is under water is seen through the sea's own surface (sea.wgsl blends over it).
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky, skyColor, underwaterFog } from "./lib/atmosphere.wgsl";
import { SEA_Y } from "./lib/terrain.wgsl";

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
@group(0) @binding(1) var<storage, read> whales: array<Bird>;
@group(0) @binding(2) var<storage, read> skins: array<u32>;

override BONES: u32 = 8u;

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
  let b = whales[ii];
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
  let wrap = clamp((dot(n, l) + 0.25) / 1.25, 0.0, 1.0);
  let albedo = frag.albedo;
  var col = albedo * (ambientSky(n, s) * 0.75 * frag.ao + G.sunColor * wrap * 0.9);
  // Out of the water the skin streams wet: a broad soft sheen and a sky reflection at the rim.
  let wet = smoothstep(-0.3, 0.2, frag.world.y - SEA_Y);
  let h = normalize(l + v);
  col += G.sunColor * (pow(max(dot(n, h), 0.0), 50.0) * 0.8 + pow(max(dot(n, h), 0.0), 8.0) * 0.06) * frag.gloss * wet;
  // The wet skin mirrors the sky (Fresnel), which is most of what makes it read as wet.
  let r = reflect(-v, n);
  let fres = 0.03 + 0.97 * pow(1.0 - abs(dot(n, v)), 5.0);
  col = mix(col, skyColor(normalize(vec3f(r.x, max(r.y, 0.02), r.z)), s), fres * 0.7 * wet);
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  col = underwaterFog(col, frag.world, G.camPos, G.underwater, G.sunColor, G.zenithColor);
  return vec4f(col, 1.0);
}

// (No light of their own: an empty halo pass, required by the shared species draw.)
@vertex
fn vs_halo(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(0.0, 0.0, -1.0, 1.0);
}

@fragment
fn fs_halo() -> @location(0) vec4f {
  return vec4f(0.0);
}
