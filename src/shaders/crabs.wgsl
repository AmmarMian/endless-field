// Shore crabs (model, rig and animation: tools/blender/model_crab.py), skinned on the GPU from
// baked animation exactly like the birds (birds.wgsl). The exoskeleton is shaded as wet,
// glossy chitin: mottled shell, dark claw tips, beady eyes; darker and shinier when wet.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky, underwaterFog } from "./lib/atmosphere.wgsl";
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
@group(0) @binding(1) var<storage, read> crabs: array<Bird>;
@group(0) @binding(2) var<storage, read> skins: array<u32>;

override BONES: u32 = 24u;

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
  let b = crabs[ii];
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
  let wrap = clamp((dot(n, l) + 0.2) / 1.2, 0.0, 1.0);
  // Fresh from the swash (near the water line) the shell is darker and wetter.
  let wet = 1.0 - smoothstep(0.3, 1.2, frag.world.y - SEA_Y);
  var albedo = frag.albedo * mix(1.0, 0.7, wet);
  var col = albedo * (ambientSky(n, s) * 0.7 * frag.ao + G.sunColor * wrap * 0.9);
  // Chitin is lacquer-glossy: a tight highlight, and a broader sheen at grazing angles.
  let h = normalize(l + v);
  let gloss = mix(frag.gloss, 1.0, wet * 0.5);
  let tight = select(60.0, 140.0, frag.part != 0u);
  col += G.sunColor * pow(max(dot(n, h), 0.0), tight) * gloss * select(0.6, 2.0, frag.part == 2u);
  col += (G.horizonColor * 0.5 + G.zenithColor * 0.3) * pow(1.0 - abs(dot(n, v)), 4.0) * gloss * 0.35;
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  col = underwaterFog(col, frag.world, G.camPos, G.underwater, G.sunColor, G.zenithColor);
  return vec4f(col, 1.0);
}

// (Crabs give off no light: an empty halo pass, required by the shared species draw.)
@vertex
fn vs_halo(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(0.0, 0.0, -1.0, 1.0);
}

@fragment
fn fs_halo() -> @location(0) vec4f {
  return vec4f(0.0);
}
