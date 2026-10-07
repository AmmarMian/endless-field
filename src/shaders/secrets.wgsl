// The wind's secrets (model: tools/blender/model_secrets.py): paper pinwheels whose sails spin
// about the pin, and glass wind bells whose bells, clappers and paper strips swing. Plus the
// glint: a small star that twinkles over a secret not yet woken, readable from far away.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Secret {
  // xyz = base, w = yaw
  pos: vec4f,
  // x = spin angle (rad), y = swing amplitude (rad), z = scale, w = seed
  anim: vec4f,
}

struct Glint {
  // xyz = position, w = strength (0 hidden .. 1)
  pos: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> secrets: array<Secret>;
@group(0) @binding(2) var<storage, read> glints: array<Glint>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) ao: f32,
  @location(4) @interpolate(flat) part: u32,
}

fn rot2(v: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let s = secrets[ii];
  let part = u32(round(p.w));
  var lp = p.xyz;
  var ln = n.xyz;
  if (part == 1u || part == 2u) {
    // Sails and pin spin about the pin's axis (+Z through the hub).
    let r = rot2(lp.xy - t, s.anim.x) + t;
    lp = vec3f(r, lp.z);
    ln = vec3f(rot2(ln.xy, s.anim.x), ln.z);
  } else if (part >= 3u && t.y > 0.01) {
    // Bells swing from their cord; each its own rhythm; the paper strips flutter further.
    let bell = round((t.x + 0.25) / 0.26);
    let w = G.time * (2.1 + bell * 0.37) + s.anim.w * 20.0 + bell * 1.7;
    var a = s.anim.y * sin(w) + 0.04 * sin(G.time * 1.3 + bell);
    var b = s.anim.y * 0.35 * sin(w * 0.7 + 1.0);
    if (part == 4u) {
      a = a * 1.6 + s.anim.y * 0.5 * sin(G.time * 7.0 + bell * 3.0) * smoothstep(0.0, 0.1, t.y - lp.y);
    }
    // Swing in the z-y plane (toward and away from the wind), and a little sideways.
    let q = lp - vec3f(t.x, t.y, 0.0);
    let yz = rot2(q.zy, a);
    var r = vec3f(q.x, yz.y, yz.x);
    let xy = rot2(r.xy, b);
    r = vec3f(xy, r.z);
    lp = r + vec3f(t.x, t.y, 0.0);
    let nyz = rot2(ln.zy, a);
    ln = vec3f(ln.x, nyz.y, nyz.x);
  }
  lp *= s.anim.z;
  let cy = cos(s.pos.w);
  let sy = sin(s.pos.w);
  lp = vec3f(lp.x * cy + lp.z * sy, lp.y, -lp.x * sy + lp.z * cy);
  ln = vec3f(ln.x * cy + ln.z * sy, ln.y, -ln.x * sy + ln.z * cy);
  let world = s.pos.xyz + lp;
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = ln;
  out.albedo = pow(e.rgb, vec3f(2.2));
  out.ao = e.a;
  out.part = part;
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
  let wrap = clamp((dot(n, l) + 0.3) / 1.3, 0.0, 1.0);
  var col = frag.albedo * (ambientSky(n, s) * 0.8 * frag.ao + G.sunColor * wrap * 0.8);
  if (frag.part == 1u || frag.part == 4u) {
    // Paper: light shines through from behind.
    let through = pow(clamp(dot(-v, l), 0.0, 1.0), 2.0) * 0.8 + max(dot(-n, l), 0.0) * 0.45;
    col += frag.albedo * G.sunColor * through;
  }
  if (frag.part == 3u) {
    // Glass: tinted, with a bright rim and a sharp highlight.
    let fres = pow(1.0 - abs(dot(n, v)), 3.0);
    let h = normalize(l + v);
    col = frag.albedo * (ambientSky(n, s) * 0.5 + G.sunColor * 0.25) + mix(G.horizonColor, G.zenithColor, 0.5) * fres * 1.2 + G.sunColor * pow(max(dot(n, h), 0.0), 120.0) * 3.0;
  }
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}

// ---- Glint ----

struct GOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) k: f32,
}

@vertex
fn vs_glint(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GOut {
  let g = glints[ii];
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let toCam = G.camPos - g.pos.xyz;
  let d = length(toCam);
  let view = toCam / d;
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), view));
  let up = cross(view, right);
  // Constant on-screen size, so it reads from far away; a slow twinkle with a sharp flash.
  let phase = fract(G.time * 0.33 + g.pos.x * 0.013 + g.pos.z * 0.007);
  let flash = exp(-pow((phase - 0.5) * 9.0, 2.0));
  let size = d * 0.012 * (0.6 + flash * 1.2);
  var out: GOut;
  out.pos = G.viewProj * vec4f(g.pos.xyz + (right * c.x + up * c.y) * size, 1.0);
  out.uv = c;
  // Hidden up close (the thing itself is there to see).
  // Only things in the area around the wind glow: near enough to be worth the detour, never
  // the whole map at once.
  let around = 1.0 - smoothstep(120.0, 170.0, d);
  out.k = g.pos.w * smoothstep(10.0, 22.0, d) * around * (0.4 + flash * 1.2) * exp(-d * G.fogDensity * 0.4);
  return out;
}

@fragment
fn fs_glint(frag: GOut) -> @location(0) vec4f {
  // A four-pointed star with a soft core.
  let a = abs(frag.uv);
  let rays = max(exp(-a.x * 18.0) * (1.0 - a.y), exp(-a.y * 18.0) * (1.0 - a.x));
  let core = exp(-dot(frag.uv, frag.uv) * 14.0);
  let k = (rays * 0.8 + core) * frag.k;
  return vec4f(vec3f(1.0, 0.95, 0.82) * k * 4.0, 0.0);
}
