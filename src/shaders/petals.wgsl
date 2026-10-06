// The petal stream: tumbling, cupped petal cards, alpha-to-coverage edges, soft glow.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Petal {
  // xyz = position, w = size
  pos: vec4f,
  // rgb = color, w = seed
  color: vec4f,
  // xyz = velocity (for stretch / orientation), w = fade in
  vel: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> petals: array<Petal>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) color: vec3f,
  @location(4) glow: f32,
}

fn rot(axis: vec3f, a: f32, v: vec3f) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = petals[ii];
  // 2x6 grid covering the petal's bounding box; the outline is cut in the fragment shader.
  let quad = vi / 6u;
  let corner = vi % 6u;
  var offs = array<vec2u, 6>(vec2u(0u, 0u), vec2u(1u, 0u), vec2u(0u, 1u), vec2u(0u, 1u), vec2u(1u, 0u), vec2u(1u, 1u));
  let cell = vec2u(quad % 2u, quad / 2u) + offs[corner];
  let uv = vec2f(f32(cell.x) - 1.0, f32(cell.y) / 6.0);

  let seed = p.color.w;
  let size = p.pos.w;
  var local = vec3f(uv.x * 0.32, uv.y - 0.4, uv.x * uv.x * 0.16 - uv.y * uv.y * 0.18) * size;

  // Tumble around a seeded axis, faster when the stream is fast.
  let t = G.time;
  let axis = normalize(vec3f(sin(seed * 12.9), cos(seed * 78.2), sin(seed * 37.7 + 1.0)));
  let spin = t * (1.2 + fract(seed * 9.1) * 2.0) + seed * 40.0;
  local = rot(axis, spin, local);
  var n = rot(axis, spin, normalize(vec3f(-uv.x * 0.6, 0.15, 1.0)));

  let world = p.pos.xyz + local;
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = n;
  out.uv = uv;
  out.color = p.color.rgb;
  out.glow = p.vel.w;
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  let v = normalize(G.camPos - frag.world);
  var n = normalize(frag.normal);
  if (dot(n, v) < 0.0) {
    n = -n;
  }
  // Teardrop outline, widest past the middle, with a small notch at the tip.
  let y = frag.uv.y;
  let outline = pow(sin(3.14159 * pow(y, 0.72)), 0.75) * (1.0 - 0.18 * smoothstep(0.85, 1.0, y) * (1.0 - abs(frag.uv.x) * 3.0));
  let d = abs(frag.uv.x) - outline;
  let alpha = 1.0 - smoothstep(-0.06, 0.0, d);
  if (alpha < 0.02) {
    discard;
  }
  let base = mix(mix(frag.color, vec3f(1.0, 0.96, 0.9), 0.5), frag.color, smoothstep(0.0, 0.7, frag.uv.y));
  let l = G.sunDir;
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0);
  var col = base * (ambientSky(n, s) * 0.7 + G.sunColor * (max(dot(n, l), 0.0) * 0.6 + back * 0.6 + 0.08));
  // Daylight: lit naturally with only a faint self-glow; the glow grows at night.
  col = col + base * (mix(0.0, 0.7, G.night) + 0.1 * frag.glow);
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s);
  return vec4f(col, alpha);
}
