// What the wind carries: seeds, leaves, chaff, snow, spray, pollen and fireflies, each with its
// own shape. Cards are cut out in the fragment shader (alpha to coverage).
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Mote {
  // xyz = position, w = size
  pos: vec4f,
  // rgb = colour, w = seed
  color: vec4f,
  // xyz = velocity, w = kind + fade (fade in [0, 1) as the fraction)
  vel: vec4f,
}

const LEAF = 0u;
const FLUFF = 1u;
const CHAFF = 2u;
const SNOW = 3u;
const SPRAY = 4u;
const POLLEN = 5u;
const FIREFLY = 6u;

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> motes: array<Mote>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) color: vec3f,
  @location(4) @interpolate(flat) kind: u32,
  @location(5) fade: f32,
}

fn rot(axis: vec3f, a: f32, v: vec3f) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let m = motes[ii];
  let kind = u32(m.vel.w);
  let fade = fract(m.vel.w);
  var offs = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let uv = offs[vi];
  let seed = m.color.w;
  let size = m.pos.w;
  var world: vec3f;
  var n: vec3f;
  let view = normalize(G.camPos - m.pos.xyz);
  if (kind == LEAF || kind == CHAFF || kind == 7u || kind == 8u) {
    // Tumbling cards: leaves cupped, chaff a thin sliver.
    var stretch = select(vec2f(0.5, 1.0), vec2f(0.12, 1.0), kind == CHAFF);
    if (kind == 7u || kind == 8u) {
      stretch = vec2f(0.62, 0.85);
    }
    var local = vec3f(uv.x * stretch.x, uv.y * stretch.y, uv.x * uv.x * 0.15 * stretch.x) * size;
    let axis = normalize(vec3f(sin(seed * 12.9), cos(seed * 78.2), sin(seed * 37.7 + 1.0)));
    let spin = G.time * (1.5 + fract(seed * 9.1) * 3.0) + seed * 40.0;
    local = rot(axis, spin, local);
    n = rot(axis, spin, vec3f(0.0, 0.0, 1.0));
    world = m.pos.xyz + local;
  } else {
    // Camera-facing; spray streaks along its velocity.
    var right = normalize(cross(view, vec3f(0.0, 1.0, 0.0)));
    var up = cross(right, view);
    var ext = vec2f(size);
    if (kind == SPRAY) {
      let v = m.vel.xyz - view * dot(m.vel.xyz, view);
      if (dot(v, v) > 1e-4) {
        up = normalize(v);
        right = normalize(cross(up, view));
        ext = vec2f(size * 0.35, size * (1.0 + length(m.vel.xyz) * 0.25));
      }
    }
    world = m.pos.xyz + right * uv.x * ext.x + up * uv.y * ext.y;
    n = view;
  }
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = n;
  out.uv = uv;
  out.color = m.color.rgb;
  out.kind = kind;
  out.fade = fade;
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
  let uv = frag.uv;
  let r = length(uv);
  var alpha = 1.0;
  var base = frag.color;
  var emit = 0.0;
  var lit = 1.0;
  switch (frag.kind) {
    case 0u: {
      // Leaf: pointed oval with a midrib.
      let y = uv.y * 0.5 + 0.5;
      let w = sin(3.14159 * pow(y, 0.8)) * 0.95;
      alpha = 1.0 - smoothstep(w - 0.12, w, abs(uv.x));
      base = base * (0.85 + 0.15 * smoothstep(0.0, 0.08, abs(uv.x)));
    }
    case 1u: {
      // Seed fluff (a dandelion pappus): fine radial filaments around a bright core.
      let a = atan2(uv.y, uv.x);
      let rays = pow(abs(cos(a * 9.0)), 24.0);
      alpha = (smoothstep(1.0, 0.15, r) * (0.25 + rays * 0.75) + smoothstep(0.25, 0.0, r)) * 0.9;
      lit = 0.6;
      emit = 0.25;
    }
    case 3u: {
      alpha = smoothstep(1.0, 0.4, r);
      emit = 0.15;
    }
    case 4u: {
      alpha = smoothstep(1.0, 0.3, r) * 0.7;
      lit = 0.4;
      emit = 0.5;
    }
    case 7u, 8u: {
      // Petal: a soft rounded teardrop, paler at its base, lit through when backlit.
      let y = uv.y * 0.5 + 0.5;
      let w = pow(sin(3.14159 * pow(y, 0.75)), 0.7) * 0.95;
      alpha = 1.0 - smoothstep(w - 0.1, w, abs(uv.x));
      base = mix(mix(base, vec3f(1.0, 0.97, 0.92), 0.45), base, smoothstep(0.0, 0.6, y));
      // Petals from a whole bed glow from within (a soft luminous bloom), pulsing gently.
      emit = select(0.08, 1.5 + 0.4 * sin(G.time * 2.0 + frag.world.x * 3.0), frag.kind == 8u);
      lit = select(1.0, 0.7, frag.kind == 8u);
    }
    case 5u, 6u: {
      // Glowing grains: pollen in the light, fireflies at night.
      alpha = smoothstep(1.0, 0.2, r);
      lit = select(0.6, 0.0, frag.kind == FIREFLY);
      emit = select(1.2, 6.0, frag.kind == FIREFLY) * (1.0 - r * 0.6);
    }
    default: {
      alpha = 1.0 - smoothstep(0.85, 1.0, abs(uv.x));
    }
  }
  alpha *= frag.fade;
  if (alpha < 0.03) {
    discard;
  }
  let l = G.sunDir;
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0);
  var col = base * lit * (ambientSky(n, s) * 0.7 + G.sunColor * (max(dot(n, l), 0.0) * 0.6 + back * 0.8 + 0.08));
  col = col + base * emit * select(mix(0.4, 1.0, G.night), mix(1.1, 1.3, G.night), frag.kind == 8u);
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, alpha);
}
