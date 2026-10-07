// The swallows' nest under the torii beam: a cup of mud pellets with a straw-flecked rim, and
// the chicks peeping over it (heads bob, yellow gapes open when the parent is near).
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Nest {
  // xyz = position (the cup's rim centre), w = yaw
  pos: vec4f,
  // x = chicks shown (0..4), y = begging (0..1, parent near), z = scale, w = unused
  info: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> N: Nest;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
}

// Vertex attributes: p.xyz model position, p.w part (0 mud, 1 chick head, 2 gape, 3 eye,
// 4 straw); c.x = chick index (for heads), c.yzw unused.
@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) c: vec4f) -> VOut {
  let part = u32(round(p.w));
  var lp = p.xyz;
  var ln = n.xyz;
  var hide = false;
  if (part >= 1u && part <= 3u) {
    let i = c.x;
    hide = i >= N.info.x - 0.5;
    // Heads bob up out of the cup, beaks wide when the parent is near.
    let beg = N.info.y;
    let bob = (0.012 + 0.02 * beg) * max(0.0, sin(G.time * (4.0 + i * 1.3) + i * 2.0)) + 0.01 * beg;
    lp.y += bob;
    if (part == 2u) {
      // The gape opens wider when begging.
      lp = lp + vec3f(0.0, beg * 0.004, 0.0);
    }
  }
  let s = N.info.z;
  let cs = vec2f(cos(N.pos.w), sin(N.pos.w));
  let w = vec3f(cs.x * lp.x + cs.y * lp.z, lp.y, -cs.y * lp.x + cs.x * lp.z) * s;
  let wn = vec3f(cs.x * ln.x + cs.y * ln.z, ln.y, -cs.y * ln.x + cs.x * ln.z);
  var out: VOut;
  let world = N.pos.xyz + w;
  out.pos = select(G.viewProj * vec4f(world, 1.0), vec4f(0.0, 0.0, -2.0, 1.0), hide);
  out.world = world;
  out.normal = wn;
  var albedo = vec3f(0.33, 0.24, 0.16);
  if (part == 1u) {
    albedo = vec3f(0.22, 0.2, 0.22);
  } else if (part == 2u) {
    albedo = vec3f(1.0, 0.82, 0.25);
  } else if (part == 3u) {
    albedo = vec3f(0.02, 0.02, 0.02);
  } else if (part == 4u) {
    albedo = vec3f(0.72, 0.62, 0.38);
  }
  out.albedo = albedo * (0.85 + 0.3 * fract(sin(dot(p.xyz, vec3f(127.1, 311.7, 74.7))) * 43758.5453));
  return out;
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  let l = G.sunDir;
  let wrap = clamp((dot(n, l) + 0.3) / 1.3, 0.0, 1.0);
  // Under the beam it is mostly in shade: sky light, a little sun.
  var col = frag.albedo * (ambientSky(n, s) * 0.85 + G.sunColor * wrap * 0.35);
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}
