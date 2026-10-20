// Whale blows: a column of mist and spray (soft billboards) rising from the blowhole,
// spreading and drifting downwind as it fades (particles simulated in src/world/whales.ts).
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog } from "./lib/atmosphere.wgsl";

struct Puff {
  // xyz = centre, w = radius
  pos: vec4f,
  // x = opacity, y = random, z = age 0..1
  info: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> puffs: array<Puff>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) info: vec4f,
  @location(2) world: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = puffs[ii];
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let view = normalize(G.camPos - p.pos.xyz);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), view));
  let up = cross(view, right);
  var out: VOut;
  out.world = p.pos.xyz + (right * c.x + up * c.y) * p.pos.w;
  out.pos = G.viewProj * vec4f(out.world, 1.0);
  out.uv = c;
  out.info = p.info;
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let r = length(frag.uv);
  // A soft, lumpy puff (a few lobes so it never reads as a disc).
  let a = atan2(frag.uv.y, frag.uv.x);
  let lobes = 0.85 + 0.15 * sin(a * 5.0 + frag.info.y * 40.0) * sin(a * 3.0 - frag.info.y * 20.0);
  let d = smoothstep(lobes, lobes * 0.25, r);
  let alpha = d * frag.info.x;
  if (alpha < 0.003) {
    discard;
  }
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  // Lit mist: bright toward the sun, a cool sky-lit body.
  let v = normalize(G.camPos - frag.world);
  let forward = pow(max(dot(-v, G.sunDir), 0.0), 4.0);
  var col = G.sunColor * (0.55 + forward * 0.9) + G.zenithColor * 0.55 + G.horizonColor * 0.2;
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col * alpha, alpha);
}
