// River gates for the kingfisher's course (model: tools/blender/model_river_gate.py): floating
// straw rope hoops with fluttering paper streamers, glowing softly over the water. The next
// gate to fly through pulses gold; gates already passed glow warm.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

struct Gate {
  // xyz = base (at the water), w = yaw
  pos: vec4f,
  // x = state (0 idle, 1 next, 2 passed), y = flash (0..1, just passed), zw unused
  state: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> gates: array<Gate>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) ao: f32,
  @location(4) @interpolate(flat) part: u32,
  @location(5) @interpolate(flat) state: vec2f,
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let g = gates[ii];
  let part = u32(round(p.w));
  var lp = p.xyz;
  if (part == 2u) {
    // Streamers flutter, more toward their tips.
    let k = t.y;
    let ph = G.time * 6.0 + t.x * 1.9 + f32(ii) * 0.7;
    lp += vec3f(sin(ph + k * 9.0) * 0.05, 0.0, sin(ph * 0.8 + k * 7.0) * 0.12 + 0.06) * k * 2.5;
  }
  let cy = cos(g.pos.w);
  let sy = sin(g.pos.w);
  let r = vec3f(lp.x * cy + lp.z * sy, lp.y, -lp.x * sy + lp.z * cy);
  let rn = vec3f(n.x * cy + n.z * sy, n.y, -n.x * sy + n.z * cy);
  let world = g.pos.xyz + r;
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = rn;
  out.albedo = pow(e.rgb, vec3f(2.2));
  out.ao = e.a;
  out.part = part;
  out.state = g.state.xy;
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
  var col = frag.albedo * (ambientSky(n, s) * 0.8 * frag.ao + G.sunColor * wrap * 0.85);
  if (frag.part == 2u) {
    col += frag.albedo * G.sunColor * pow(clamp(dot(-v, l), 0.0, 1.0), 2.0) * 0.6;
  }
  if (frag.part == 1u || frag.part == 2u) {
    // Lit rope: gold pulse on the next gate, a steady warm glow once flown through.
    let next = select(0.0, 0.6 + 0.4 * sin(G.time * 4.0), frag.state.x > 0.5 && frag.state.x < 1.5);
    let passed = select(0.0, 1.2, frag.state.x > 1.5);
    // Every ring glows on its own (it floats by some magic of the wind); bright enough to
    // bloom, so it reads from far down the river.
    let glow = 0.9 + next * 3.0 + passed + frag.state.y * 5.0;
    col += vec3f(1.0, 0.72, 0.3) * glow * mix(0.8, 1.4, G.night);
  }
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}
