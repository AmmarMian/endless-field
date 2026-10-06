// The wind made visible: a handful of thin, curling strands of brighter air that trail the
// wind along its recent path, twisting around it, widening into the wake and flowing back.
import { Globals } from "./lib/globals.wgsl";

struct Trail {
  /** Valid path samples. */
  count: f32,
  /** 0 calm .. 1 full gust: brightness and length. */
  strength: f32,
  /** Meters between path samples. */
  step: f32,
  /** Arc length (m) the flow pattern has travelled, so it streams with the wind's speed. */
  flow: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> path: array<vec4f>;
@group(0) @binding(2) var<uniform> T: Trail;

const SAMPLES = 128u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) across: f32,
  @location(2) alpha: f32,
  @location(3) arc: f32,
  @location(4) seed: f32,
}

fn hash11(n: f32) -> f32 {
  return fract(sin(n * 127.1 + 311.7) * 43758.5453);
}

fn pathAt(i: i32) -> vec3f {
  let n = i32(T.count);
  return path[u32(clamp(i, 0, max(n - 1, 0)))].xyz;
}

/** A point on strand `s` at path sample `i`. */
fn strandPoint(i: i32, s: f32) -> vec3f {
  let p = pathAt(i);
  var t = pathAt(i - 1) - pathAt(i + 1);
  t = select(vec3f(0.0, 0.0, 1.0), normalize(t), dot(t, t) > 1e-6);
  let side = normalize(cross(t, vec3f(0.0, 1.0, 0.0)) + vec3f(1e-4, 0.0, 0.0));
  let up = cross(side, t);
  let arc = f32(i) * T.step;
  let r0 = 0.25 + hash11(s * 3.1) * 0.7;
  // The wake opens out behind the wind, and each strand breathes.
  let r = r0 * (0.55 + 0.45 * smoothstep(0.0, 8.0, arc)) * (0.75 + 0.25 * sin(arc * 0.45 + s * 4.0 + G.time * 0.7));
  let twist = (0.07 + hash11(s * 7.7) * 0.12) * 6.2831;
  let a = hash11(s * 1.9) * 6.2831 + arc * twist - G.time * (0.8 + hash11(s * 5.3));
  // Strands sit mostly beside and above the path (flattened orbit), never in the grass.
  return p + side * cos(a) * r + up * (sin(a) * r * 0.55 + 0.1);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let seg = i32(vi / 6u);
  let corner = vi % 6u;
  let along = array<i32, 6>(0, 1, 0, 0, 1, 1)[corner];
  let sideSign = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0)[corner];
  let i = seg + along;
  let s = f32(ii) + 1.0;
  let p = strandPoint(i, s);
  let dir = strandPoint(i + 1, s) - strandPoint(i - 1, s);
  let view = normalize(G.camPos - p);
  var side = cross(dir, view);
  side = select(vec3f(1.0, 0.0, 0.0), normalize(side), dot(side, side) > 1e-8);

  // Each strand starts a little behind the wind and runs a few meters; gusts stretch them.
  let arc = f32(i) * T.step;
  let start = 0.3 + hash11(s * 2.3) * 2.5;
  let len = (5.0 + hash11(s * 9.1) * 9.0) * (0.8 + T.strength * 0.7);
  let u = (arc - start) / len;
  var alpha = smoothstep(0.0, 0.18, u) * (1.0 - smoothstep(0.55, 1.0, u));
  alpha *= select(0.0, 1.0, f32(i) < T.count - 1.0);
  // Strands passing right by the lens fade, instead of smearing across the screen.
  let camDist = length(G.camPos - p);
  alpha *= smoothstep(1.2, 3.0, camDist);
  let width = (0.04 + hash11(s * 4.4) * 0.04) * (1.0 + 0.5 * T.strength) * mix(1.0, 0.5, u);

  var out: VOut;
  let world = p + side * sideSign * width;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.across = sideSign;
  out.alpha = alpha;
  out.arc = arc;
  out.seed = s;
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let edge = 1.0 - frag.across * frag.across;
  // Longer bright filaments flow back along each strand at the wind's own speed.
  let flow = fract((frag.arc + T.flow) * 0.11 + frag.seed * 0.37);
  let filament = 0.35 + 0.65 * smoothstep(0.0, 0.25, flow) * (1.0 - smoothstep(0.45, 0.95, flow));
  // Warm daylight air; a faint moonlit silver at night.
  let tint = mix(vec3f(1.0, 0.95, 0.85), vec3f(0.6, 0.72, 1.0), G.night);
  let fog = exp(-length(G.camPos - frag.world) * G.fogDensity * 2.0);
  let k = frag.alpha * edge * edge * filament * (0.12 + 0.3 * T.strength) * fog * mix(1.0, 0.45, G.night);
  return vec4f(tint * k, 0.0);
}
