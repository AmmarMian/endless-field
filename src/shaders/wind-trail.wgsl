// Wind lines: short streaks of air that ride along with the wind, drawn through each line's
// recent positions as a tapered ribbon (thin at both ends), softly additive.
import { Globals } from "./lib/globals.wgsl";

@group(0) @binding(0) var<uniform> G: Globals;
// POINTS positions per line, newest first.
@group(0) @binding(1) var<storage, read> pts: array<vec4f>;
// Per line: x = valid points, y = opacity, z = width (m), w = unused.
@group(0) @binding(2) var<storage, read> lines: array<vec4f>;

const POINTS = 24u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) across: f32,
  @location(2) alpha: f32,
}

fn pointAt(line: u32, i: i32, n: i32) -> vec3f {
  return pts[line * POINTS + u32(clamp(i, 0, n - 1))].xyz;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let info = lines[ii];
  let n = i32(info.x);
  let seg = i32(vi / 6u);
  let corner = vi % 6u;
  let along = array<i32, 6>(0, 1, 0, 0, 1, 1)[corner];
  let sideSign = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0)[corner];
  let i = seg + along;
  let p = pointAt(ii, i, n);
  let dir = pointAt(ii, i - 1, n) - pointAt(ii, i + 1, n);
  let view = normalize(G.camPos - p);
  var side = cross(dir, view);
  side = select(vec3f(0.0, 1.0, 0.0), normalize(side), dot(side, side) > 1e-10);
  let t = f32(i) / max(f32(n - 1), 1.0);
  // Tapered at both ends, fullest just behind the head.
  let taper = sin(3.14159 * pow(clamp(t, 0.0, 1.0), 0.7));
  let valid = select(0.0, 1.0, seg < n - 1);
  var out: VOut;
  let world = p + side * sideSign * info.z * taper;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.across = sideSign;
  out.alpha = info.y * valid * (1.0 - t * 0.5);
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let edge = 1.0 - frag.across * frag.across;
  let tint = mix(vec3f(1.0, 0.97, 0.9), vec3f(0.65, 0.75, 1.0), G.night);
  let fog = exp(-length(G.camPos - frag.world) * G.fogDensity * 2.0);
  let k = frag.alpha * edge * fog * mix(0.8, 0.3, G.night);
  return vec4f(tint * k, 0.0);
}
