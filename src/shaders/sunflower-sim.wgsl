// Sunflower wind physics: each plant is a cantilevered stem carrying a heavy head, reduced to
// its first bending mode (a damped 2D oscillator at the head) plus a torsion mode (the head
// twisting about the stem). Forcing is quadratic aerodynamic drag on the head and leaves from
// the same gusting wind field that moves the grass, plus the player's wind as it passes.
import { Globals, TRAIL_LEN } from "./lib/globals.wgsl";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";

struct Plant {
  // xyz = root, w = scale
  root: vec4f,
  // x = yaw, y = variant, z = seed, w = natural frequency (Hz)
  info: vec4f,
}

struct State {
  // xy = head displacement (world x, z; meters), zw = velocity
  bend: vec4f,
  // x = twist (radians), y = twist rate, z = local wind speed (m/s), w = unused
  twist: vec4f,
}

struct SimParams {
  count: u32,
  dt: f32,
  pad0: f32,
  pad1: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> plants: array<Plant>;
@group(0) @binding(2) var<storage, read_write> state: array<State>;
@group(0) @binding(3) var<uniform> S: SimParams;

const RHO = 1.2;          // air density, kg/m^3
const HEAD_MASS = 0.45;   // effective modal mass of head + upper stem, kg
const GRAVITY = 9.81;

// Wind velocity (m/s) at a point: broad gusts with ripple fronts travelling downwind (the
// grass uses the same field, so a gust visibly crosses the meadow and then the field).
fn windAt(xz: vec2f, t: f32) -> vec2f {
  let wdir = G.windDir;
  let along = dot(xz, wdir);
  let ripple = simplex2d(vec2f(along * 0.085 - t * 0.68, dot(xz, vec2f(-wdir.y, wdir.x)) * 0.03));
  let broad = simplex2d(xz * 0.014 - wdir * t * 0.09);
  let gust = smoothstep(-0.6, 0.9, broad * 0.6 + ripple * 0.55);
  // Small eddies: crosswind turbulence on the scale of a few plants.
  let eddy = vec2f(simplex2d(xz * 0.35 + vec2f(t * 0.9, 0.0)), simplex2d(xz * 0.35 + vec2f(0.0, t * 0.8) + 17.0));
  let speed = G.windStrength * 7.0 * (0.25 + 0.9 * gust);
  return wdir * speed + eddy * speed * 0.18;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= S.count) {
    return;
  }
  let p = plants[i];
  var st = state[i];
  let xz = p.root.xz;
  let scale = p.root.w;
  let height = 1.9 * scale;
  let omega = 6.2831853 * p.info.w;
  // Stem stiffness from the natural frequency, softened by the head's own weight leaning out
  // (P-delta): a displaced heavy head adds a moment that helps the wind.
  let k = HEAD_MASS * omega * omega - HEAD_MASS * GRAVITY / height;
  let c = 2.0 * 0.07 * HEAD_MASS * omega;
  // Drag area (Cd * A) of head + leaves; tall plants catch more.
  let cdA = 0.11 * scale * scale;
  let ktw = 0.02 * pow(6.2831853 * 0.9, 2.0);
  let ctw = 2.0 * 0.12 * 0.02 * 6.2831853 * 0.9;

  var wind = windAt(xz, G.time);
  // The petal stream is a moving pocket of wind: it blows plants outward and along its path.
  let toPlayer = distance(xz, G.playerPos.xz);
  if (toPlayer < 30.0) {
    var push = vec2f(0.0);
    for (var j = 0u; j < TRAIL_LEN; j = j + 1u) {
      let tp = G.trail[j];
      if (tp.w <= 0.001) {
        continue;
      }
      let d = xz - tp.xz;
      let d2 = dot(d, d);
      let radius = 2.4 + G.gust * 1.5;
      let fall = exp(-d2 / (radius * radius));
      let above = tp.y - (p.root.y + height);
      let vert = 1.0 - smoothstep(0.5, 4.5, above);
      push = push + d / max(sqrt(d2), 0.05) * fall * vert * tp.w;
    }
    wind = wind + push * 9.0;
  }

  let dt = min(S.dt, 1.0 / 30.0) * 0.5;
  for (var s = 0; s < 2; s = s + 1) {
    let x = st.bend.xy;
    let v = st.bend.zw;
    // Drag acts on the air speed relative to the moving head (this is also aerodynamic damping).
    let rel = wind - v;
    let drag = 0.5 * RHO * cdA * length(rel) * rel;
    let acc = (drag - k * x - c * v) / HEAD_MASS;
    var nv = v + acc * dt;
    var nx = x + nv * dt;
    // The stem cannot bend past ~40% of its height at the head.
    let lim = 0.4 * height;
    let len = length(nx);
    if (len > lim) {
      nx = nx * (lim / len);
      nv = nv * 0.5;
    }
    st.bend = vec4f(nx, nv);
    // Torsion: crosswind on a head hanging off-axis twists it about the stem.
    let yaw = p.info.x;
    let face = vec2f(sin(yaw), cos(yaw));
    let side = vec2f(face.y, -face.x);
    let torque = dot(drag, side) * 0.12;
    let tw = st.twist.x;
    let twv = st.twist.y + (torque - ktw * tw - ctw * st.twist.y) / 0.02 * dt;
    st.twist = vec4f(clamp(tw + twv * dt, -0.6, 0.6), twv, length(wind), 0.0);
  }
  state[i] = st;
}
