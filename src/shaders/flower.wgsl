// Procedural flowers. One shared template mesh carries parametric coordinates; the vertex
// shader shapes stem, leaves, petals and center from per-flower bloom, wind and the stream.
import { Globals, TRAIL_LEN } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky } from "./lib/atmosphere.wgsl";

// Scene pass outputs: colour, plus reversed-Z depth for depth of field (an MSAA depth
// buffer cannot be sampled after the pass, so depth is written out as a colour too).
struct SceneOut {
  @location(0) color: vec4f,
  @location(1) depth: vec4f,
}


struct Flower {
  // xyz = root, w = bloom (0 closed bud .. 1 open)
  root: vec4f,
  // rgb = petal color, w = seed
  color: vec4f,
  // x = height, y = petal length, z = glow, w = petal count
  shape: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> flowers: array<Flower>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) emissive: vec3f,
  @location(4) @interpolate(flat) part: u32,
  @location(5) param: vec2f,
}

const PI: f32 = 3.14159265;

fn sky() -> SkyParams {
  return SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
}

fn rotY(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn streamPush(xz: vec2f, y: f32) -> vec2f {
  var push = vec2f(0.0);
  for (var i = 0u; i < TRAIL_LEN; i = i + 1u) {
    let tp = G.trail[i];
    if (tp.w <= 0.001) {
      continue;
    }
    let d = xz - tp.xz;
    let d2 = dot(d, d);
    let fall = exp(-d2 / 7.0) * (1.0 - smoothstep(1.5, 6.0, tp.y - y));
    push = push + d / max(sqrt(d2), 0.05) * fall * tp.w;
  }
  return push;
}

@vertex
fn vs_main(@location(0) a: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let f = flowers[ii];
  let root = f.root.xyz;
  let bloom = f.root.w;
  let seed = f.color.w;
  let height = f.shape.x;
  let part = u32(a.x + 0.5);

  // Stem curve: wind sway plus being pushed by the passing stream.
  let t = G.time;
  let sway = G.windDir * G.windStrength * (0.10 + 0.06 * sin(t * 1.7 + seed * 30.0))
    + vec2f(sin(t * 2.3 + seed * 11.0), cos(t * 1.9 + seed * 7.0)) * 0.025;
  let bend = sway + streamPush(root.xz, root.y) * 0.45;
  let tipDir = normalize(vec3f(bend.x, 1.0, bend.y));
  let top = root + tipDir * height;
  let ctrl = root + vec3f(0.0, height * 0.8, 0.0);
  let up = normalize(top - ctrl);

  var pos = root;
  var n = vec3f(0.0, 1.0, 0.0);
  var albedo = vec3f(0.0);
  var emissive = vec3f(0.0);
  let petalColor = f.color.rgb;
  let budGlow = f.shape.z;

  // Frame at the flower head.
  let side = normalize(cross(up, vec3f(0.0, 0.0, 1.0)));
  let fwd = cross(side, up);

  if (part == 0u) {
    // Stem ribbon: a.y = t along, a.z = side
    let s = a.y;
    let p0 = mix(root, ctrl, s);
    let p1 = mix(ctrl, top, s);
    let c = mix(p0, p1, s);
    let tangent = normalize(mix(ctrl - root, top - ctrl, s));
    let toCam = normalize(G.camPos - c);
    let across = normalize(cross(tangent, toCam));
    pos = c + across * a.z * mix(0.008, 0.005, s);
    n = normalize(cross(across, tangent));
    albedo = vec3f(0.05, 0.13, 0.03);
  } else if (part == 3u) {
    // Leaves: a.y = along, a.z = across, a.w = leaf index
    let li = a.w;
    let along = a.y;
    let attach = mix(root, top, 0.22 + 0.2 * li);
    let ang = seed * 20.0 + li * 2.4;
    let dir = rotY(vec3f(1.0, 0.0, 0.0), ang);
    let len = 0.22 * (0.8 + 0.4 * fract(seed * 13.0 + li));
    let droop = along * along * 0.12;
    let width = 0.035 * sin(PI * pow(along, 0.7));
    let perp = rotY(vec3f(0.0, 0.0, 1.0), ang);
    pos = attach + dir * along * len + vec3f(0.0, along * 0.06 - droop, 0.0) + perp * a.z * width;
    n = normalize(cross(perp, dir + vec3f(0.0, 0.25 - along * 0.5, 0.0)));
    albedo = vec3f(0.06, 0.16, 0.035);
  } else if (part == 1u) {
    // Petals: a.y = along (u), a.z = across (v), a.w = petal index
    let u = a.y;
    let v = a.z;
    let count = f.shape.w;
    if (a.w >= count) {
      // Unused petal slot for flowers with fewer petals: collapse it.
      var dead: VOut;
      dead.pos = vec4f(0.0, 0.0, -2.0, 1.0);
      return dead;
    }
    let b = smoothstep(0.0, 1.0, bloom);
    let phi = a.w / count * 2.0 * PI + seed * 6.0;
    let len = f.shape.y * mix(0.62, 1.0, b);
    // Elevation from the head axis: a tight bud curling inward, opening into a cup.
    let open = mix(0.12, 1.15 + 0.15 * fract(seed * 5.0), b);
    let curl = mix(-0.55, 0.65, b);
    let theta = open + curl * u * 0.5;
    let radial = vec3f(cos(phi), 0.0, sin(phi));
    let tangentDir = vec3f(-sin(phi), 0.0, cos(phi));
    let spine = radial * sin(theta) * u * len + vec3f(0.0, cos(theta) * u * len, 0.0);
    let width = len * 0.42 * pow(sin(PI * pow(u, 0.75)), 0.6) * mix(0.55, 1.0, b);
    let cup = vec3f(0.0, 1.0, 0.0) * v * v * width * 0.35;
    let local = spine + tangentDir * v * width + cup + radial * 0.012;
    pos = top + side * local.x + up * local.y + fwd * local.z;
    let spineDir = radial * sin(theta) + vec3f(0.0, cos(theta), 0.0);
    let ln = normalize(cross(tangentDir, spineDir));
    n = normalize(side * ln.x + up * ln.y + fwd * ln.z);
    // Pale throat to saturated tips.
    albedo = mix(mix(petalColor, vec3f(1.0, 0.97, 0.88), 0.55), petalColor, smoothstep(0.1, 0.8, u));
    let pulse = (0.65 + 0.35 * sin(t * 2.2 + seed * 17.0)) * (1.0 - smoothstep(45.0, 90.0, distance(top, G.camPos)));
    emissive = petalColor * (budGlow * (1.0 - b) * pulse * 3.0 * (1.0 + G.night) + 0.1 + G.night * b * (0.8 + 0.4 * pulse));
  } else {
    // Center disk: a.y = radius, a.z = angle
    let r = a.y * f.shape.y * 0.24 * mix(0.6, 1.0, bloom);
    let local = vec3f(cos(a.z) * r, 0.012 + (1.0 - a.y) * 0.01, sin(a.z) * r);
    pos = top + side * local.x + up * local.y + fwd * local.z;
    n = up;
    albedo = vec3f(0.9, 0.62, 0.12);
    emissive = vec3f(0.9, 0.55, 0.1) * (0.25 + G.night * 1.2) * bloom;
  }

  var out: VOut;
  out.pos = G.viewProj * vec4f(pos, 1.0);
  out.world = pos;
  out.normal = n;
  out.albedo = albedo;
  out.emissive = emissive;
  out.part = part;
  out.param = a.yz;
  return out;
}

fn fs_mainColor(frag: VOut) -> vec4f {
  let s = sky();
  let v = normalize(G.camPos - frag.world);
  var n = normalize(frag.normal);
  if (dot(n, v) < 0.0) {
    n = -n;
  }
  let l = G.sunDir;
  let ndl = max(dot(n, l), 0.0);
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 4.0);
  var trans = 0.6;
  if (frag.part == 1u) {
    trans = 1.6;
  }
  var col = frag.albedo * (ambientSky(n, s) * 0.6 + G.sunColor * (ndl * 0.85 + back * trans * 0.5));
  col = col + frag.emissive;
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}

@fragment
fn fs_main(frag: VOut) -> SceneOut {
  return SceneOut(fs_mainColor(frag), vec4f(frag.pos.z, 0.0, 0.0, 1.0));
}
