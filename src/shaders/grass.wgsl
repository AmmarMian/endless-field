// Grass blades, one instance per culled blade. The blade is a tapered strip bent along a
// quadratic Bezier; NSEG picks the LOD (near blades get more segments).
import { Globals } from "./lib/globals.wgsl";
import { Blade, fieldColor } from "./lib/field.wgsl";
import { lanternLight } from "./lib/path.wgsl";
import { SkyParams, applyFogPre, morningFog, ambientSky } from "./lib/atmosphere.wgsl";

override NSEG: u32 = 5u;

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> blades: array<Blade>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  // Morning fog evaluated per vertex (see morningFog).
  @location(14) fog: vec4f,
  // Warm light from the path lanterns at night (per vertex).
  @location(13) lamp: vec3f,
  @location(1) normal: vec3f,
  @location(2) t: f32,
  @location(3) color: vec3f,
  @location(4) groundN: vec3f,
  @location(5) side: f32,
  @location(6) bladeH: f32,
  @location(7) glow: vec3f,
}

fn sky() -> SkyParams {
  return SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
}

fn bezier(p0: vec3f, p1: vec3f, p2: vec3f, t: f32) -> vec3f {
  let a = mix(p0, p1, t);
  let b = mix(p1, p2, t);
  return mix(a, b, t);
}

fn bezierTangent(p0: vec3f, p1: vec3f, p2: vec3f, t: f32) -> vec3f {
  return normalize(2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let b = blades[ii];
  let tri = vi / 3u;
  let corner = vi % 3u;

  // Map (triangle, corner) to (level, side) along the strip; the last triangle is the tip.
  var level = 0u;
  var sideSign = -1.0;
  let lastTri = 2u * (NSEG - 1u);
  if (tri == lastTri) {
    level = select(NSEG - 1u, NSEG, corner == 2u);
    sideSign = select(select(1.0, -1.0, corner == 0u), 0.0, corner == 2u);
  } else {
    let seg = tri / 2u;
    if (tri % 2u == 0u) {
      level = seg + select(0u, 1u, corner == 2u);
      sideSign = select(-1.0, 1.0, corner == 1u);
    } else {
      level = seg + select(1u, 0u, corner == 1u);
      sideSign = select(-1.0, 1.0, corner >= 1u);
    }
  }
  let t = f32(level) / f32(NSEG);

  let root = b.root.xyz;
  let h = b.root.w;
  let facing = vec3f(b.shape.x, 0.0, b.shape.y);
  let rnd = b.bend.z;

  // Natural lean along the blade's own facing normal, plus wind/interaction bend.
  let lean = vec3f(-facing.z, 0.0, facing.x) * (0.18 + rnd * 0.25);
  let flutterPhase = G.time * (6.0 + rnd * 3.5) + rnd * 40.0;
  let flutter = sin(flutterPhase) * 0.05 * G.windStrength * (0.5 + t * t);
  let horiz = vec3f(b.bend.x, 0.0, b.bend.y) + lean + facing * flutter;
  let tipDir = normalize(vec3f(horiz.x, 1.0, horiz.z));
  let p2 = root + tipDir * h;
  let p1 = root + vec3f(0.0, (p2.y - root.y) * 0.95 + h * 0.05, 0.0);

  var pos = bezier(root, p1, p2, t);
  let tangent = bezierTangent(root, p1, p2, max(t, 0.02));
  // Unpack meadow / kind / hue (see Blade.bend.w).
  let packed = b.bend.w;
  let hueQ = floor(packed / 4.0);
  let isSeed = (packed - hueQ * 4.0) >= 2.0;
  let meadow = fract(packed);
  let hue = (hueQ - 7.0) / 7.0;
  var width = b.shape.z * (1.0 - pow(t, 1.4)) * (1.0 + 0.25 * (1.0 - t));
  if (isSeed && NSEG >= 5u) {
    // Thin stalk ending in a slender spindle-shaped seed head.
    let head = sin(3.14159 * clamp((t - 0.58) / 0.42, 0.0, 1.0));
    width = b.shape.z * (0.55 * (1.0 - t * 0.4) + head * 0.75);
  }

  // Keep far blades at least ~1px wide so the field never shimmers into gaps.
  let toCam = G.camPos - pos;
  let camDist = length(toCam);
  let minWidth = camDist * 1.6 / G.viewport.y;
  let w = max(width, minWidth * (1.0 - t));
  pos = pos + facing * sideSign * w;

  var n = normalize(cross(facing, tangent));
  // Face the camera side so lighting is consistent for both faces.
  if (dot(n, toCam) < 0.0) {
    n = -n;
  }

  // Each blade flips to its restored color at its own threshold, so restoration edges are a
  // scattered mix of gold and green blades instead of a smooth colored line.
  let threshold = fract(rnd * 7.31 + b.root.x * 0.013) * 0.7 + 0.1;
  let life = smoothstep(threshold - 0.12, threshold + 0.12, b.shape.w);
  var out: VOut;
  out.pos = G.viewProj * vec4f(pos, 1.0);
  out.world = pos;
  out.lamp = lanternLight(out.world, G.night, G.time, G.lamps);
  out.fog = morningFog(out.world, G.camPos, SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor), vec4f(G.mist, G.mistBase, G.canopy, G.time));
  out.normal = n;
  out.t = t;
  out.side = sideSign;
  out.bladeH = h;
  // Night: some restored blades glow at the tip, teal through warm gold.
  let sparkle = step(0.72, fract(rnd * 13.7));
  let pulse = 0.6 + 0.4 * sin(G.time * 1.3 + rnd * 40.0);
  out.glow = mix(vec3f(0.25, 0.9, 0.7), vec3f(1.0, 0.75, 0.3), fract(rnd * 5.3)) * life * sparkle * pulse * G.night * 1.6;
  var color = fieldColor(meadow, life, t, hue) * mix(0.85, 1.15, rnd);
  if (isSeed && t > 0.6) {
    color = mix(color, vec3f(0.46, 0.33, 0.15) * mix(1.0, 0.8, life), 0.45);
  }
  out.color = color;
  out.groundN = b.ground.xyz;
  return out;
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = sky();
  let v = normalize(G.camPos - frag.world);
  // Rounded blade: bend the normal across the width, then blend toward the ground normal
  // with height so the field shades like a continuous surface from afar.
  var n = normalize(frag.normal + cross(frag.normal, vec3f(0.0, 1.0, 0.0)) * frag.side * 0.6);
  let dist = length(G.camPos - frag.world);
  n = normalize(mix(n, frag.groundN, clamp(0.35 + dist / 120.0, 0.0, 0.85)));

  let l = G.sunDir;
  let albedo = frag.color;
  let ndl = max(dot(n, l), 0.0) * 0.8 + 0.2 * max(dot(frag.groundN, l), 0.0);
  // Thin blades glow when the sun is behind them.
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0) * (0.35 + 0.65 * frag.t);
  let trans = back * 1.4 * albedo * vec3f(1.0, 1.05, 0.7);
  let hlf = normalize(l + v);
  let spec = pow(max(dot(n, hlf), 0.0), 40.0) * 0.18 * frag.t;
  // AO by absolute height above the ground, so short lawn blades are not all shadow.
  let ao = mix(0.25, 1.0, smoothstep(0.0, 0.55, frag.t * frag.bladeH));

  var col = albedo * (ambientSky(n, s) * 0.55 * ao + G.sunColor * ndl * mix(0.6, 1.0, frag.t));
  col = col + G.sunColor * (trans + vec3f(spec));
  col = col + frag.glow * pow(frag.t, 6.0);
  col = col + albedo * frag.lamp;
  col = applyFogPre(col, frag.world, G.camPos, G.fogDensity, s, frag.fog);
  return vec4f(col, 1.0);
}
