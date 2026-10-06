// Precipitation around the camera: rain streaks slanted by the wind, or (in winter) snowflakes
// drifting down. Each instance lives in a world-anchored box that wraps around the camera, so
// drops stay put in the world as you move and fall at their own pace.
import { Globals } from "./lib/globals.wgsl";
import { seasonWeights } from "./lib/season.wgsl";
import { pcg3d, unitFloat } from "@vgpu/wgsl-std/hash";

@group(0) @binding(0) var<uniform> G: Globals;

const BOX = vec3f(36.0, 24.0, 36.0);

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) alpha: f32,
  @location(2) @interpolate(flat) snow: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let h = pcg3d(vec3u(ii, ii * 7u + 3u, 0x9E3779B9u));
  let r = vec3f(unitFloat(h.x), unitFloat(h.y), unitFloat(h.z));
  var out: VOut;
  // Only a share of the drops fall, by intensity.
  if (r.x * 0.999 > G.rain) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }
  let snow = smoothstep(0.4, 0.8, seasonWeights(G.season).z);
  let speed = mix(9.5, 1.1, snow) * mix(0.85, 1.15, r.y);
  let wind = vec3f(G.windDir.x, 0.0, G.windDir.y) * G.windStrength * mix(2.6, 1.4, snow);
  let vel = vec3f(wind.x, -speed, wind.z);
  // World position: a fixed spot per drop in a box tiled over the world, falling over time.
  var p = r * BOX + vel * G.time;
  // Snow wanders as it falls.
  p = p + vec3f(sin(G.time * 0.9 + r.z * 40.0), 0.0, cos(G.time * 0.7 + r.x * 40.0)) * 0.6 * snow;
  let local = p - G.camPos + BOX * 0.5;
  let wrapped = local - floor(local / BOX) * BOX - BOX * 0.5;
  let center = G.camPos + wrapped;
  let toCam = G.camPos - center;
  let dist = length(toCam);
  // Rain: a thin streak along the velocity (motion blur). Snow: a small round flake.
  let along = normalize(vel);
  let side = normalize(cross(along, toCam / max(dist, 1e-3)));
  let streakLen = mix(0.38, 0.035, snow);
  let width = mix(0.008, 0.035, snow) * (1.0 + dist * 0.015);
  var world = center + side * c.x * width + along * c.y * streakLen;
  if (snow > 0.5) {
    let right = normalize(cross(vec3f(0.0, 1.0, 0.0), toCam));
    let up = cross(toCam / max(dist, 1e-3), right);
    world = center + (right * c.x + up * c.y) * width;
  }
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.uv = c;
  // Fade very near (no drops in your face) and toward the box edge.
  out.alpha = smoothstep(0.6, 2.0, dist) * (1.0 - smoothstep(12.0, 17.0, dist));
  out.snow = snow;
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let across = 1.0 - abs(frag.uv.x);
  let shape = select(across * (1.0 - abs(frag.uv.y) * 0.6), 1.0 - smoothstep(0.3, 1.0, length(frag.uv)), frag.snow > 0.5);
  // Lit by the overcast sky; a little brighter for snow; dim at night.
  let light = (G.horizonColor * 0.9 + G.zenithColor * 0.5 + G.sunColor * 0.08) * mix(0.55, 1.4, frag.snow);
  let a = shape * frag.alpha * mix(0.4, 0.55, frag.snow);
  return vec4f(light * a, 0.0);
}
