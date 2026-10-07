// Insect swarms: tiny specks catching the light, drawn as small additive glows that stay a
// few pixels wide at any distance (so a swarm reads as a shimmer from afar).
import { Globals } from "./lib/globals.wgsl";

@group(0) @binding(0) var<uniform> G: Globals;
// xyz = position, w = brightness (0 hidden)
@group(0) @binding(1) var<storage, read> specks: array<vec4f>;

// Thermals reuse this shader with larger, paler specks.
override SIZE_K: f32 = 1.0;
override WARMTH: f32 = 1.0;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) k: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let sp = specks[ii];
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let toCam = G.camPos - sp.xyz;
  let d = length(toCam);
  let view = toCam / d;
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), view));
  let up = cross(view, right);
  let size = (0.018 + d * 0.0022) * SIZE_K;
  var out: VOut;
  out.pos = G.viewProj * vec4f(sp.xyz + (right * c.x + up * c.y) * size, 1.0);
  out.uv = c;
  out.k = sp.w * exp(-d * G.fogDensity * 0.8) * (1.0 - smoothstep(60.0, 110.0, d));
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let glow = exp(-r2 * 7.0);
  // Warm in the low sun and at dusk, paler at midday.
  let tint = mix(vec3f(0.95, 0.95, 0.92), mix(vec3f(1.0, 0.92, 0.7), vec3f(1.0, 0.75, 0.4), G.night * 0.5 + 0.3), WARMTH);
  return vec4f(tint * glow * frag.k * mix(0.9, 1.6, G.night), 0.0);
}
