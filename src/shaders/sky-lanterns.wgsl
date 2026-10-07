// Sky lanterns: paper lanterns lit from within, rising slowly and drifting on the wind.
// Camera-facing cards cut to a lantern's rounded silhouette; additive, warm, flickering.
import { Globals } from "./lib/globals.wgsl";

struct SkyLantern {
  // xyz = position, w = fade (0..1)
  pos: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> lanterns: array<SkyLantern>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) k: f32,
  @location(2) flicker: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let l = lanterns[ii];
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let toCam = G.camPos - l.pos.xyz;
  let d = length(toCam);
  let view = toCam / d;
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), view));
  let up = vec3f(0.0, 1.0, 0.0);
  // The card is larger than the lantern, leaving room for its halo.
  let size = vec2f(0.55, 0.7);
  var out: VOut;
  out.pos = G.viewProj * vec4f(l.pos.xyz + right * c.x * size.x + up * c.y * size.y, 1.0);
  out.uv = c;
  out.k = l.pos.w * exp(-d * G.fogDensity * 0.5);
  out.flicker = 0.85 + 0.15 * sin(G.time * 9.0 + f32(ii) * 3.1) * sin(G.time * 5.3 + f32(ii));
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  // Lantern body: a soft-edged rounded box in the middle of the card, brightest at the base
  // where the flame is; a warm halo around it.
  let p = frag.uv * vec2f(0.55, 0.7);
  let q = abs(p - vec2f(0.0, 0.02)) - vec2f(0.13, 0.2);
  let box = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - 0.05;
  let body = 1.0 - smoothstep(-0.01, 0.015, box);
  let flame = exp(-pow((p.y + 0.12) * 6.0, 2.0)) * 0.6 + 0.55;
  let halo = exp(-max(box, 0.0) * 9.0) * 0.35;
  let col = vec3f(1.0, 0.55, 0.22) * (body * flame * 2.4 + halo) * frag.flicker;
  return vec4f(col * frag.k * mix(0.5, 1.0, G.night), 0.0);
}
