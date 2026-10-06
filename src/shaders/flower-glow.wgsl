// Soft additive halos around closed buds. They grow with distance so unbloomed flowers stay
// visible across the field, and fade out as the flower opens.
import { Globals } from "./lib/globals.wgsl";

struct Flower {
  root: vec4f,
  color: vec4f,
  shape: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> flowers: array<Flower>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let f = flowers[ii];
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let head = f.root.xyz + vec3f(0.0, f.shape.x + 0.05, 0.0);
  let dist = distance(head, G.camPos);
  let closed = 1.0 - smoothstep(0.0, 0.6, f.root.w);
  let pulse = 0.75 + 0.25 * sin(G.time * 2.2 + f.color.w * 17.0);
  let size = (0.35 + dist * mix(0.014, 0.009, G.night)) * mix(0.0, 1.0, closed);
  // Camera-facing billboard from the view matrix rows encoded in viewProj's inverse.
  let right = normalize(cross(head - G.camPos, vec3f(0.0, 1.0, 0.0)));
  let up = normalize(cross(right, head - G.camPos));
  let world = head + (right * c.x + up * c.y) * size;
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.uv = c;
  // Distant beacons stay readable through the fog but never overpower the scene.
  let range = 1.0 - smoothstep(120.0, 190.0, dist);
  out.color = f.color.rgb * closed * pulse * range * mix(0.9, 0.55, smoothstep(5.0, 60.0, dist)) * (1.0 + G.night * 0.4);
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let core = exp(-r2 * 9.0);
  let halo = exp(-r2 * 3.0) * 0.35;
  let a = (core + halo) * (1.0 - smoothstep(0.85, 1.0, r2));
  return vec4f(frag.color * a, a);
}
