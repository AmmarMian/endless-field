// Fireflies: world-anchored cells around the camera, each maybe holding one firefly that
// wanders on a noise path, bobs above the grass and blinks. More gather over flower beds.
import { Globals } from "./lib/globals.wgsl";
import { riverInfo, terrainHeightM } from "./lib/terrain.wgsl";
import { biome } from "./lib/biome.wgsl";
import { bedMask } from "./lib/beds.wgsl";
import { pcg2d, unitFloat } from "@vgpu/wgsl-std/hash";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";

struct FlyParams {
  centerCell: vec2i,
  gridSize: u32,
  cellSize: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> F: FlyParams;
@group(0) @binding(2) var mtnTex: texture_2d_array<f32>;
@group(0) @binding(3) var mtnSamp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
  let gx = i32(ii % F.gridSize);
  let gz = i32(ii / F.gridSize);
  let cell = F.centerCell + vec2i(gx, gz) - vec2i(i32(F.gridSize / 2u));
  let h = pcg2d(bitcast<vec2u>(cell) ^ vec2u(0xA511E9B3u, 0x63D83595u));
  let base = (vec2f(cell) + vec2f(unitFloat(h.x), unitFloat(h.y))) * F.cellSize;
  // Where fireflies gather: drifting swarm pockets, flower beds, riverbanks, forest edges.
  // Evaluate the cheap swarm noise first so most empty cells exit early.
  if (G.night < 0.02) {
    return out;
  }
  let roll = unitFloat(h.x ^ h.y);
  let swarmN = simplex2d(base * 0.025 + vec2f(G.time * 0.01, 3.0));
  let swarm = smoothstep(0.35, 0.75, swarmN);
  var density = 0.025 + swarm * 0.35;
  if (roll > density) {
    let bed = bedMask(base).x;
    let river = riverInfo(base);
    let bank = 1.0 - smoothstep(river.z * 0.8, river.z * 3.0, abs(river.x - river.z));
    let bio = biome(base);
    density = density + bed * 0.45 + bank * 0.3 + bio.z * (1.0 - bio.z) * 1.0;
    if (roll > density) {
      return out;
    }
  }
  let seed = unitFloat(h.y * 747796405u);
  // Lazy drift: slow noise wander, hovering just over the grass tops.
  let t = G.time * 0.08 + seed * 100.0;
  let wander = vec2f(simplex2d(vec2f(t, seed * 37.0)), simplex2d(vec2f(seed * 53.0, t))) * F.cellSize * 1.2;
  let xz = base + wander;
  let ground = terrainHeightM(xz, mtnTex, mtnSamp);

  // Swarms flash roughly together (shared phase per pocket) with per-fly jitter;
  // each flash is a short "J" that lifts the firefly a little while lit.
  let cycle = 3.2 + seed * 0.8;
  let pocket = floor(swarmN * 4.0);
  let phase = fract(G.time / cycle + pocket * 0.37 + seed * 0.15);
  let flash = smoothstep(0.0, 0.05, phase) * (1.0 - smoothstep(0.12, 0.3, phase));
  let y = ground + 0.7 + seed * 1.1 + sin(G.time * (0.5 + seed * 0.4) + seed * 9.0) * 0.2 + flash * 0.25;
  let center = vec3f(xz.x, y, xz.y);

  let dist = distance(center, G.camPos);
  let glow = (0.06 + 0.94 * flash) * G.night * (1.0 - smoothstep(55.0, 80.0, dist));
  if (glow <= 0.002) {
    return out;
  }
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let toCam = normalize(G.camPos - center);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), toCam));
  let up = cross(toCam, right);
  // Grow with distance a little so far fireflies stay a soft point instead of vanishing.
  let size = 0.06 + dist * 0.004;
  let world = center + (right * c.x + up * c.y) * size;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.uv = c;
  out.color = mix(vec3f(0.85, 1.0, 0.35), vec3f(1.0, 0.75, 0.3), fract(seed * 13.0)) * glow * 9.0;
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let a = exp(-r2 * 6.0) * (1.0 - smoothstep(0.8, 1.0, r2));
  return vec4f(frag.color * a, a);
}
