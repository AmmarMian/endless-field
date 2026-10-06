// Sky dome: gradient, sun, and soft painterly clouds on a high plane.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, skyColor, sunDisk } from "./lib/atmosphere.wgsl";
import { fbmSimplex2d } from "@vgpu/wgsl-std/noise/simplex";
import { pcg3d, unitFloat } from "@vgpu/wgsl-std/hash";

@group(0) @binding(0) var<uniform> G: Globals;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VOut;
  // Reversed-Z: depth 0 is the far plane.
  out.pos = vec4f(p[vi], 0.0, 1.0);
  out.ndc = p[vi];
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  let hp = G.invViewProj * vec4f(frag.ndc, 0.5, 1.0);
  let dir = normalize(hp.xyz / hp.w - G.camPos);

  // The moon reuses the sun disk but much dimmer than daylight.
  var col = skyColor(dir, s) + sunDisk(dir, s) * mix(1.0, 0.12, G.night);

  // Night: stars on a direction lattice (twinkling), a soft moon halo and a milky band.
  if (G.night > 0.01 && dir.y > -0.02) {
    let grid = dir * 220.0;
    let cell = floor(grid);
    let h = pcg3d(bitcast<vec3u>(vec3i(cell)));
    let rnd = unitFloat(h.x);
    if (rnd > 0.965) {
      let center = cell + vec3f(unitFloat(h.y), unitFloat(h.z), fract(rnd * 91.0));
      let d = length(grid - center);
      let tw = 0.65 + 0.35 * sin(G.time * (1.5 + rnd * 5.0) + rnd * 70.0);
      let temp = mix(vec3f(0.75, 0.85, 1.0), vec3f(1.0, 0.85, 0.65), fract(rnd * 37.0));
      col = col + temp * smoothstep(0.32, 0.0, d) * tw * 2.2 * G.night * smoothstep(-0.02, 0.15, dir.y);
    }
    let band = exp(-pow(dot(dir, normalize(vec3f(0.35, 0.25, -0.9))), 2.0) * 18.0);
    col = col + vec3f(0.06, 0.07, 0.12) * band * G.night * smoothstep(0.0, 0.3, dir.y);
    let mu = max(dot(dir, G.sunDir), 0.0);
    col = col + vec3f(0.35, 0.42, 0.6) * pow(mu, 120.0) * 0.25 * G.night;
  }

  if (dir.y > 0.0) {
    let t = 1100.0 / (dir.y + 0.04);
    let drift = G.windDir * G.time * 0.004;
    let uv = (dir.xz * t + G.camPos.xz * 0.6) * 0.00055 + drift;
    let n = fbmSimplex2d(uv, 5, 2.03, 0.5) * 0.5 + 0.5;
    let cover = smoothstep(0.50, 0.78, n) * smoothstep(0.0, 0.22, dir.y);
    // Density a step toward the sun gives a cheap self-shadowing gradient.
    let toward = fbmSimplex2d(uv + normalize(G.sunDir.xz + vec2f(1e-4)) * 0.035, 4, 2.03, 0.5) * 0.5 + 0.5;
    let shade = clamp(1.0 - (toward - n) * 3.5, 0.35, 1.25);
    let mu = max(dot(dir, G.sunDir), 0.0);
    let lit = G.sunColor * (0.55 * shade + 0.9 * pow(mu, 12.0)) + G.zenithColor * 0.35;
    let cloudCol = mix(G.horizonColor * 0.9, lit, 0.75);
    // At night clouds thin out so the stars come through.
    col = mix(col, cloudCol, cover * mix(0.85, 0.55, G.night));
  }
  return vec4f(col, 1.0);
}
