// Ground surface. A camera-centered grid whose vertex density falls off with distance,
// snapped to world steps; heights come from the shared terrain field.
import { Globals } from "./lib/globals.wgsl";
import { riverInfo, terrainHeight, terrainNormal } from "./lib/terrain.wgsl";
import { LifeCell, fieldColor, fieldKind, lifeIndex, lifeKey } from "./lib/field.wgsl";
import { SkyParams, applyFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";
import { biome } from "./lib/biome.wgsl";
import { bedColor, bedMask } from "./lib/beds.wgsl";

struct GridParams {
  center: vec2f,
  radius: f32,
  power: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> grid: GridParams;
@group(0) @binding(2) var<storage, read> life: array<LifeCell>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var pebbles: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
}

fn sky() -> SkyParams {
  return SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
}

fn sampleLife(xz: vec2f) -> f32 {
  // Bilinear so restored regions have soft edges on the ground too.
  let p = xz - 0.5;
  let c = vec2i(floor(p));
  let f = p - floor(p);
  var v = vec4f(0.0);
  let o = array<vec2i, 4>(vec2i(0, 0), vec2i(1, 0), vec2i(0, 1), vec2i(1, 1));
  for (var i = 0; i < 4; i = i + 1) {
    let cell = c + o[i];
    let lc = life[lifeIndex(cell)];
    v[i] = select(0.0, lc.life, lc.key == lifeKey(cell));
  }
  return mix(mix(v.x, v.y, f.x), mix(v.z, v.w, f.x), f.y);
}

@vertex
fn vs_main(@location(0) g: vec2f) -> VOut {
  let warped = sign(g) * pow(abs(g), vec2f(grid.power)) * grid.radius;
  let xz = grid.center + warped;
  let y = terrainHeight(xz);
  let world = vec3f(xz.x, y, xz.y);
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = terrainNormal(xz, 1.5);
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let s = sky();
  let xz = frag.world.xz;
  let dist = length(G.camPos - frag.world);
  let n = normalize(frag.normal);

  let meadow = simplex2d(xz * 0.018) * 0.5 + 0.5;
  let lifeV = sampleLife(xz);
  let grain = simplex2d(xz * 0.9) * 0.5 + 0.5;
  // Under nearby grass the ground reads as shaded soil; far away it stands in for the
  // whole field, so it takes the blades' mid-height color.
  let kindN = simplex2d(xz * 0.0055 + vec2f(113.0, -41.0) + vec2f(simplex2d(xz * 0.013), simplex2d(xz * 0.013 + 9.0)) * 0.9);
  let hueN = simplex2d(xz * 0.011 + vec2f(-7.0, 59.0) + vec2f(simplex2d(xz * 0.027 + 3.0), simplex2d(xz * 0.027 - 11.0)) * 0.8);
  let bio = biome(xz);
  let kind = fieldKind(xz, kindN + bio.y * 0.35 - bio.x * 0.3 - bio.z * 0.6, hueN + bio.x * 0.35 + bio.z * 0.5);
  let under = fieldColor(meadow, lifeV, 0.12, kind.z) * mix(0.7, 1.2, grain);
  // Tall meadows read lighter from afar (seed heads), lawns darker and greener.
  let field = fieldColor(meadow, lifeV, mix(0.62, 0.75, kind.x) - 0.15 * kind.y, kind.z) * mix(0.9, 1.08, grain);
  let farMix = smoothstep(60.0, 140.0, dist);
  var albedo = mix(under, field, farMix);
  // Beds: only a faint speckle of bloom color, mostly visible from afar.
  let bed = bedMask(xz);
  if (bed.x > 0.0) {
    let speck = smoothstep(0.3, 0.8, simplex2d(xz * 2.3));
    let bedAlbedo = mix(albedo, bedColor(u32(bed.y)) * 0.55, speck * smoothstep(25.0, 70.0, dist) * 0.6);
    albedo = mix(albedo, bedAlbedo, bed.x);
  }
  // Riverbank zones: wet pebbles at the waterline, drying up the beach, then a band of mud
  // and sparse grass into the field. The pebbles match the riverbed seen through the water.
  let river = riverInfo(xz);
  let hwR = river.z;
  let jitter = simplex2d(xz * 0.25) * hwR * 0.3;
  let shore = 1.0 - smoothstep(hwR * 1.15, hwR * 1.75, river.x + jitter);
  let mud = 1.0 - smoothstep(hwR * 1.5, hwR * 2.2, river.x + jitter);
  let peb = textureSample(pebbles, samp, xz * 0.45).rgb * 0.8;
  let above = frag.world.y - river.y;
  let wet = 1.0 - smoothstep(0.0, 0.45, above);
  let shoreAlbedo = mix(peb, peb * 0.5, wet);
  albedo = mix(albedo, vec3f(0.085, 0.065, 0.04) * mix(0.8, 1.15, grain), mud * (1.0 - farMix * 0.5));
  albedo = mix(albedo, shoreAlbedo, shore);
  let ao = mix(0.45, 1.0, farMix);

  let l = G.sunDir;
  let diff = wrapDiffuse(n, l, 0.4);
  var col = albedo * (ambientSky(n, s) * 0.6 * ao + G.sunColor * diff * mix(0.55, 1.0, farMix));
  // Blade-scale streaks so distant ground reads as grass rather than paint.
  let streak = simplex2d(xz * vec2f(1.7, 0.6) + G.windDir * G.time * 0.4) * 0.5 + 0.5;
  col = col * mix(1.0, mix(0.82, 1.12, streak), farMix);
  // Far field picks up the blades' backlit sheen.
  let v = normalize(G.camPos - frag.world);
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0);
  col = col + G.sunColor * albedo * back * 0.8 * farMix;
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s);
  return vec4f(col, 1.0);
}
