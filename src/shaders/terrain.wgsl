// Ground surface. A camera-centered grid whose vertex density falls off with distance,
// snapped to world steps; heights come from the shared terrain field.
import { SNOW, seasonGrass, seasonWeights } from "./lib/season.wgsl";
import { Globals } from "./lib/globals.wgsl";
import { mountainCorridor, mountainHeight, riverInfo, riverInfoAt, riverValley, terrainBroad, terrainHeightR } from "./lib/terrain.wgsl";
import { LifeCell, fieldColor, fieldKind, lifeIndex, lifeKey } from "./lib/field.wgsl";
import { SkyParams, applyFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";
import { biome, canopyLight } from "./lib/biome.wgsl";
import { bedColor, bedMask } from "./lib/beds.wgsl";
import { LANTERN_COLOR, PATH_WIDTH, lanternFirst, lanternTerm, pathDistance } from "./lib/path.wgsl";
import { SF_ROW, SF_ROW_PHASE, sunflowerField, sunflowerLocal } from "./lib/sunflowers.wgsl";

struct GridParams {
  center: vec2f,
  radius: f32,
  power: f32,
  cells: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> grid: GridParams;
@group(0) @binding(2) var<storage, read> life: array<LifeCell>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var pebbles: texture_2d<f32>;
@group(0) @binding(5) var mtnTex: texture_2d_array<f32>;
@group(0) @binding(6) var mtnSamp: sampler;
@group(0) @binding(7) var rockTex: texture_2d<f32>;
@group(0) @binding(8) var screeTex: texture_2d<f32>;
@group(0) @binding(9) var floorTex: texture_2d<f32>;

// Warm light from the lit path lanterns at night (see lib/path.wgsl).
fn lampLight(p: vec3f) -> vec3f {
  let k0 = lanternFirst(p, G.lampPos[1].w, G.lampPos[2].w, G.night);
  if (k0 < -50) {
    return vec3f(0.0);
  }
  let count = i32(G.lampPos[0].w);
  var sum = 0.0;
  for (var k = max(k0, 0); k < min(k0 + 5, count); k = k + 1) {
    sum = sum + lanternTerm(p, G.lampPos[k].xyz, G.lamps[u32(k) / 4u][u32(k) % 4u], f32(k), G.time);
  }
  return LANTERN_COLOR * sum * G.night;
}


struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) mtn: f32,
  // x = distance to the river, y = water height, z = half width
  @location(3) river: vec3f,
  // field kind: x = tall meadow, y = lawn, z = hue
  @location(4) kind: vec3f,
  // x = meadow patch, y = bed amount, z = snowline noise, w = forest canopy
  @location(5) misc: vec4f,
  @location(6) @interpolate(flat) bedSpecies: u32,
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

fn heightAt(xz: vec2f, r: vec4f, lod: f32) -> vec2f {
  let m = mountainHeight(xz, mtnTex, mtnSamp, lod) * mountainCorridor(r);
  let base = terrainHeightR(xz, r);
  let hgt = riverValley(base + m, r, xz);
  // Altitude above the surrounding plain drives the alpine zones.
  return vec2f(hgt, max(hgt - (terrainBroad(xz) * 0.6 + 22.0), 0.0) * smoothstep(0.0, 30.0, max(hgt - base, m)));
}

@vertex
fn vs_main(@location(0) g: vec2f) -> VOut {
  let warped = sign(g) * pow(abs(g), vec2f(grid.power)) * grid.radius;
  // Snap each vertex onto a fixed world lattice sized to its local spacing (power of two), so
  // distant vertices always sample the same places as the camera moves: no shimmering ridges.
  let ds = grid.radius * grid.power * pow(max(abs(g), vec2f(1e-4)), vec2f(grid.power - 1.0)) * (2.0 / grid.cells);
  let lattice = exp2(ceil(log2(max(max(ds.x, ds.y), 1.0))));
  let xz = round((grid.center + warped) / lattice) * lattice;
  // Pre-filtered mountain heights matching the vertex spacing (texel = 2400 / 512 m).
  let lod = max(log2(lattice / 4.69), 0.0);
  let r = riverInfo(xz);
  let hm = heightAt(xz, r, lod);
  let world = vec3f(xz.x, hm.x, xz.y);
  let e = max(lattice, 1.5);
  let ax = xz + vec2f(e, 0.0);
  let az = xz + vec2f(0.0, e);
  let hx = heightAt(ax, riverInfoAt(ax, r.w), lod).x - hm.x;
  let hz = heightAt(az, riverInfoAt(az, r.w), lod).x - hm.x;

  // Low-frequency fields, evaluated per vertex rather than per pixel.
  let kindN = simplex2d(xz * 0.0055 + vec2f(113.0, -41.0) + vec2f(simplex2d(xz * 0.013), simplex2d(xz * 0.013 + 9.0)) * 0.9);
  let hueN = simplex2d(xz * 0.011 + vec2f(-7.0, 59.0) + vec2f(simplex2d(xz * 0.027 + 3.0), simplex2d(xz * 0.027 - 11.0)) * 0.8);
  let bio = biome(xz);
  let bed = bedMask(xz);

  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = normalize(vec3f(-hx, e, -hz));
  out.mtn = hm.y;
  out.river = r.xyz;
  out.kind = fieldKind(xz, kindN + bio.y * 0.35 - bio.x * 0.3 - bio.z * 0.6, hueN + bio.x * 0.35 + bio.z * 0.5);
  out.misc = vec4f(simplex2d(xz * 0.018) * 0.5 + 0.5, bed.x, simplex2d(xz * 0.004), bio.z);
  out.bedSpecies = u32(bed.y + 0.5);
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let s = sky();
  let xz = frag.world.xz;
  let dist = length(G.camPos - frag.world);
  let n = normalize(frag.normal);

  let meadow = frag.misc.x;
  let lifeV = sampleLife(xz);
  let grain = simplex2d(xz * 0.9) * 0.5 + 0.5;
  // Under nearby grass the ground reads as shaded soil; far away it stands in for the
  // whole field, so it takes the blades' mid-height color.
  let kind = frag.kind;
  let sw = seasonWeights(G.season);
  let under = seasonGrass(fieldColor(meadow, lifeV, 0.12, kind.z), sw, 0.12) * mix(0.7, 1.2, grain);
  // Tall meadows read lighter from afar (seed heads), lawns darker and greener.
  let tField = mix(0.62, 0.75, kind.x) - 0.15 * kind.y;
  let field = seasonGrass(fieldColor(meadow, lifeV, tField, kind.z), sw, tField) * mix(0.9, 1.08, grain);
  let farMix = smoothstep(60.0, 140.0, dist);
  // From afar a field is the average of lit tips and shadowed stems: darker and richer than
  // the tip color, matching the blades of the coarse rings so they merge into it.
  let fieldLum = dot(field, vec3f(0.3, 0.59, 0.11));
  let fieldAgg = mix(vec3f(fieldLum), field, 1.18) * 0.8;
  var albedo = mix(under, fieldAgg, farMix);
  // Beds: only a faint speckle of bloom color, mostly visible from afar.
  let bedAmt = frag.misc.y;
  if (bedAmt > 0.01) {
    let speck = smoothstep(0.3, 0.8, simplex2d(xz * 2.3));
    let bedAlbedo = mix(albedo, bedColor(frag.bedSpecies) * 0.55, speck * smoothstep(25.0, 70.0, dist) * 0.6);
    albedo = mix(albedo, bedAlbedo, bedAmt);
  }
  // Sunflower field: tilled soil ridged along the drill rows, crumbly clods, a few weeds;
  // from afar it reads as the field's green-gold canopy.
  let sf = sunflowerField(xz);
  if (sf > 0.01) {
    let l = sunflowerLocal(xz);
    let ridge = 0.5 + 0.5 * cos((l.y + SF_ROW_PHASE) / SF_ROW * 6.2831853);
    let clod = simplex2d(xz * 3.1) * 0.5 + 0.5;
    var soil = mix(vec3f(0.1, 0.07, 0.045), vec3f(0.19, 0.14, 0.09), ridge * 0.6 + clod * 0.4) * mix(0.85, 1.1, grain);
    soil = mix(soil, under * 0.8, smoothstep(0.55, 0.9, simplex2d(xz * 0.6) * 0.5 + 0.5) * 0.45);
    albedo = mix(albedo, mix(soil, vec3f(0.17, 0.2, 0.05), farMix), sf);
  }
  // Lantern path: packed gravel with flat stepping stones, its edge frayed into the grass.
  let pd = pathDistance(xz);
  var pathAmt = 0.0;
  if (pd < PATH_WIDTH + 1.5) {
    pathAmt = 1.0 - smoothstep(PATH_WIDTH - 0.35, PATH_WIDTH + 0.5 + simplex2d(xz * 1.7) * 0.35, pd);
    let fine = simplex2d(xz * 9.0) * 0.5 + 0.5;
    let coarse = simplex2d(xz * 2.6 + vec2f(-3.0, 5.0)) * 0.5 + 0.5;
    let gravel = vec3f(0.29, 0.25, 0.19) * mix(0.7, 1.15, fine) * mix(0.85, 1.08, coarse);
    let stoneN = simplex2d(xz * 1.4 + vec2f(4.0, -2.0));
    let stone = vec3f(0.36, 0.34, 0.31) * mix(0.85, 1.1, simplex2d(xz * 3.3) * 0.5 + 0.5);
    let pathCol = mix(gravel, stone, smoothstep(0.55, 0.6, stoneN) * (1.0 - smoothstep(0.6, 1.0, pd / PATH_WIDTH)));
    albedo = mix(albedo, pathCol, pathAmt * (1.0 - farMix * 0.3));
  }
  // Alpine zones on the mountains: short turf, then scree and rock on steep ground, and snow
  // on the high, flatter faces. Cliff rock is sampled triplanar-ish (xz for flats, side
  // projections for walls) so steep faces do not smear.
  let mtn = frag.mtn;
  // Sampled in uniform control flow (WGSL rule), used only on mountain ground.
  let wn = pow(abs(n), vec3f(4.0));
  let wsum = wn.x + wn.y + wn.z;
  let rockRaw = (textureSample(rockTex, samp, frag.world.zy * 0.035).rgb * wn.x + textureSample(rockTex, samp, xz * 0.035).rgb * wn.y + textureSample(rockTex, samp, frag.world.xy * 0.035).rgb * wn.z) / wsum;
  // Granite-grey with large-scale tone variation so cliffs never tile visibly.
  let tone = simplex2d(xz * 0.012 + frag.world.y * 0.01) * 0.5 + 0.5;
  let rockC = mix(vec3f(dot(rockRaw, vec3f(0.33))), rockRaw, 0.55) * mix(1.25, 1.75, tone) * vec3f(1.0, 0.99, 0.96);
  let scree = textureSample(screeTex, samp, xz * 0.08).rgb * mix(0.95, 1.25, tone);
  let litter = textureSample(floorTex, samp, xz * 0.18).rgb;
  if (mtn > 1.0) {
    let alpine = smoothstep(80.0, 150.0, mtn);
    let steep = 1.0 - smoothstep(0.62, 0.82, n.y);
    let turf = mix(albedo, vec3f(0.11, 0.13, 0.06) * mix(0.85, 1.1, grain), alpine * 0.7);
    var ground = mix(turf, scree * 0.85, smoothstep(0.35, 0.7, steep) * smoothstep(40.0, 120.0, mtn));
    ground = mix(ground, rockC * 0.9, smoothstep(0.55, 0.85, steep + (1.0 - smoothstep(0.0, 1.0, alpine)) * -0.2));
    let snowLine = 205.0 + frag.misc.z * 35.0;
    let snow = smoothstep(snowLine - 20.0, snowLine + 25.0, mtn) * smoothstep(0.45, 0.75, n.y);
    ground = mix(ground, vec3f(0.86, 0.9, 0.97), snow);
    albedo = mix(albedo, ground, smoothstep(1.0, 20.0, mtn));
  }

  // Forest floor: leaf litter and needles under the canopy, mottled with moss.
  let canopy = frag.misc.w;
  if (canopy > 0.01) {
    let moss = smoothstep(0.2, 0.8, simplex2d(xz * 0.15) * 0.5 + 0.5);
    let floorC = mix(litter * vec3f(0.85, 0.8, 0.72), vec3f(0.09, 0.13, 0.04), moss * 0.45);
    albedo = mix(albedo, floorC, smoothstep(0.15, 0.7, canopy) * (1.0 - farMix * 0.35));
  }

  // Riverbank zones: wet pebbles at the waterline, drying up the beach, then a band of mud
  // and sparse grass into the field. The pebbles match the riverbed seen through the water.
  let river = frag.river;
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
  // Winter: snow settles on the flatter ground (and on the path), patchy where wind scours it.
  let snowCover = sw.z * smoothstep(0.62, 0.9, n.y) * smoothstep(0.2, 0.55, simplex2d(xz * 0.05) * 0.35 + 0.65 + sw.z * 0.25 - 0.6);
  albedo = mix(albedo, SNOW * mix(0.92, 1.0, grain), clamp(snowCover * 1.4, 0.0, 1.0) * (1.0 - shore * 0.7));
  let ao = mix(0.45, 1.0, farMix);

  let l = G.sunDir;
  let diff = wrapDiffuse(n, l, 0.4);
  let shade = canopyLight(xz.x, frag.misc.w);
  var col = albedo * (ambientSky(n, s) * 0.6 * ao * mix(0.5, 1.0, shade) + G.sunColor * diff * mix(0.55, 1.0, farMix) * shade * shade);
  // Grass canopy seen from afar (where blades thin out or stop): the ground carries the
  // field's look. Weighted off on beds, forest floor, shores and fields of other crops.
  let v = normalize(G.camPos - frag.world);
  let grassy = farMix * (1.0 - smoothstep(0.15, 0.6, canopy)) * (1.0 - shore) * (1.0 - mud * 0.7) * (1.0 - sf) * (1.0 - pathAmt);
  // Clumps at three scales, each fading out once it is smaller than a pixel.
  let fp = length(fwidth(xz));
  let c0 = simplex2d(xz * 1.6 + vec2f(3.0, 1.0)) * (1.0 - smoothstep(0.15, 0.6, fp));
  let c1 = simplex2d(xz * 0.42 + vec2f(-7.0, 2.0)) * (1.0 - smoothstep(0.6, 2.4, fp));
  let c2 = simplex2d(xz * 0.11 + vec2f(1.0, 9.0)) * (1.0 - smoothstep(2.4, 9.0, fp));
  let clumps = 1.0 + (c0 * 0.12 + c1 * 0.1 + c2 * 0.08);
  // Wind waves: the gust field that bends the blades also bends the far grass; bent blades
  // show their lighter flanks, so gusts sweep across distant hills as bright moving bands.
  let wdir = G.windDir;
  let along = dot(xz, wdir);
  let ripple = simplex2d(vec2f(along * 0.085 - G.time * 0.68, dot(xz, vec2f(-wdir.y, wdir.x)) * 0.03));
  let broad = simplex2d(xz * 0.014 - wdir * G.time * 0.09);
  let gust = smoothstep(-0.6, 0.9, broad * 0.6 + ripple * 0.55);
  let lookDownwind = abs(dot(normalize(v.xz + vec2f(1e-4)), wdir));
  let wave = 1.0 + (gust - 0.45) * mix(0.28, 0.45, lookDownwind) * G.windStrength;
  // Canopy depth: at grazing angles only the lit upper blades show; looking down, the dark
  // gaps between stems do.
  let ndv = clamp(dot(n, v), 0.0, 1.0);
  let canopyView = mix(1.08, 0.82, smoothstep(0.25, 0.9, ndv));
  col = col * mix(1.0, clumps * wave * canopyView, grassy);
  // Backlit sheen: blades glow when the sun is behind them, strongest at grazing angles.
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0) * mix(1.0, 0.5, ndv);
  col = col + G.sunColor * albedo * back * 0.9 * farMix * mix(1.0, wave, grassy);
  col = col + albedo * lampLight(frag.world);
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  return vec4f(col, 1.0);
}
