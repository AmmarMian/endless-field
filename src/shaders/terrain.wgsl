// Ground surface. A camera-centered grid whose vertex density falls off with distance,
// snapped to world steps; heights come from the shared terrain field.
import { SNOW, seasonGrass, seasonWeights } from "./lib/season.wgsl";
import { Globals } from "./lib/globals.wgsl";
import { SEA_Y, coastInfo, duneAmount, mountainCorridor, mountainHeight, riverInfo, riverInfoAt, riverValley, terrainBroad, terrainHeightR } from "./lib/terrain.wgsl";
import { sandEdge, sandColor, swashReach } from "./lib/shore.wgsl";
import { LifeCell, fieldColor, fieldKind, lifeIndex, lifeKey } from "./lib/field.wgsl";
import { SkyParams, applyFog, ambientSky, caustics, underwaterFog, wrapDiffuse } from "./lib/atmosphere.wgsl";
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
// Beach sand (Poly Haven coast_sand_01, CC0): colour and normal detail.
@group(0) @binding(10) var sandTex: texture_2d<f32>;
@group(0) @binding(11) var sandNor: texture_2d<f32>;

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
  // x = signed distance inland from the sea's edge, y = dunes amount, z = exposure to the open sea
  @location(7) coast: vec3f,
  // Direction inland (the gradient of the coast distance).
  @location(8) coastGrad: vec2f,
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
  let c = coastInfo(xz);
  out.coast = vec3f(c.x, duneAmount(xz), c.y);
  out.coastGrad = vec2f(0.0, -1.0);
  if (c.x < 300.0 && c.x > -100.0) {
    out.coastGrad = normalize(vec2f(coastInfo(xz + vec2f(2.0, 0.0)).x - c.x, coastInfo(xz + vec2f(0.0, 2.0)).x - c.x) + vec2f(1e-5));
  }
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let s = sky();
  let xz = frag.world.xz;
  let dist = length(G.camPos - frag.world);
  var n = normalize(frag.normal);
  // Sand detail, sampled in uniform control flow: two scales of the photo texture (the
  // broad one hides the fine one's tiling), and its normal map.
  let sandFine = textureSample(sandTex, samp, frag.world.xz * 0.42).rgb;
  let sandBroad = textureSample(sandTex, samp, frag.world.xz * 0.061 + vec2f(0.37, 0.71)).rgb;
  let sandN = (textureSample(sandNor, samp, frag.world.xz * 0.42).xyz * 2.0 - 1.0) * 0.6 + (textureSample(sandNor, samp, frag.world.xz * 0.13 + vec2f(0.5, 0.2)).xyz * 2.0 - 1.0) * 0.5;
  // Sand ripples, as slopes (they catch the low sun): fine wind ripples (~9 cm) across the
  // onshore breeze on the dry sand, and broad wave ripples (~60 cm) along the shore where the
  // swash works the beach. Each fades by its size on screen before it could shimmer.
  let windR = normalize(vec2f(0.25, -1.0) + vec2f(simplex2d(frag.world.xz * 0.05), simplex2d(frag.world.xz * 0.05 + 7.0)) * 0.35);
  let warp = simplex2d(frag.world.xz * 0.7) * 1.6 + simplex2d(frag.world.xz * 2.3) * 0.5;
  let phW = dot(frag.world.xz, windR) * (6.2831853 / 0.09) + warp * 3.0;
  let aaW = 1.0 - smoothstep(0.8, 2.2, fwidth(phW));
  let slopeW = (cos(phW) + 0.55 * cos(2.0 * phW + 1.0)) * 0.22 * aaW;
  let phS = frag.coast.x * (6.2831853 / 0.6) + warp * 1.2;
  let aaS = 1.0 - smoothstep(0.8, 2.2, fwidth(phS));
  let slopeS = cos(phS) * 0.14 * aaS;
  // Quartz grains glinting up close (each 4 mm cell its own facet, flashing as the eye moves).
  let gcell = floor(frag.world.xz * 250.0);
  let gh = fract(sin(dot(gcell, vec2f(12.9898, 78.233))) * 43758.5453);
  let glintAA = 1.0 - smoothstep(0.5, 1.5, length(fwidth(frag.world.xz * 250.0)));

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
  // From 25 m the ground takes the canopy look: the gaps between the coarser rings' blades
  // must show grass, not soil.
  let farMix = smoothstep(25.0, 80.0, dist);
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
    // Raked-looking gravel: fine grain, a few larger pebbles.
    let pebble = smoothstep(0.62, 0.7, simplex2d(xz * 11.0 + vec2f(2.0, 7.0)) * 0.5 + 0.5);
    let pathCol = mix(gravel, stone * 1.05, pebble * 0.6 + 0.0 * stoneN);
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
  // The shore: sand from under the sea up the beach and into the dunes (marram grass grows
  // there, see grass-cull), wet where the swash reaches, then the field takes over.
  let coastS = frag.coast.x;
  var sandAmt = 0.0;
  var wetSand = 0.0;
  if (coastS < 260.0) {
    let edge = sandEdge(xz, frag.coast.y);
    sandAmt = 1.0 - smoothstep(edge - 18.0, edge + 6.0, coastS);
    // On the open beach the height above the sea is known exactly from the shore profile
    // (the mesh's far, coarse triangles would move the wet line as the camera moves).
    let onBeach = coastS > -10.0 && coastS < 28.0 && frag.river.x > frag.river.z * 5.0;
    let above = select(frag.world.y - SEA_Y, coastS * 0.045, onBeach);
    let reach = swashReach(xz, G.time, frag.coast.z);
    // Freshly wet below the swash's reach; damp sand dries over a band above it.
    wetSand = max(1.0 - smoothstep(reach - 0.05, reach + 0.08, above), (1.0 - smoothstep(reach, reach + 0.3, above)) * 0.45) * sandAmt;
    var sandC = sandColor(xz, SEA_Y + above, coastS, wetSand, dist);
    // The photo's grain and pebbles, as relative variation around our tone.
    // (Brightness only: the photo's own hue is warmer than this beach.)
    let fineRel = dot(sandFine, vec3f(0.33)) / 0.172;
    let broadRel = dot(sandBroad, vec3f(0.33)) / 0.172;
    sandC = sandC * mix(1.0, fineRel, 0.3 * (1.0 - smoothstep(30.0, 90.0, dist))) * mix(1.0, broadRel, 0.15);
    albedo = mix(albedo, sandC, sandAmt);
    // Its relief: grains and footprints of wind; smoothed where the water has been.
    let bump = sandAmt * (1.0 - wetSand * 0.7) * (1.0 - smoothstep(10.0, 45.0, dist)) * 0.8;
    n = normalize(n + vec3f(sandN.x, 0.0, -sandN.y) * bump);
    // Wind ripples on the dry sand above the tide; wave ripples on the lower beach.
    let dryK = smoothstep(reach + 0.2, reach + 0.6, above) * sandAmt;
    let lowK = (1.0 - smoothstep(reach - 0.1, reach + 0.3, above)) * sandAmt;
    n = normalize(n - vec3f(windR.x, 0.0, windR.y) * slopeW * dryK - vec3f(frag.coastGrad.x, 0.0, frag.coastGrad.y) * slopeS * lowK);
  }
  // Winter: snow settles on the flatter ground (and on the path), patchy where wind scours it.
  let snowCover = sw.z * smoothstep(0.62, 0.9, n.y) * smoothstep(0.2, 0.55, simplex2d(xz * 0.05) * 0.35 + 0.65 + sw.z * 0.25 - 0.6);
  albedo = mix(albedo, SNOW * mix(0.92, 1.0, grain), clamp(snowCover * 1.4, 0.0, 1.0) * (1.0 - shore * 0.7));
  // (Bare sand is not shaded by a canopy.)
  let ao = mix(mix(0.45, 1.0, farMix), 1.0, sandAmt);

  let l = G.sunDir;
  // Far away the ground stands in for the grass canopy, which is lit like the blades are: its
  // normal leans up (blades stand up on slopes) and it never goes as dark as bare soil on a
  // slope turned from the sun.
  let grassyLight = farMix * (1.0 - smoothstep(0.15, 0.6, frag.misc.w)) * (1.0 - smoothstep(1.0, 20.0, frag.mtn)) * (1.0 - sandAmt);
  let nC = normalize(mix(n, vec3f(0.0, 1.0, 0.0), 0.55 * grassyLight));
  let diff = mix(wrapDiffuse(n, l, 0.4), wrapDiffuse(nC, l, 0.6), grassyLight);
  let shade = canopyLight(xz.x, frag.misc.w);
  // Rain: wet ground darkens (water fills the pores) and turns a little glossy. (Snow does
  // not wet: winter keeps its cover.)
  let wetness = G.wet * (1.0 - snowCover) * (1.0 - sw.z * 0.7);
  albedo = albedo * mix(1.0, 0.62, wetness * (1.0 - shore * 0.5));
  var col = albedo * (ambientSky(n, s) * 0.6 * ao * mix(0.5, 1.0, shade) + G.sunColor * diff * mix(mix(0.55, 1.0, farMix), 1.0, sandAmt) * shade * shade);
  // A wet sheen: wet ground mirrors a little of the sky at grazing angles.
  if (wetness > 0.01) {
    let vw = normalize(G.camPos - frag.world);
    let fres = 0.03 + 0.97 * pow(1.0 - max(dot(n, vw), 0.0), 5.0);
    col = mix(col, (G.horizonColor * 0.8 + G.zenithColor * 0.4), clamp(fres * 0.18 * wetness, 0.0, 0.3));
  }
  // Wet sand mirrors the sky (a glassy film right after a wave has drained back).
  if (wetSand > 0.01) {
    let vw = normalize(G.camPos - frag.world);
    let fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, vw), 0.0), 5.0);
    let r = reflect(-vw, vec3f(0.0, 1.0, 0.0));
    let sheen = (G.horizonColor * 0.9 + G.zenithColor * 0.3) * fres + G.sunColor * pow(max(dot(r, l), 0.0), 220.0) * 6.0;
    col = mix(col, sheen, clamp(wetSand * (0.25 + 0.6 * fres), 0.0, 0.85));
  }
  // Quartz glints on the sand, up close in the sun.
  if (sandAmt > 0.01) {
    let vg = normalize(G.camPos - frag.world);
    let facet = fract(gh * 7.13 + dot(vg, vec3f(3.1, 1.7, 2.3)) * 1.3 + dot(l, vec3f(1.9, 2.7, 0.7)));
    let glint = smoothstep(0.985, 1.0, facet) * glintAA * sandAmt * (1.0 - wetSand) * max(dot(n, l), 0.0);
    col += G.sunColor * glint * 1.8;
  }
  // Grass canopy seen from afar (where blades thin out or stop): the ground carries the
  // field's look. Weighted off on beds, forest floor, shores and fields of other crops.
  let v = normalize(G.camPos - frag.world);
  let grassy = farMix * (1.0 - smoothstep(0.15, 0.6, canopy)) * (1.0 - shore) * (1.0 - mud * 0.7) * (1.0 - sf) * (1.0 - pathAmt) * (1.0 - sandAmt);
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
  col = col + G.sunColor * albedo * back * 0.9 * farMix * mix(1.0, wave, grassy) * (1.0 - sandAmt);
  col = col + albedo * lampLight(frag.world);
  // Under the river: wet and dark, with caustics dancing on the bed (fading with depth).
  let inRiver = frag.river.x < frag.river.z * 1.8;
  let submerged = select(SEA_Y - frag.world.y, frag.river.y - frag.world.y, inRiver);
  if ((inRiver || coastS < 0.0) && submerged > 0.0) {
    let xz = frag.world.xz;
    let c1 = simplex2d(xz * 1.1 + vec2f(G.time * 0.55, G.time * 0.2));
    let c2 = simplex2d(xz * 1.7 - vec2f(G.time * 0.4, -G.time * 0.3));
    // (In the sea the swell's caustics are broader and softer, and fade out from afar.)
    let seaK = select(0.3 * (1.0 - smoothstep(15.0, 60.0, dist)), 1.0, inRiver);
    let caust = caustics(xz, G.time, c1, c2) * exp(-submerged * 0.5) * max(G.sunDir.y, 0.05) * seaK;
    col = col * mix(1.0, select(0.8, 0.55, inRiver), smoothstep(0.0, 0.4, submerged)) + albedo * G.sunColor * caust * 1.6;
  }
  // The swallow's light on the grass around it (a small warm pool that moves with it).
  if (G.playerGlow > 0.0) {
    let dp = frag.world - G.playerPos;
    col += albedo * vec3f(1.0, 0.82, 0.55) * G.playerGlow * 1.6 / (1.0 + dot(dp, dp) * 0.45);
  }
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  col = underwaterFog(col, frag.world, G.camPos, G.underwater, G.sunColor, G.zenithColor);
  return vec4f(col, 1.0);
}
