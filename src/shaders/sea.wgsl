// The sea. A camera-centered grid (dense near, sparse to the horizon, every vertex snapped to
// a world lattice so nothing swims) lies at sea level wherever coastInfo() says there is sea.
// Offshore it rides the FFT swell (ocean-*.wgsl); toward the beach the swell gives way to surf
// (lib/shore.wgsl): breakers that steepen and spill, foaming bores, and the swash running up
// the sand and draining back. Blended like the river: out = rgb + scene * a, a being what the
// water column lets through, so the sandy bed shows in the shallows.
import { Globals } from "./lib/globals.wgsl";
import { SEA_Y, coastInfo, terrainHeight } from "./lib/terrain.wgsl";
import { surf } from "./lib/shore.wgsl";
import { swell, swellGrad } from "./lib/swell.wgsl";
import { SkyParams, applyFog, skyColor, underwaterFog } from "./lib/atmosphere.wgsl";
import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";

struct SeaParams {
  center: vec2f,
  radius: f32,
  power: f32,
  cells: f32,
  patchSize: f32,
  // Night glow of breaking water (bioluminescence), 0..1.
  glow: f32,
  pad: f32,
  // Whales at the surface: x, z, heading, wash (0 none .. 1 lying in the water); and their
  // footprints: x, z of the dive, its strength (fading), and swimming (1 cruising: a wake).
  whales: array<vec4f, 4>,
  prints: array<vec4f, 4>,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> S: SeaParams;
@group(0) @binding(2) var disp: texture_2d<f32>;
@group(0) @binding(3) var dispSamp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  // The bed's height, and the still-water depth over it.
  @location(1) ground: f32,
  @location(2) still: f32,
  // x = signed distance inland, y = exposure to the open sea, z = how much swell rides here
  @location(3) coast: vec3f,
  // The swell's height here (the surf is evaluated per pixel: a crisp waterline).
  @location(4) swellY: f32,
  // The surf's slope (d height / d xz).
  @location(5) surfGrad: vec2f,
  // Position on the FFT patch (undisplaced), for per-pixel normals.
  @location(6) uv: vec2f,
  // How much of the long swell rides here.
  @location(7) swellAmt: f32,
}

fn sky() -> SkyParams {
  return SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
}

fn swellAt(uv: vec2f) -> vec4f {
  return textureSampleLevel(disp, dispSamp, uv, 0.0);
}

@vertex
fn vs_main(@location(0) g: vec2f) -> VOut {
  let warped = sign(g) * pow(abs(g), vec2f(S.power)) * S.radius;
  let ds = S.radius * S.power * pow(max(abs(g), vec2f(1e-4)), vec2f(S.power - 1.0)) * (2.0 / S.cells);
  let lattice = exp2(ceil(log2(max(max(ds.x, ds.y), 0.5))));
  let xz = round((S.center + warped) / lattice) * lattice;
  var out: VOut;
  let c = coastInfo(xz);
  out.uv = xz / S.patchSize;
  if (c.x > 80.0) {
    // Inland: no water (the triangles reaching here are cut at the waterline by `depth`).
    out.world = vec3f(xz.x, SEA_Y, xz.y);
    out.pos = G.viewProj * vec4f(out.world, 1.0);
    out.ground = SEA_Y + 4.0;
    out.coast = vec3f(c, 0.0);
    return out;
  }
  // The bed matters only near the shore; far out the sea is deep.
  var ground = SEA_Y - 30.0;
  if (c.x > -220.0) {
    ground = terrainHeight(xz);
  }
  let still = SEA_Y - ground;
  let sw = surf(c.x, still, xz, G.time, c.y);
  // The swell dies in the shallows (it has become surf) and up the sheltered estuary, and
  // eases far out where its tiling would show.
  let dist = distance(G.camPos.xz, xz);
  // The wind raises the chop: stronger gusts, steeper and taller little waves.
  let chop = 0.6 + 0.9 * G.windStrength;
  let swellK = smoothstep(0.15, 1.6, still) * mix(0.2, 1.0, c.y) * mix(1.0, 0.4, smoothstep(600.0, 2500.0, dist)) * chop;
  // The long swell rolling in (lib/swell.wgsl), dying into surf in the shallows.
  let swAmt = smoothstep(0.5, 2.8, still) * mix(0.15, 1.0, c.y);
  let d = swellAt(out.uv).xyz * swellK + swell(xz, G.time, swAmt);
  // Slope of the surf, by differences across the shore (only where there is surf).
  var grad = vec2f(0.0);
  if (c.x > -160.0 && c.y > 0.01) {
    let e = max(lattice, 0.5);
    let cx = coastInfo(xz + vec2f(e, 0.0)).x;
    let cz = coastInfo(xz + vec2f(0.0, e)).x;
    let hx = surf(cx, still - (cx - c.x) * 0.05, xz + vec2f(e, 0.0), G.time, c.y).x;
    let hz = surf(cz, still - (cz - c.x) * 0.05, xz + vec2f(0.0, e), G.time, c.y).x;
    grad = vec2f(hx - sw.x, hz - sw.x) / e;
  }
  let y = SEA_Y + sw.x + d.y;
  out.world = vec3f(xz.x + d.x, y, xz.y + d.z);
  // At the beach the triangles never dip under the sand: they are lifted to just above the
  // ground, so neither the ground's mesh nor this grid's density (which changes as the camera
  // moves) decides where the water ends. The per-pixel waterline in fs_main does.
  var drawY = y;
  var drawP = vec3f(out.world.x, y, out.world.z);
  let camD = distance(G.camPos, drawP);
  if (c.x > -60.0) {
    // (Near the eye the ground mesh is fine and exact: barely lifted, so small things on the
    // sand, crabs and shells, are not hidden under a raised film.)
    drawY = max(y, ground + 0.004 + min(camD * 0.0025, 0.14));
    drawP.y = drawY;
    // And pulled toward the eye by more than the ground mesh's error at that distance (its
    // coarse far triangles change as the camera moves): the ground never pokes through it.
    // The water only shows where the per-pixel waterline says there is water.
    drawP += normalize(G.camPos - drawP) * camD * 0.005 * smoothstep(3.0, 15.0, camD) * (1.0 - smoothstep(-60.0, -30.0, -c.x));
  }
  out.pos = G.viewProj * vec4f(drawP, 1.0);
  out.swellAmt = swAmt;
  out.ground = ground;
  out.still = still;
  out.coast = vec3f(c, swellK);
  out.swellY = d.y;
  out.surfGrad = grad;
  return out;
}

// Foam texture: clustered bubbles and lacy holes, drifting with the water.
fn foamTex(p: vec2f, t: f32, aa: f32) -> f32 {
  let a = simplex2d(p * 0.35 + vec2f(t * 0.05, -t * 0.03)) * 0.6;
  let b = simplex2d(p * 1.3 - vec2f(t * 0.11, 0.0)) * 0.3 * aa;
  let c = simplex2d(p * 4.1 + vec2f(0.0, t * 0.2)) * 0.15 * aa;
  return a + b + c;
}

fn hash2(c: vec2f) -> vec2f {
  let q = vec2f(dot(c, vec2f(127.1, 311.7)), dot(c, vec2f(269.5, 183.3)));
  return fract(sin(q) * 43758.5453);
}

// Distance to the nearest cell border of a jittered grid (F2 - F1): foam's bubble network.
fn cells(p: vec2f, t: f32) -> f32 {
  let i = floor(p);
  let f = p - i;
  var d1 = 9.0;
  var d2 = 9.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      let o = vec2f(f32(x), f32(y));
      let h = hash2(i + o);
      let c = o + 0.5 + 0.38 * sin(h * 6.2831 + t * (0.3 + h.yx * 0.4)) - f;
      let d = dot(c, c);
      if (d < d1) {
        d2 = d1;
        d1 = d;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  return sqrt(d2) - sqrt(d1);
}

// Foam coverage for an amount `a` (0..1): thick white water when fresh, opening into lace as
// it thins (only the borders between bubbles stay white). Averages out with distance.
fn foamLace(p: vec2f, a: f32, t: f32, aa: f32, ft: f32) -> f32 {
  if (a <= 0.003) {
    return 0.0;
  }
  let e = cells(p * vec2f(0.7, 1.1), t) * 0.6 + cells(p * 2.3 + vec2f(5.2, 1.3), t * 1.3) * 0.4;
  let w = clamp(a * 0.7 + ft * 0.35, 0.0, 1.0) * 0.5;
  let lace = 1.0 - smoothstep(w - 0.05, w + 0.03, e);
  // Thin foam is a translucent film, thick foam opaque white.
  return mix(clamp(a * 0.75, 0.0, 1.0), lace * mix(0.55, 1.0, smoothstep(0.3, 0.8, a)), aa);
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = sky();
  let p = frag.world;
  let t = G.time;
  // Derivatives first (uniform control flow), discards after.
  let footprint = length(fwidth(p.xz));
  let aa = 1.0 - smoothstep(0.15, 1.2, footprint);
  let uvFoot = length(fwidth(frag.uv)) * 256.0;
  let swellK = frag.coast.z;
  let open = frag.coast.y;
  let xz0 = frag.uv * S.patchSize;
  var sw = surf(frag.coast.x, frag.still, xz0, t, open);
  var depth = SEA_Y + sw.x + frag.swellY - frag.ground;
  // Near the waterline, the true ground and the surf over it (the vertex grid is too coarse
  // for a clean edge, and its spacing changes as the camera moves).
  if (abs(depth) < 1.5 && frag.coast.x > -80.0 && frag.coast.x < 80.0) {
    let g = terrainHeight(xz0);
    sw = surf(frag.coast.x, SEA_Y - g, xz0, t, open);
    depth = SEA_Y + sw.x + frag.swellY - g;
  }
  if (depth < 0.0 || frag.coast.x > 80.0) {
    discard;
  }
  let camDist = distance(G.camPos, p);

  // Per-pixel normal from the swell texture (its own cells), plus a finer, rotated sample of
  // the same field for capillary detail; both fade by their footprint before they alias.
  let e = 1.0 / 256.0;
  let cell = S.patchSize / 256.0;
  let c0 = swellAt(frag.uv);
  let cx = swellAt(frag.uv + vec2f(e, 0.0));
  let cz = swellAt(frag.uv + vec2f(0.0, e));
  let tX = vec3f(cell + cx.x - c0.x, cx.y - c0.y, cx.z - c0.z);
  let tZ = vec3f(cz.x - c0.x, cz.y - c0.y, cell + cz.z - c0.z);
  var nSwell = normalize(cross(tZ, tX));
  let swellAA = 1.0 / (1.0 + uvFoot * uvFoot * 0.8);
  let fineUv = mat2x2f(0.8, -0.6, 0.6, 0.8) * frag.uv * 4.7 + vec2f(0.31, 0.77);
  let f0 = swellAt(fineUv).y;
  let fx = swellAt(fineUv + vec2f(e, 0.0)).y - f0;
  let fz = swellAt(fineUv + vec2f(0.0, e)).y - f0;
  let fineAA = 1.0 / (1.0 + uvFoot * uvFoot * 20.0);
  let fineGrad = vec2f(fx, fz) / (cell / 4.7) * 0.35 * fineAA;
  // Ripples ride everywhere, even on the swash; the swell only where it lives.
  // Gusts sweep downwind over the water as dark, rougher patches (cat's paws).
  let wd = G.windDir;
  let gustN = simplex2d(xz0 * 0.006 - wd * t * 0.045) * 0.65 + simplex2d(xz0 * 0.021 - wd * t * 0.09 + vec2f(3.0, 7.0)) * 0.35;
  let gust = smoothstep(-0.25, 0.55, gustN) * G.windStrength;
  let chopGrad = vec2f(-nSwell.x, -nSwell.z) / max(nSwell.y, 0.2) * swellK * swellAA;
  var grad = chopGrad * mix(0.75, 1.35, gust) + fineGrad * mix(0.5, 1.9, gust) + frag.surfGrad * aa + swellGrad(xz0, t, frag.swellAmt);
  // A wingtip cutting the water leaves a thin furrow and a spreading V (as on the river).
  var wakeFoam = 0.0;
  if (G.dip > 0.01 && camDist < 90.0) {
    var along = 0.0;
    for (var i = 0u; i < 23u; i = i + 1u) {
      let a = G.trail[i];
      let b = G.trail[i + 1u];
      if (b.w <= 0.0 && i > 0u) {
        break;
      }
      let ab = b.xz - a.xz;
      let len = length(ab);
      if (len < 1e-3) {
        continue;
      }
      let dir = ab / len;
      let tseg = clamp(dot(p.xz - a.xz, dir), 0.0, len);
      let off = p.xz - (a.xz + dir * tseg);
      let sd = length(off);
      let behind = along + tseg;
      let k = G.dip * exp(-behind * 0.1);
      wakeFoam = max(wakeFoam, exp(-sd * sd * 120.0) * exp(-behind * 0.3) * k);
      let ridge = sd - behind * 0.34;
      grad = grad + normalize(off + vec2f(1e-4, 0.0)) * exp(-ridge * ridge * 24.0) * k * smoothstep(0.0, 0.8, behind) * 0.45 * cos(ridge * 22.0);
      along = along + len;
      if (along > 20.0) {
        break;
      }
    }
  }
  // Whales: water washing round a body lying awash, the V of a cruising whale's wake, and the
  // glassy-smooth footprint a diving whale's tail leaves on the surface.
  var whaleFoam = 0.0;
  var calm = 0.0;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let W = S.whales[i];
    let P = S.prints[i];
    if (P.z > 0.0) {
      calm = max(calm, (1.0 - smoothstep(5.0, 11.0 + 2.0 * simplex2d(xz0 * 0.2), distance(xz0, P.xy))) * P.z);
    }
    if (W.w <= 0.0) {
      continue;
    }
    let d = xz0 - W.xy;
    if (dot(d, d) > 1600.0) {
      continue;
    }
    let fwd = vec2f(sin(W.z), cos(W.z));
    let along = dot(d, fwd);
    let across = dot(d, vec2f(fwd.y, -fwd.x));
    let a = clamp(along, -6.0, 3.6);
    let r = 1.35 * sqrt(max(0.0, 1.0 - pow((a + 1.0) / 5.4, 2.0)));
    let sd = length(vec2f(along - a, across)) - r;
    whaleFoam = max(whaleFoam, W.w * exp(-max(sd, 0.0) / 0.45) * (0.6 + 0.4 * simplex2d(xz0 * 1.3 + vec2f(t * 0.7, 0.0))));
    if (P.w > 0.0) {
      // Wake: a V spreading back from the tail, with a turbulent band along the track.
      let back = -along - 5.5;
      if (back > 0.0) {
        let arm = abs(abs(across) - back * 0.36);
        whaleFoam = max(whaleFoam, P.w * exp(-arm * arm * 3.0) * exp(-back * 0.06) * 0.7);
        whaleFoam = max(whaleFoam, P.w * exp(-across * across * 0.6) * exp(-back * 0.1) * 0.5);
        grad += fwd * exp(-arm * arm * 3.0) * exp(-back * 0.06) * 0.3 * P.w * cos(arm * 8.0);
      }
    }
  }
  grad = grad * (1.0 - 0.85 * calm);
  let n = normalize(vec3f(-grad.x, 1.0, -grad.y));
  let v = normalize(G.camPos - p);
  if (G.camPos.y < p.y) {
    return underside(p, n, v, s);
  }
  let ndv = max(dot(n, v), 0.0);
  var fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);

  // Reflection: the sky, and the sun's glitter path (its sharpness falling with the footprint
  // so it never sparkles into single pixels).
  let r = reflect(-v, n);
  var refl = skyColor(normalize(vec3f(r.x, max(r.y, 0.01), r.z)), s);
  let glintExp = 1400.0 / (1.0 + footprint * 60.0);
  let sunSpec = pow(max(dot(r, G.sunDir), 0.0), glintExp) * 50.0 * glintExp / 1400.0 + pow(max(dot(r, G.sunDir), 0.0), 90.0) * 0.4;
  refl = refl + G.sunColor * sunSpec;

  // The water column along the view ray: clear and turquoise over the sand, deep blue-green
  // offshore. The light it scatters back brightens in the crests where the sun shines
  // through them.
  let path = depth / max(v.y, 0.1);
  let absorb = exp(-path * vec3f(0.55, 0.12, 0.1));
  let deep = smoothstep(1.0, 9.0, depth);
  let body = mix(vec3f(0.03, 0.16, 0.14), vec3f(0.006, 0.035, 0.055), deep);
  var scatter = body * (G.sunColor * 0.55 + s.zenith * 0.7);
  // Light through the crests: wave tops (swell and chop) glow green-turquoise, troughs stay
  // dark, most when looking toward the sun.
  let crest = clamp((p.y - SEA_Y) * 0.6 + 0.45, 0.0, 1.0);
  scatter = scatter * mix(0.7, 1.25, crest);
  scatter += vec3f(0.04, 0.24, 0.2) * G.sunColor * (0.25 + pow(max(dot(v, -G.sunDir) * 0.5 + 0.5, 0.0), 3.0)) * crest * crest * 0.45;
  var through = dot(absorb, vec3f(0.25, 0.45, 0.3));
  let tint = scatter * (vec3f(1.0) - absorb);
  fresnel = fresnel * smoothstep(0.0, 0.25, depth);
  var col = tint * (1.0 - fresnel) + refl * fresnel;
  through *= 1.0 - fresnel;

  // Foam. Whitecaps where the swell folds over (Jacobian), the spilling crest of a breaker,
  // the bore's churned water fading behind it, and lace at the swash's edge.
  let ft = foamTex(p.xz, t, aa);
  let jac = mix(1.0, c0.w, swellK);
  let whitecap = smoothstep(0.78 + 0.1 * gust, 0.5, jac) * mix(0.3, 0.7, gust);
  // Wind streaks: lines of old foam drawn out downwind from broken crests.
  let along = dot(xz0, wd);
  let across = dot(xz0, vec2f(-wd.y, wd.x));
  let streakN = simplex2d(vec2f(along * 0.02 - t * 0.05, across * 0.35)) * 0.6 + simplex2d(vec2f(along * 0.05, across * 0.9) + vec2f(4.0, 1.0)) * 0.4;
  let streaks = smoothstep(0.45, 0.8, streakN) * gust * 0.35 * swellK * aa;
  let u = sw.z;
  // The spilling crest: a bright band riding the wave's face just as it passes.
  let crestBand = max(smoothstep(0.93, 1.0, u), 1.0 - smoothstep(0.0, 0.05, u));
  let spill = sw.y * crestBand;
  let edge = (1.0 - smoothstep(0.0, 0.05 + 0.05 * open, depth)) * open;
  // Old foam lingers as a faint lacy sheet over the surf zone.
  let sheet = (1.0 - smoothstep(10.0, 70.0, -frag.coast.x)) * open * smoothstep(0.1, 0.6, ft) * 0.14;
  let amount = clamp((whitecap * 0.85 + streaks) * (1.0 - calm) + spill + sw.w * 0.9 + edge * 0.8 + sheet + whaleFoam, 0.0, 1.0);
  let foamK = clamp(foamLace(frag.uv * S.patchSize, amount, t, aa, ft) + wakeFoam * 1.6, 0.0, 1.0);
  let foamCol = (G.sunColor * 0.8 + s.zenith * 0.6) * mix(0.85, 1.0, ft * 0.5 + 0.5);
  col = mix(col, foamCol, foamK);
  through *= 1.0 - foamK;
  // Night: the breaking water glows blue-green (bioluminescent plankton stirred up).
  if (S.glow > 0.0) {
    let stir = max(spill * 1.2, max(sw.w * 0.8, max(edge, wakeFoam * 2.0))) * mix(0.4, 1.0, foamK);
    let spark = smoothstep(0.2, 0.9, simplex2d(p.xz * 2.2 + vec2f(t * 0.4, -t * 0.3)) * 0.5 + 0.5);
    col += vec3f(0.1, 0.65, 0.95) * stir * mix(0.5, 1.0, spark) * S.glow * 1.8;
  }

  let mist = vec4f(G.mist, G.mistBase, G.canopy, G.time);
  let fog0 = applyFog(vec3f(0.0), p, G.camPos, G.fogDensity, s, mist);
  let fog1 = applyFog(vec3f(1.0), p, G.camPos, G.fogDensity, s, mist);
  col = col * (fog1 - fog0) + fog0;
  through *= fog1.g - fog0.g;
  // Up the river's last reach the sea hands over to the river's own surface (water.wgsl).
  let k = 1.0 - smoothstep(40.0, 80.0, frag.coast.x);
  return vec4f(col * k, mix(1.0, clamp(through, 0.0, 1.0), k));
}

// The surface from below (as for the river): Snell's window, total internal reflection.
fn underside(p: vec3f, nUp: vec3f, v: vec3f, s: SkyParams) -> vec4f {
  let tr = refract(-v, -nUp, 1.333);
  let deepCol = vec3f(0.01, 0.06, 0.08) * (G.sunColor * 0.6 + s.zenith * 0.8);
  var col = deepCol;
  if (dot(tr, tr) > 1e-4) {
    let up = normalize(tr);
    let transmit = 1.0 - pow(1.0 - max(dot(up, nUp), 0.0), 5.0);
    col = skyColor(normalize(vec3f(up.x, max(up.y, 0.01), up.z)), s) * transmit;
    col += G.sunColor * pow(max(dot(up, G.sunDir), 0.0), 300.0) * 6.0;
    col = mix(deepCol, col, smoothstep(0.0, 0.25, length(tr)));
  }
  col = underwaterFog(col, p, G.camPos, max(G.underwater, 1.0), G.sunColor, G.zenithColor);
  return vec4f(col, 0.0);
}
