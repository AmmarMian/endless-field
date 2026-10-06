import { simplex2d } from "@vgpu/wgsl-std/noise/simplex";
// Sky gradient, sun and aerial perspective. The sky pass and every surface's fog share
// these functions so distant geometry dissolves into exactly the sky behind it.

export struct SkyParams {
  sunDir: vec3f,
  sunColor: vec3f,
  horizon: vec3f,
  zenith: vec3f,
}

// Sky radiance along `dir`, without the sun disk.
export fn skyColor(dir: vec3f, s: SkyParams) -> vec3f {
  let y = max(dir.y, 0.0);
  let mu = max(dot(dir, s.sunDir), 0.0);
  // Horizon band, warming toward the sun.
  var col = mix(s.horizon, s.zenith, pow(y, 0.45));
  let warm = s.sunColor * (0.55 * pow(mu, 6.0) + 0.35 * pow(mu, 64.0));
  col = col + warm * (1.0 - y * 0.6);
  // A soft glow band right above the horizon.
  col = col + s.sunColor * 0.12 * exp(-y * 9.0) * (0.4 + 0.6 * mu);
  // Below the horizon fade to a hazy ground bounce.
  let below = clamp(-dir.y * 4.0, 0.0, 1.0);
  return mix(col, s.horizon * 0.8, below);
}

export fn sunDisk(dir: vec3f, s: SkyParams) -> vec3f {
  let mu = dot(dir, s.sunDir);
  let disk = smoothstep(0.99965, 0.99985, mu);
  let halo = pow(max(mu, 0.0), 900.0) * 2.0;
  return s.sunColor * (disk * 24.0 + halo);
}

// Height-attenuated exponential fog toward the sky color in the view direction.
// Henyey-Greenstein phase function (g > 0 scatters forward, toward the light).
fn hgPhase(cosT: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (12.566371 * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-4), 1.5));
}

// Morning fog density (1/m) at a point: radiation fog pooled under an inversion cap over the
// eastern forest, broken into drifting banks.
fn fogDensity(p: vec3f, cap: f32, t: f32, boost: f32) -> f32 {
  let forest = smoothstep(300.0, 900.0, p.x);
  if (forest <= 0.0) {
    return 0.0;
  }
  // Under the cap the air is saturated; above it the fog thins within a few meters.
  let layer = 1.0 - smoothstep(cap - 6.0, cap + 4.0, p.y);
  let drift = vec2f(0.8, 0.6) * t * 0.6;
  // Fog banks tens of meters across, drifting downwind and slowly reshaping with height.
  let banks = simplex2d((p.xz - drift) * 0.016 + vec2f(p.y * 0.03, 0.0));
  let cover = smoothstep(-0.5, 0.6, banks);
  return (0.012 + 0.02 * boost) * forest * layer * cover;
}

// Morning fog along the view ray to `worldPos`: rgb = in-scattered light, a = transmittance.
// Smooth in space, so overdraw-heavy shaders (grass, leaves) evaluate it per vertex.
// mist: x = local fog boost near the camera, y = ground height under the camera,
// z = canopy gloom around the camera, w = time.
export fn morningFog(worldPos: vec3f, camPos: vec3f, s: SkyParams, mist: vec4f) -> vec4f {
  let d = worldPos - camPos;
  let dist = length(d);
  let dir = d / max(dist, 1e-4);
  // Fog pools under an inversion cap ~16 m above the local ground, and only forms over the
  // forest (x > 300): only the part of the ray inside that slab is marched.
  let cap = mist.y + 16.0;
  let top = cap + 4.0;
  if (max(camPos.x, worldPos.x) <= 300.0 || min(camPos.y, worldPos.y) >= top) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  var t0 = 0.0;
  var t1 = min(dist, 700.0);
  if (camPos.y > top) {
    t0 = (top - camPos.y) / min(dir.y, -1e-4);
  } else if (dir.y > 1e-4) {
    t1 = min(t1, (top - camPos.y) / dir.y);
  }
  if (t1 <= t0) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  // Banks vary over tens of meters, so three samples integrate the slab well.
  let steps = 3;
  let dt = (t1 - t0) / f32(steps);
  var trans = 1.0;
  var light = vec3f(0.0);
  let cosT = dot(dir, s.sunDir);
  // Sun light reaching into the fog (the canopy dims it), plus sky light from above.
  let sunIn = s.sunColor * (hgPhase(cosT, 0.6) * 9.0 + 0.25) * mix(1.0, 0.35, mist.z);
  let skyIn = (s.horizon * 0.6 + s.zenith * 0.4) * mix(1.0, 0.45, mist.z);
  for (var i = 0; i < steps; i = i + 1) {
    let tt = t0 + (f32(i) + 0.5) * dt;
    let rho = fogDensity(camPos + dir * tt, cap, mist.w, mist.x);
    let a = 1.0 - exp(-rho * dt);
    light = light + trans * a * (sunIn + skyIn);
    trans = trans * (1.0 - a);
  }
  return vec4f(light, trans);
}

export fn applyFog(colIn: vec3f, worldPos: vec3f, camPos: vec3f, density: f32, s: SkyParams, mist: vec4f) -> vec3f {
  return applyFogPre(colIn, worldPos, camPos, density, s, morningFog(worldPos, camPos, s, mist));
}

// applyFog with the morning fog already evaluated (e.g. interpolated from the vertices).
export fn applyFogPre(colIn: vec3f, worldPos: vec3f, camPos: vec3f, density: f32, s: SkyParams, fog: vec4f) -> vec3f {
  let d = worldPos - camPos;
  let dist = length(d);
  let dir = d / max(dist, 1e-4);
  let col = colIn * fog.a + fog.rgb;
  let heightFall = 0.018;
  let h0 = camPos.y;
  // Analytic integral of density * exp(-heightFall * y) along the ray.
  let k = heightFall * dir.y;
  var optical = dist;
  if (abs(k) > 1e-4) {
    optical = (1.0 - exp(-k * dist)) / k;
  }
  let amount = 1.0 - exp(-density * exp(-heightFall * (h0 - 10.0)) * optical);
  let fogDir = normalize(vec3f(dir.x, max(dir.y, 0.02), dir.z));
  let inscatter = skyColor(fogDir, s) + s.sunColor * 0.25 * pow(max(dot(dir, s.sunDir), 0.0), 8.0);
  return mix(col, inscatter, clamp(amount, 0.0, 1.0));
}

// Cheap wrapped diffuse that keeps the shadow side warm instead of black.
export fn wrapDiffuse(n: vec3f, l: vec3f, w: f32) -> f32 {
  return max((dot(n, l) + w) / ((1.0 + w) * (1.0 + w)), 0.0);
}

export fn ambientSky(n: vec3f, s: SkyParams) -> vec3f {
  let up = n.y * 0.5 + 0.5;
  return mix(s.horizon * 0.55, s.zenith * 0.9, up);
}
