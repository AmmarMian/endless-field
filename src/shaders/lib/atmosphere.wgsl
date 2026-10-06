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
export fn applyFog(colIn: vec3f, worldPos: vec3f, camPos: vec3f, density: f32, s: SkyParams, mist: vec3f) -> vec3f {
  let d = worldPos - camPos;
  let dist = length(d);
  let dir = d / max(dist, 1e-4);
  var col = colIn;
  // Ground mist: a dense layer hugging the ground (base mist.y), lit by forward-scattered sun.
  if (mist.x > 0.001) {
    let mk = 0.11;
    let km = mk * dir.y;
    var optM = dist;
    if (abs(km) > 1e-4) {
      optM = (1.0 - exp(-km * dist)) / km;
    }
    let amountM = 1.0 - exp(-mist.x * 0.016 * exp(-mk * max(camPos.y - mist.y, -25.0)) * optM);
    let fwd = pow(max(dot(dir, s.sunDir), 0.0), 5.0);
    // mist.z: canopy gloom around the camera darkens the mist in the deep forest.
    let mistCol = (mix(s.horizon * 0.55 + s.zenith * 0.25, s.sunColor * 0.5, 0.35) + s.sunColor * 0.45 * fwd) * mix(1.0, 0.32, mist.z);
    col = mix(col, mistCol, clamp(amountM, 0.0, 0.85));
  }
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
