// Sunflowers: Blender-modeled plants deformed by the simulated stem physics (sunflower-sim).
// The stem follows the static deflection curve of a cantilever with an end load; the head
// rides on the stem tip, tilting with the tip slope and twisting with the torsion mode;
// leaves and petals add their own flutter driven by the local wind speed.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFogPre, morningFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { lodKeep, rotateYaw } from "./lib/tree.wgsl";

struct Plant {
  root: vec4f,
  info: vec4f,
}

struct State {
  bend: vec4f,
  twist: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> plants: array<Plant>;
@group(0) @binding(2) var<storage, read> state: array<State>;
// Per draw-instance: x = plant index, y = LOD crossfade (f32 bits, see lodKeep).
@group(0) @binding(3) var<storage, read> visible: array<vec2u>;
// Per variant: xyz = head pivot (model space), w = stem height.
@group(0) @binding(4) var<storage, read> variants: array<vec4f>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var albedoTex: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  // Morning fog evaluated per vertex (see morningFog).
  @location(14) fog: vec4f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) ao: f32,
  @location(4) @interpolate(flat) part: u32,
  @location(5) @interpolate(flat) lodFade: f32,
  @location(6) height: f32,
}

// Rodrigues rotation of v about unit axis k by angle a.
fn rotateAxis(v: vec3f, k: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

@vertex
fn vs_main(
  @location(0) p: vec4f,
  @location(1) n: vec4f,
  @location(2) t: vec2f,
  @location(3) e: vec4f,
  @builtin(instance_index) ii: u32,
) -> VOut {
  let vis = visible[ii];
  let plant = plants[vis.x];
  let st = state[vis.x];
  let variant = variants[u32(plant.info.y)];
  let scale = plant.root.w;
  let yaw = plant.info.x;
  let cs = vec2f(cos(yaw), sin(yaw));
  let H = variant.w;
  let part = u32(round(p.w));

  // Head displacement in model space (undo the yaw and scale).
  let dw = vec3f(st.bend.x, 0.0, st.bend.y) / scale;
  let dl = rotateYaw(dw, vec2f(cs.x, -cs.y));
  let s = e.x;
  // Cantilever with an end load: w(s) = s^2 (3 - s) / 2, tip slope 1.5 / H.
  let shape = s * s * (3.0 - s) * 0.5;
  let dlen = length(dl);
  // Bending keeps the stem's length: points sink as they swing out.
  let sink = dlen * dlen * shape / (1.3 * H);

  var local = p.xyz;
  var nrm = n.xyz;
  let windMag = st.twist.z;
  let phase = e.w * 6.2831853;
  if (part >= 2u) {
    // Head: rigid about the pivot. Tilt with the tip slope, twist with the torsion mode.
    let pivot = variant.xyz;
    var r = local - pivot;
    if (dlen > 1e-4) {
      let axis = normalize(cross(vec3f(0.0, 1.0, 0.0), dl));
      let tilt = dlen * 1.5 / H;
      r = rotateAxis(r, axis, tilt);
      nrm = rotateAxis(nrm, axis, tilt);
    }
    let tw = vec2f(cos(st.twist.x), sin(st.twist.x));
    r = rotateYaw(r, tw);
    nrm = rotateYaw(nrm, tw);
    local = pivot + r + dl - vec3f(0.0, sink, 0.0);
    if (part == 3u) {
      // Ray petals quiver in the wind.
      local = local + nrm * e.y * (0.004 + 0.0025 * windMag) * sin(G.time * 13.0 + phase + plant.info.z * 20.0);
    }
  } else {
    local = local + dl * shape - vec3f(0.0, sink, 0.0);
    if (part == 1u) {
      // Leaves: flutter about the midrib, and lift and stream with the wind.
      let f = e.y;
      let gustBeat = sin(G.time * (6.5 + 3.0 * e.w) + phase + plant.info.z * 31.0) * 0.6 + sin(G.time * 11.3 + phase * 2.0) * 0.4;
      local = local + nrm * f * f * (0.01 + 0.006 * windMag) * gustBeat;
      let streamDir = rotateYaw(vec3f(G.windDir.x, 0.0, G.windDir.y), vec2f(cs.x, -cs.y));
      local = local + (streamDir * 0.02 * windMag + vec3f(0.0, 0.006 * windMag, 0.0)) * f * f;
    }
  }

  let world = plant.root.xyz + rotateYaw(local * scale, cs);
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.fog = morningFog(out.world, G.camPos, SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor), vec4f(G.mist, G.mistBase, G.canopy, G.time));
  out.normal = rotateYaw(nrm, cs);
  out.uv = t;
  out.ao = e.z;
  out.part = part;
  out.lodFade = bitcast<f32>(vis.y);
  out.height = clamp(p.y / max(H, 0.1), 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  if (!lodKeep(frag.lodFade, frag.pos.xy)) {
    discard;
  }
  let texel = textureSample(albedoTex, samp, frag.uv);
  let dims = vec2f(textureDimensions(albedoTex));
  let duv = max(length(dpdx(frag.uv * dims)), length(dpdy(frag.uv * dims)));
  let mip = max(log2(duv), 0.0);
  var a = texel.a * (1.0 + mip * 0.25);
  a = clamp((a - 0.5) / max(fwidth(a), 1e-4) + 0.5, 0.0, 1.0);
  if (a < 0.01) {
    discard;
  }
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  let v = normalize(G.camPos - frag.world);
  let l = G.sunDir;
  // Inside the field the lower plant is shaded by its neighbours' leaves.
  let canopy = mix(0.45, 1.0, smoothstep(0.15, 0.85, frag.height));
  let ao = frag.ao * canopy;
  var albedo = texel.rgb;
  // The far face card is single: its back shows the green calyx.
  if (frag.part == 2u && !front) {
    albedo = vec3f(0.17, 0.27, 0.07);
  }
  var diff = wrapDiffuse(n, l, 0.4);
  // Thin petals and leaves glow when the sun is behind them.
  let backlit = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0);
  var trans = 0.0;
  var spec = 0.0;
  if (frag.part == 3u) {
    trans = 0.85;
    diff = wrapDiffuse(n, l, 0.8);
  } else if (frag.part == 1u) {
    trans = 0.6;
    let hv = normalize(v + l);
    spec = pow(max(dot(n, hv), 0.0), 40.0) * 0.12;
  } else if (frag.part == 2u) {
    diff = diff * 0.9;
  }
  var col = albedo * (ambientSky(n, s) * 0.55 * ao + G.sunColor * (diff * ao * canopy) * 0.9);
  col = col + albedo * G.sunColor * backlit * trans * canopy * (albedo + 0.25);
  col = col + G.sunColor * spec * canopy;
  // At night the petals hold a faint warm glow so the field stays readable in the dark.
  let petalLike = select(0.0, 1.0, frag.part == 3u) + select(0.0, smoothstep(0.25, 0.6, albedo.r - albedo.b), frag.part == 2u && front);
  col = col + albedo * vec3f(1.0, 0.75, 0.35) * petalLike * G.night * 0.14;
  col = applyFogPre(col, frag.world, G.camPos, G.fogDensity, s, frag.fog);
  return vec4f(col, a);
}
