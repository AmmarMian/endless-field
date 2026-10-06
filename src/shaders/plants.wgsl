// Wildflower bed plants (Poly Haven ground cover): alpha-to-coverage foliage with wind sway.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFogPre, morningFog, ambientSky, wrapDiffuse } from "./lib/atmosphere.wgsl";
import { LifeCell, lifeIndex, lifeKey } from "./lib/field.wgsl";

struct Plant {
  // xyz = root, w = scale
  root: vec4f,
  // xy = (cos, sin) yaw, z = seed, w = distance fade
  rot: vec4f,
  // y = light under the canopy (computed once on the CPU)
  env: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> plants: array<Plant>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var albedoTex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> life: array<LifeCell>;

struct PlantParams {
  // How much foliage takes the field's dry tone before restoration (wildflowers, ferns).
  dryTint: f32,
  // Night glow of non-leaf texels (petals, mushroom caps).
  nightGlow: f32,
  pad0: f32,
  pad1: f32,
}
@group(0) @binding(5) var<uniform> P: PlantParams;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  // Morning fog evaluated per vertex (see morningFog).
  @location(14) fog: vec4f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) ao: f32,
  @location(4) @interpolate(flat) fade: f32,
  @location(5) @interpolate(flat) life: f32,
  @location(6) @interpolate(flat) shade: f32,
}

fn rotY(v: vec3f, cs: vec2f) -> vec3f {
  return vec3f(cs.x * v.x + cs.y * v.z, v.y, -cs.y * v.x + cs.x * v.z);
}

@vertex
fn vs_main(
  @location(0) p: vec4f,
  @location(1) n: vec4f,
  @location(2) t: vec2f,
  @location(3) e: vec4f,
  @builtin(instance_index) ii: u32,
) -> VOut {
  let inst = plants[ii];
  let s = inst.root.w * inst.rot.w;
  var world = inst.root.xyz + rotY(p.xyz * s, inst.rot.xy);
  let h = e.x;
  let phase = inst.rot.z * 40.0;
  let sway = sin(G.time * 1.9 + phase + world.x * 0.3) * 0.5 + sin(G.time * 3.3 + phase * 1.7) * 0.25;
  let windOff = (G.windDir * (0.5 + sway) * G.windStrength) * h * h * 0.08 * inst.root.w;
  world = world + vec3f(windOff.x, 0.0, windOff.y);
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.fog = morningFog(out.world, G.camPos, SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor), vec4f(G.mist, G.mistBase, G.canopy, G.time));
  out.normal = rotY(n.xyz, inst.rot.xy);
  out.uv = t;
  out.ao = e.z;
  out.fade = inst.rot.w;
  let cell = vec2i(floor(inst.root.xz));
  let lc = life[lifeIndex(cell)];
  out.life = select(0.0, lc.life, lc.key == lifeKey(cell));
  out.shade = inst.env.y;
  return out;
}

@fragment
fn fs_main(frag: VOut) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  let texel = textureSample(albedoTex, samp, frag.uv);
  let dims = vec2f(textureDimensions(albedoTex));
  let duv = max(length(dpdx(frag.uv * dims)), length(dpdy(frag.uv * dims)));
  var a = texel.a * (1.0 + max(log2(duv), 0.0) * 0.3);
  a = clamp((a - 0.5) / max(fwidth(a), 1e-4) + 0.5, 0.0, 1.0);
  if (a < 0.01) {
    discard;
  }
  let v = normalize(G.camPos - frag.world);
  var n = normalize(frag.normal);
  if (dot(n, v) < 0.0) {
    n = -n;
  }
  // Foliage (green-dominant texels) shares the field's dry gold until restored; petals keep color.
  let leafness = smoothstep(0.02, 0.12, texel.g - max(texel.r, texel.b));
  let dry = vec3f(dot(texel.rgb, vec3f(0.5, 0.4, 0.1))) * vec3f(1.25, 0.95, 0.5);
  let base = mix(texel.rgb, dry, leafness * (1.0 - frag.life) * 0.6 * P.dryTint);
  let l = G.sunDir;
  let back = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0) * 0.8;
  var col = base * (ambientSky(n, s) * 0.6 * frag.ao * mix(0.5, 1.0, frag.shade) + G.sunColor * (wrapDiffuse(n, l, 0.4) + back) * 0.85 * frag.ao * frag.shade * frag.shade);
  let petal = 1.0 - leafness;
  col = col + texel.rgb * petal * G.night * 0.9 * P.nightGlow * (0.7 + 0.3 * sin(G.time * 1.1 + frag.world.x * 0.7 + frag.world.z));
  col = applyFogPre(col, frag.world, G.camPos, G.fogDensity, s, frag.fog);
  return vec4f(col, a);
}
