// River fish (model: tools/blender/model_fish.py): the body swims with a travelling wave that
// grows toward the tail; wet skin with a sheen. Fogged by the water when seen from inside it.
import { Globals } from "./lib/globals.wgsl";
import { SkyParams, applyFog, ambientSky, underwaterFog } from "./lib/atmosphere.wgsl";

struct Fish {
  // xyz = position, w = heading (yaw)
  pos: vec4f,
  // x = pitch, y = length (m), z = swim phase (rad), w = tail beat amplitude
  pose: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> fish: array<Fish>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) ao: f32,
}

@vertex
fn vs_main(@location(0) p: vec4f, @location(1) n: vec4f, @location(2) t: vec2f, @location(3) e: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let f = fish[ii];
  var lp = p.xyz;
  // Body wave: little at the head, most at the tail (carangiform swimming).
  let u = t.x;
  let amp = f.pose.w * (0.02 + u * u * 0.16);
  let wave = sin(f.pose.z - u * 5.5);
  lp.x += wave * amp;
  var ln = normalize(n.xyz + vec3f(-cos(f.pose.z - u * 5.5) * amp * 5.5, 0.0, 0.0));
  lp *= f.pose.y;
  let cp = cos(f.pose.x);
  let sp = sin(f.pose.x);
  lp = vec3f(lp.x, lp.y * cp + lp.z * sp, -lp.y * sp + lp.z * cp);
  ln = vec3f(ln.x, ln.y * cp + ln.z * sp, -ln.y * sp + ln.z * cp);
  let cy = cos(f.pos.w);
  let sy = sin(f.pos.w);
  lp = vec3f(lp.x * cy + lp.z * sy, lp.y, -lp.x * sy + lp.z * cy);
  ln = vec3f(ln.x * cy + ln.z * sy, ln.y, -ln.x * sy + ln.z * cy);
  let world = f.pos.xyz + lp;
  var out: VOut;
  out.pos = G.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.normal = ln;
  out.albedo = pow(e.rgb, vec3f(2.2));
  out.ao = e.a;
  return out;
}

@fragment
fn fs_main(frag: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = SkyParams(G.sunDir, G.sunColor, G.horizonColor, G.zenithColor);
  var n = normalize(frag.normal);
  if (!front) {
    n = -n;
  }
  let v = normalize(G.camPos - frag.world);
  let l = G.sunDir;
  let wrap = clamp((dot(n, l) + 0.25) / 1.25, 0.0, 1.0);
  var col = frag.albedo * (ambientSky(n, s) * 0.7 * frag.ao + G.sunColor * wrap * 0.8);
  // Wet, scaled skin: a sharp highlight and a silvery fresnel sheen.
  let h = normalize(l + v);
  col += G.sunColor * pow(max(dot(n, h), 0.0), 90.0) * 0.8;
  col += mix(G.horizonColor, G.zenithColor, 0.5) * pow(1.0 - abs(dot(n, v)), 4.0) * 0.35;
  col = applyFog(col, frag.world, G.camPos, G.fogDensity, s, vec4f(G.mist, G.mistBase, G.canopy, G.time));
  col = underwaterFog(col, frag.world, G.camPos, G.underwater, G.sunColor, G.zenithColor);
  return vec4f(col, 1.0);
}
