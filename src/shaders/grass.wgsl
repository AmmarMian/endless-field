// Grass blades, one instance per blade the compute pass kept (grass-cull.wgsl). Drawing and
// shading live in lib/grass-blade.wgsl (shared with the direct fallback).
import { Globals } from "./lib/globals.wgsl";
import { Blade } from "./lib/field.wgsl";
import { GrassEnv, GrassOut, bladeVertex, shadeBlade } from "./lib/grass-blade.wgsl";
import { LANTERN_COLOR, lanternFirst, lanternTerm } from "./lib/path.wgsl";

override NSEG: u32 = 5u;

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> blades: array<Blade>;

fn grassEnv() -> GrassEnv {
  return GrassEnv(G.viewProj, G.camPos, G.time, G.sunDir, G.windStrength, G.sunColor, G.fogDensity, G.horizonColor, G.night, G.zenithColor, G.season, G.playerPos, G.playerGlow, G.viewport, G.mist, G.mistBase, G.canopy, G.wet, G.underwater);
}

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

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GrassOut {
  var out = bladeVertex(blades[ii], vi, NSEG, grassEnv());
  out.lamp = lampLight(out.world);
  return out;
}

@fragment
fn fs_main(frag: GrassOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  return shadeBlade(frag, grassEnv());
}
