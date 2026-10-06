// Subtle biomes from one very low-frequency, domain-warped field. Mirrored by
// src/world/biome.ts. x = grove amount (clustered trees), y = open meadow amount,
// z = forest amount (dense woodland, the moistest ground).
import { gnoise } from "./terrain.wgsl";

export fn biome(xz: vec2f) -> vec3f {
  let w = vec2f(gnoise(xz * 0.0011 + vec2f(3.7, 1.3)), gnoise(xz * 0.0011 + vec2f(-8.1, 4.4))) * 380.0;
  let m = gnoise((xz + w) * (1.0 / 1150.0) + vec2f(21.0, -13.0));
  // To the east lies the great forest: it overrides the open biomes.
  let east = smoothstep(380.0, 900.0, xz.x);
  let forest = max(smoothstep(0.36, 0.48, m), east);
  return vec3f(smoothstep(0.1, 0.3, m) * (1.0 - east), smoothstep(0.08, 0.28, -m) * (1.0 - east), forest);
}

// The forest's dense core: shade under the canopy deepens eastward into the realm.
export fn deepForest(x: f32) -> f32 {
  return smoothstep(1000.0, 1900.0, x);
}

// Light reaching the forest floor (1 open .. ~0.3 under the dense core canopy).
export fn canopyLight(x: f32, forest: f32) -> f32 {
  return 1.0 - forest * mix(0.35, 0.7, deepForest(x));
}
