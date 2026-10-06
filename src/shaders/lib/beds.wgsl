// Flower beds: deterministic per-cell clearings filled with wildflowers. Mirrored by
// src/world/bed-shape.ts so the CPU places plants exactly where the GPU clears the grass.
import { gnoise, hash2i } from "./terrain.wgsl";
import { biome } from "./biome.wgsl";

export const BED_CELL: f32 = 64.0;
export const BED_SPECIES: u32 = 5u;

export struct BedInfo {
  center: vec2f,
  radius: f32,
  species: u32,
  exists: bool,
}

export fn bedInCell(cell: vec2i) -> BedInfo {
  let h = hash2i(cell + vec2i(7919, -104729));
  let h2 = hash2i(cell + vec2i(-31337, 2203));
  var b: BedInfo;
  let bio = biome((vec2f(cell) + 0.5) * BED_CELL);
  b.exists = f32(h & 0xFFFFu) / 65536.0 < 0.3 * (1.0 + bio.y * 0.9 - bio.x * 0.6 - bio.z * 0.4);
  let u = f32((h >> 16u) & 0xFFu) / 255.0;
  let v = f32((h >> 24u) & 0xFFu) / 255.0;
  b.center = (vec2f(cell) * BED_CELL) + vec2f(14.0) + vec2f(u, v) * (BED_CELL - 28.0);
  b.radius = 6.0 + f32(h2 & 0xFFu) / 255.0 * 4.0;
  b.species = (h2 >> 8u) % BED_SPECIES;
  return b;
}

// x = inside amount (1 deep inside, soft edge), y = species index.
export fn bedMask(xz: vec2f) -> vec2f {
  let cell = vec2i(floor(xz / BED_CELL));
  let b = bedInCell(cell);
  if (!b.exists) {
    return vec2f(0.0, 0.0);
  }
  let wobble = 1.0 + 0.28 * gnoise(xz * 0.12 + vec2f(cell) * 3.1);
  let d = distance(xz, b.center) / (b.radius * wobble);
  return vec2f(1.0 - smoothstep(0.45, 1.0, d), f32(b.species));
}

export fn bedColor(species: u32) -> vec3f {
  var c = array<vec3f, 5>(
    vec3f(1.0, 0.42, 0.08),
    vec3f(1.0, 0.62, 0.12),
    vec3f(1.0, 0.85, 0.15),
    vec3f(0.98, 0.8, 0.1),
    vec3f(0.85, 0.9, 1.0),
  );
  return c[min(species, 4u)];
}
