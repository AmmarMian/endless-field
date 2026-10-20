// Bakes the FFT's displacement buffer into a filterable texture for the sea surface, in
// meters (height and choppy sideways shift), with the surface's Jacobian in w: below ~0.5 the
// choppy displacement folds the surface over itself, which is where whitecaps break.
const NU: u32 = 256u;

struct Bake {
  patchSize: f32,
  heightScale: f32,
  choppyScale: f32,
  pad: f32,
}

@group(0) @binding(0) var<storage, read> disp: array<vec4f>;
@group(0) @binding(1) var<uniform> B: Bake;

fn at(x: u32, z: u32) -> vec3f {
  let d = disp[(z % NU) * NU + (x % NU)].xyz;
  return vec3f(d.x * B.choppyScale, d.y * B.heightScale, d.z * B.choppyScale);
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let x = min(u32(uv.x * f32(NU)), NU - 1u);
  let z = min(u32(uv.y * f32(NU)), NU - 1u);
  let c = at(x, z);
  let dx = (at(x + 1u, z) - at(x + NU - 1u, z)) * 0.5;
  let dz = (at(x, z + 1u) - at(x, z + NU - 1u)) * 0.5;
  let cell = B.patchSize / f32(NU);
  let jac = (1.0 + dx.x / cell) * (1.0 + dz.z / cell) - (dz.x / cell) * (dx.z / cell);
  return vec4f(c, jac);
}
