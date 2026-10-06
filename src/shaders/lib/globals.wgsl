// Per-frame values shared by every pass. Bound once as `G` in each entry shader via a
// single `uniforms(gpu)` object, so every entry must declare exactly this struct.
export struct Globals {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  camPos: vec3f,
  time: f32,
  sunDir: vec3f,
  windStrength: f32,
  sunColor: vec3f,
  fogDensity: f32,
  horizonColor: vec3f,
  exposure: f32,
  zenithColor: vec3f,
  gust: f32,
  playerPos: vec3f,
  playerSpeed: f32,
  windDir: vec2f,
  viewport: vec2f,
  // 0 = golden afternoon, 1 = night
  night: f32,
  explore: f32,
  // Ground mist density near the camera (forest, wet hollows) and the mist layer's base height.
  mist: f32,
  mistBase: f32,
  // Deep-forest gloom around the camera (0 open land .. 1 dense core).
  canopy: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  frustum: array<vec4f, 6>,
  // xyz = trail sample position, w = push strength (fades with age).
  trail: array<vec4f, 24>,
}

export const TRAIL_LEN: u32 = 24u;
