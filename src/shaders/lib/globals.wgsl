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
  // Rain intensity [0, 1], ground wetness [0, 1] (lags the rain), season in [0, 4):
  // 0 summer, 1 autumn, 2 winter, 3 spring (fractional values blend).
  rain: f32,
  wet: f32,
  season: f32,
  // Camera below the water's surface (0..1, eased) and the surface height above it.
  underwater: f32,
  waterY: f32,
  // The swallow's own soft light (0 by day .. ~1 at night), lighting the grass around it.
  playerGlow: f32,
  // A wingtip cutting the water (0..1): the river splits along the wind's trail.
  dip: f32,
  // A rainbow after a shower (0..1).
  rainbow: f32,
  // The swallow resting on the ground at playerPos (0..1, eased): the grass lies down there.
  rest: f32,
  frustum: array<vec4f, 6>,
  // xyz = trail sample position, w = push strength (fades with age).
  trail: array<vec4f, 24>,
  // Animals on the ground near the camera: xyz, w = radius (0: unused). The grass parts.
  critters: array<vec4f, 16>,
  // Lantern brightness by index (4 per vec4): 0 dark, 1 lit, above 1 while flaring.
  lamps: array<vec4f, 16>,
  // Lantern positions (xyz). w of entries 0..2 carries: count, first x, mean x spacing.
  lampPos: array<vec4f, 64>,
}

export const TRAIL_LEN: u32 = 24u;
