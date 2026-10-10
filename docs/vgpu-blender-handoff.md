# vgpu + Blender: handoff notes

What building Endless Field taught about making high-quality WebGPU scenes with
[vgpu](https://github.com/vercel-labs/vgpu) and Blender, written for another project starting
from scratch. Everything here was learned the hard way in this repo; file paths point at working
examples you can copy.

---

## 1. The working loop (most important)

**Never judge a change from the code. Render it, look at it, measure it.** Almost every real
improvement here came from a screenshot or a pixel diff that contradicted what the code
"obviously" did.

1. Make the change.
2. `./tools/check.sh` validates every WGSL entry shader against a real WebGPU device (catches
   errors that the dev server only shows at runtime).
3. `npx tsc --noEmit`.
4. Capture the running app headless with Playwright + real WebGPU (`tools/shot.mjs`), look at
   the image, and where it matters compare numbers (fps, GPU ms, pixel diffs).
5. Commit with a message that says *why*.

### Headless capture with real WebGPU

`tools/shot.mjs` (single shot) and `tools/dev/burst2.mjs` (a frame every N ms) launch Chromium with:

```js
chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal", "--ignore-gpu-blocklist"] });
```

Lessons:

- **Block Google Fonts** in captures (`page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort())`)
  and navigate with `waitUntil: "domcontentloaded"`. A slow font host otherwise hangs `goto` for 30 s.
- **Run captures one at a time.** Four parallel browsers stalled the Vite dev server.
- Use a fixed port (`npx vite --port 5199`).
- `--eval` JS runs after load; expose a debug object on `window` (here `window.__ef`) so captures
  can drive the scene.

### Debug hooks worth building on day one (`src/main.ts`, `debug`)

- `fixedCamera = { pos, target }`: frame exactly what you want to judge; re-set it every
  `requestAnimationFrame` to follow a moving object.
- `fixedTime`: freeze animated time (wind, clouds) so two frames differ only by your change.
- A pause key: freezes the simulation (moving actors and their side effects add noise to diffs).
- `hide.{grass,terrain,trees,...}`: see what each system contributes.
- `setDay(phase)`: test day, golden hour, night.
- Per-system counters (here grass blade counts read back from the GPU) and the frame stats text.

### Measure, don't eyeball, subtle problems

- **Contact sheets**: for animation, render N frames of each clip into one image
  (`tools/dev/sheet.py`). One look shows a whole cycle.
- **Isolate the variable.** To find LOD popping, the camera stays still and only the point the
  LOD rings are measured from moves (`grass.lodShift`, `tools/dev/lodpop.mjs`). The diff is then
  pure popping. With the camera moving, parallax swamps the signal.
- **Small steps separate pops from smooth change.** At a 0.25 m step a smooth fade changes
  almost nothing; a pop is still full contrast. Count pixels whose change exceeds ~20/255.
- **On/off comparisons per row** (grass hidden vs shown) show where a system stops contributing
  with distance.
- Look at a **crop at full resolution** before concluding anything; downscaled grids hide detail.

---

## 2. vgpu

- Use vgpu, not three.js. Read docs locally: `npm exec --no -- vgpu docs cat getting-started.md`,
  `vgpu docs grep -i <term>`. The repo's `.claude/skills/vgpu` skill is installed with
  `npx skills add vercel-labs/vgpu`.
- **Imported WGSL modules must be pure**: no `@group/@binding` outside the entry file. Shared code
  exports structs and functions; bindings live in each entry (see `src/shaders/lib/*.wgsl`).
- Import paths in a lib are relative to that lib (`./season.wgsl`, not `./lib/season.wgsl`).
- `override` constants are cheap per-draw specialisation (LOD segment counts, bone counts, species).
- One shared uniform block (`Globals`) for camera, time, sun, weather; set once per frame.

### WGSL performance traps

- **Never pass a huge struct by value into functions called per vertex or per pixel.** Passing
  the whole `Globals` (with its arrays) into a shared grass function made the GPU about 20x slower
  (4 fps). Pass a small purpose-built struct (`GrassEnv` in `src/shaders/lib/grass-blade.wgsl`).
- Derivatives (`fwidth`, `dpdx`) must be taken in uniform control flow, before branching.
- Fade fine procedural detail by its screen frequency (`1 - smoothstep(a, b, length(fwidth(q)))`)
  so it never aliases into shimmer at distance.
- Reversed-Z depth (`depth: { compare: "greater" }`) for long view distances.

### GPU-driven rendering

- Compute pass places and culls instances, appends with `atomicAdd` into a storage buffer, and
  feeds an indirect draw (`src/shaders/grass-cull.wgsl`, `src/world/grass.ts`).
- **Some phones run that compute pass and draw nothing.** Keep the placement logic in one
  function and add a direct fallback: a plain instanced draw over the whole grid where each
  vertex computes its own instance and degenerate instances collapse. Detect the failure by
  reading back the indirect count a few seconds in; if it stays 0, switch.

---

## 3. Mobile

- Phones fail silently. Put **errors above the loading screen**, report **device loss**, and after
  ~30 s show **what is still pending**. Otherwise a crash looks like "loading forever".
- **Texture memory is the killer.** Ship small copies for phones (`tools/phone-textures.py`,
  512 px; impostor atlases 1024) and load at most two at a time.
- **ImageBitmap lifetime**: closing a bitmap right after `queue.copyExternalImageToTexture` can
  upload nothing. Keep the mip bitmaps until `await queue.onSubmittedWorkDone()`, then close them.
- When resizing images with alpha, **resize each channel separately** (PIL resizes RGBA
  premultiplied, which blackens the colour bleed under transparent texels and ruins
  alpha-tested foliage).
- **Crash-safe boot tiers**: mark a start as pending in localStorage, clear it after 8 s of
  running; if the next load finds it pending, start lighter. Do it on touch devices only (a
  desktop reload mid-load is not a crash) and allow `?lite=N` for one visit.
- iOS: WebGPU is on by default in Safari 26; on iOS 18 it is a feature flag. Say so when
  `navigator.gpu` is missing.
- Emulated iPhone (Playwright WebKit and Chromium device profiles) catches a lot but not
  everything; get real-device reports (`tools/dev/mem.mjs`, `tools/dev/wk.mjs`).

---

## 4. Asset delivery

- Images as **WebP** (`exact=True` keeps colour under transparent texels; quality ~88, normals ~92,
  `alpha_quality=100`): about 3x smaller than PNG/JPG.
- Meshes as **gzip** copies decompressed with `DecompressionStream`. Dev servers may already send
  `.gz` with `Content-Encoding: gzip` (the browser unpacks it), so **check the gzip magic bytes
  (1f 8b)** instead of assuming.
- Keep originals as the editable sources; generate delivery copies with a script
  (`tools/web-assets.py`) and a manifest the loaders consult, with fallback to originals.
- Result here: 69 MB down to about 27 MB.
- Vertex formats: `float16x4` positions (pack a part id in w), `snorm8x4` normals, `unorm8x4`
  colours and AO, `uint8x4` bone indices and weights. Convert Blender Z-up to Y-up once at export
  and flip triangle winding.

---

## 5. Blender as a code-driven asset pipeline

- **Scripts are the source of truth**, run headless:
  `blender -b --factory-startup --python tools/blender/model_birds.py -- <out dir> [--preview <dir>] [--blend <file>]`.
  Parameters at the top, deterministic output, a `--preview` mode that renders the result, and
  `--blend` to save the scene for hand editing.
- Install Blender rather than asking; it lives at `/Applications/Blender.app/Contents/MacOS/Blender`.
- **Smooth organic bodies**: metaballs (ellipsoids, capsules for limbs) → voxel remesh → smooth →
  decimate. Limbs, necks and tails then grow out of the body without seams.
  - Blender polygonises metaballs no finer than ~5 mm: build small creatures 10x larger and
    scale down before remeshing.
  - Remove the source objects before preview renders, or they show up unanimated.
- Wings: a lofted airfoil (NACA-like thickness, camber) through planform stations, resampled with
  Catmull-Rom; scalloped trailing edges; separate "finger" primaries for broad wings.
- Paint with vertex colours computed from position and normal (cheap, no UVs to unwrap); the
  shader adds feather and fur detail procedurally.
- **Preview everything** with EEVEE from several views, then a contact sheet per clip.

### Rigging and animation

- Build the armature in code. **Set every bone's roll deliberately** (wings: Z up; legs: Z
  forward), or the same rotation means different things on different bones.
- Weights from geometry (distance to bone over its radius, two bones per vertex), with **forced
  bones for rigid parts** (wings, ears) and no cross-side bleeding.
- Author clips as **key poses** keyed on pose bones (Bezier-eased), then **bake** each frame to
  skinning matrices (pose × inverse bind), stored as float16 3x4 per bone per frame.
- Runtime (`src/world/bird-model.ts`, `src/shaders/birds.wgsl`): the shader blends two baked
  frames and crossfades between two clips; the CPU only advances clip times. Hundreds of animated
  instances cost almost nothing.
- Play locomotion at a rate matched to speed (`speed / (stride × size)`) so feet do not slide.
- **Quality lessons:**
  - Gaits sampled from sine-based procedural leg cycles look mechanical. Key poses of the real
    phases (gathered, push-off, airborne stretch, landing) on a **periodic Catmull-Rom spline**
    are far better.
  - Idle loops must never hold still between keys: interpolate continuously and add breathing.
  - Standing animals should step round when turning, not spin on the spot.
  - Animal running was judged still "not convincing" even after this: for creatures, study
    reference footage frame by frame, or start from motion-captured or artist-made rigs.
  - Landing (birds): time the clip so touchdown in the animation meets the perch exactly, flare
    with braking beats and feet forward, then fold the wings.

---

## 6. Large-scale scenes: grass, LOD, distance

- Nested LOD rings where each ring keeps every 3rd blade of the finer ring: a blade keeps its
  identity (position, height, colour) across rings, so nothing swaps.
- Crossfades must be **continuous in every per-blade parameter**: a leaving blade shrinks to
  exactly zero where it is dropped; width, height and shape features (seed heads) all ease into
  the next ring's values. Any one of these done wrong reads as "grass appearing".
- Wide crossfade bands (~40% of a ring) beat narrow ones; fading across the whole ring was
  measured worse.
- **Do not cull by slope with distance.** Far hillsides are exactly what the eye sees; removing
  grass there makes distant hills look empty.
- A very sparse outer ring of wide blades out to ~1 km keeps distant hills textured.
- Far blades turn broadside to the camera (edge-on they cover nothing).
- The ground under far grass must be shaded like the grass canopy (blade colour, lit like
  blades, clumps, wind waves) from fairly close (~25 m), so gaps between sparse blades read as
  grass, not soil.
- Anything streamed in cells (flowers, props) needs a distance grow-in, or whole rows pop.

---

## 7. Lighting and look

- HDR target, bloom, ACES tone mapping, reversed-Z, 4x MSAA; alpha-to-coverage for foliage
  (no discard).
- Wrapped diffuse and back-lit translucency for thin things (blades, feathers, leaves).
- Fog and atmosphere shared by every shader through one library function.
- At night, give important things a soft warm halo (an additive billboard) and a faint inner
  glow so they stay readable without looking like lamps.

---

## 8. Performance

- GPU timestamp queries when available (frame counter shows CPU and GPU ms), a frame cap and
  dynamic resolution.
- Measure before and after with the same camera and settings; state numbers, not impressions.
- Budgets that held here on an M-series Mac: ~9 ms GPU at 1280x720 for the full scene.

---

## 9. Working with this user

- Wants top quality and to see it verified: show screenshots or numbers, not claims.
- Prefers diegetic design: no modal pop-ups or toasts for in-world encounters; creatures live
  their own lives (no companions that follow you).
- Wants tools installed rather than being asked; commits and pushes when asked
  (GitHub Pages deploys from `main`).
- Keeps unfinished work on a branch rather than shipping it (`animals` branch here).

---

## Files to copy

| What | Where |
| --- | --- |
| Shader validation script | `tools/check.sh` |
| Headless WebGPU screenshot | `tools/shot.mjs` |
| Frame bursts, popping and coverage tests | `tools/dev/burst2.mjs`, `tools/dev/lodpop.mjs`, `tools/dev/popdiff.mjs`, `tools/dev/cover.mjs` |
| Phone memory and download profiling | `tools/dev/mem.mjs`, `tools/dev/dl.mjs`, `tools/dev/wk.mjs` |
| Contact sheets | `tools/dev/sheet.py` |
| Delivery and phone texture copies | `tools/web-assets.py`, `tools/phone-textures.py` |
| Rigged, animated, baked Blender models | `tools/blender/model_birds.py` (and `model_animals.py` on the `animals` branch) |
| Skinned animation runtime | `src/world/bird-model.ts`, `src/shaders/birds.wgsl` |
| GPU-driven grass with direct fallback | `src/world/grass.ts`, `src/shaders/grass-cull.wgsl`, `src/shaders/lib/grass-blade.wgsl` |
| Asset loaders (WebP, gzip, phone copies) | `src/engine/assets.ts`, `src/engine/textures.ts` |
