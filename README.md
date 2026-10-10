# Endless Field

A *Flower*-inspired (thatgamecompany) wind game: an endless procedural grassland rendered
with WebGPU through [vgpu](https://github.com/vercel-labs/vgpu). You are the wind. Carry a
stream of petals, bloom glowing flowers, and watch the dry golden field turn green behind you.

## Run

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + production build
./tools/check.sh   # validate every WGSL entry shader against a real WebGPU device
```

Needs a WebGPU browser (recent Chrome / Edge / Safari; on iPhone, iOS 26).

On phones the game loads small texture copies (`tools/phone-textures.py`, regenerate after
changing a texture) a couple at a time and draws near trees with their lighter mesh. A start
that never finished (a phone killed the page while loading) makes the next one start lighter
still; `?lite=0|1|2` picks a level for one visit.

## Controls

| Input | Action |
| --- | --- |
| Mouse / touch position | Steer |
| Hold click / Space | Gust (speed up) |
| Shift / W, Ctrl / S | Rise, dive |
| A / D | Turn |
| B | Rest: the swallow lands and settles (it also does when left alone for 40 s) |
| O | Settings (quality presets, render scale, grass density, draw distance, frame cap, auto resolution) |
| F | Frame counter (fps, CPU and GPU ms) |
| N | Day / night |
| M | Free roam: walk (WASD, mouse look, Shift sprint) or fly (V; Space / C up and down) |

## What is in it

- **Grass**: GPU-driven. A compute pass places world-anchored blades around the camera,
  frustum-culls them, computes wind ripples and the stream's push, and appends instances for
  indirect draws. Three LOD rings reach the horizon with dithered crossfades.
- **World**: endless procedural hills, subtle biomes (meadow, grove, savanna, forest), a
  meandering river with shallows, a pebble shore and reeds, flower beds, and sparse trees on
  crests. Terrain, river, biome and bed functions are mirrored in WGSL and TypeScript so the
  CPU and GPU always agree.
- **Gameplay**: closed flowers glow; passing through blooms them with a chime, adds their
  petal to your stream and restores the land around them.
- **Birds**: the swallow you fly, meadow sparrows, the river's kingfisher and buzzards in the
  thermals are modeled, rigged and animated in Blender (flap, glide, stoop, landing flare,
  take-off, perched idle, preening, hops and pecks), baked to skinning matrices and blended on
  the GPU.
- **Animals**: brown hares, roe deer and a red fox in the meadows; butterflies (six species)
  over the flower beds; frogs, dragonflies and mallards at the river. Modeled, rigged and
  animated in Blender (`tools/blender/model_animals.py`: grazing, alert, walking and galloping
  gaits, a fox's mousing pounce, a frog's call, a duck up-ending), they react to the wind
  and the grass parts around them. Rush at them and they flee; come gently (no gust, low)
  and linger, and they trust you: hares, deer and the fox run beside you, butterflies dance
  around you, ducks paddle in your wake, a dragonfly flies at your wingtip, the fox springs at
  you in play, and frogs sing when you rest by the river. At night they carry a soft light.
- **Night**: moon and stars, swarming fireflies, glowing flowers and bioluminescent grass.
- **Rendering**: HDR + 4x MSAA, reversed-Z, bloom, ACES tone mapping, GPU timers, frame cap
  and dynamic resolution.

## Asset pipeline

Raw downloads go to `assets-src/` (git-ignored); built web assets live in `public/assets/`.

```sh
node tools/fetch-polyhaven.mjs jacaranda_tree 1k
blender -b --factory-startup --python tools/blender/build_tree.py -- tools/trees/jacaranda.json
blender -b --factory-startup --python tools/blender/build_plants.py -- tools/plants/gazania.json
```

- Trees: each leaf card is refit as a 2-triangle quad, bark is simplified with meshoptimizer,
  and Blender bakes an 8-view impostor (albedo + normals) for far LODs.
- Birds: `blender -b --factory-startup --python tools/blender/model_birds.py -- public/assets/birds
  [--preview <dir>] [--blend <file.blend>]` builds every species; `--blend` saves the rigged
  scenes with their actions for editing, `--preview` renders frames of each clip.
- Animals: `blender -b --factory-startup --python tools/blender/model_animals.py -- public/assets/animals
  [--species hare,deer,...] [--preview <dir>] [--clips a,b]`.
- Plants: variants are re-centered, simplified, and their diffuse + alpha merged with color
  bleeding for clean mips.

## Credits

All third-party assets are CC0 from [Poly Haven](https://polyhaven.com):
Jacaranda Tree, Island Tree 02, Gazania, Ursinia, Empodium, Dandelion 01, Heliophila
(models) and Ganges River Pebbles (texture). Everything else (grass, flowers, petals, sky,
water, audio) is procedural.
