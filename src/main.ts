import { clock, frameLoop, init, timer, type Frame, type FrameLoopHandle } from "vgpu";
import { SettingsPanel, loadSettings, type Settings } from "./ui/settings";
import { WorldMap } from "./ui/map";
import { Camera, type Vec3 } from "./engine/camera";
import { GOLDEN_HOUR, Globals, NIGHT, mixAtmosphere, seasonWeights as seasonWeightsTs, weatherAtmosphere } from "./engine/globals";
import { Renderer } from "./engine/renderer";
import { loadTexture } from "./engine/textures";
import { Audio } from "./game/audio";
import { Input } from "./game/input";
import { PetalStream } from "./game/petals";
import { Player } from "./game/player";
import { FreeCam } from "./game/freecam";
import { Flowers, PALETTES } from "./world/flowers";
import { Grass } from "./world/grass";
import { LifeMap } from "./world/life";
import { Terrain } from "./world/terrain";
import { Fireflies } from "./world/fireflies";
import { Water } from "./world/water";
import { setWorldSeed, mountainZone, riverCenter, riverHalfWidth, riverWater, terrainHeightM as terrainHeight } from "./world/height";
import { biome } from "./world/biome";
import { ecology } from "./world/ecology";
import { Trees } from "./world/trees";
import { loadMountains } from "./world/mountains";
import { FlowerBeds } from "./world/beds";
import { Undergrowth } from "./world/undergrowth";
import { SUNFLOWERS, Sunflowers, gradeSunflowerField } from "./world/sunflowers";
import { Lanterns } from "./world/lanterns";
import { Torii } from "./world/torii";
import { Rain } from "./world/rain";
import { PATH, pathZ } from "./world/lantern-path";

const canvas = document.getElementById("view") as HTMLCanvasElement;
const errorBox = document.getElementById("error")!;
const titleEl = document.getElementById("title")!;
const controlsEl = document.getElementById("controls")!;
const petalsEl = document.getElementById("petals")!;
const statsEl = document.getElementById("stats")!;
const WIND_HINT = "arrows / WASD (ZQSD) or mouse to steer · space to gust · up / down to rise and dive · M free roam · K map";
const EXPLORE_HINT = "free roam · WASD / ZQSD move · arrows (or click + mouse) look · shift sprint · V fly (space / C up and down) · M back to the wind · K map";

const params = new URLSearchParams(location.search);

function showError(message: string): void {
  errorBox.hidden = false;
  errorBox.textContent = message;
}

async function main(): Promise<void> {
  if (!navigator.gpu) {
    showError("Endless Field needs WebGPU.\nTry a recent Chrome, Edge or Safari.");
    return;
  }
  // GPU timings for the frame counter when the adapter supports timestamp queries.
  // Loading screen: progress follows real work (each texture, model and pipeline).
  const loadingEl = document.getElementById("loading");
  const loadBar = loadingEl?.querySelector<HTMLElement>(".ld-bar span");
  const loadStage = loadingEl?.querySelector<HTMLElement>(".ld-stage");
  let loadTotal = 0;
  let loadDone = 0;
  const stage = (label: string) => {
    if (loadStage) loadStage.textContent = label;
  };
  const track = <T,>(p: Promise<T>): Promise<T> => {
    loadTotal++;
    return p.then((v) => {
      loadDone++;
      if (loadBar) loadBar.style.width = `${Math.round((loadDone / Math.max(loadTotal, 1)) * 100)}%`;
      return v;
    });
  };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const canTime = adapter?.features.has("timestamp-query") ?? false;
  const gpu = await init({ requiredFeatures: canTime ? ["timestamp-query"] : [] });
  const settings: Settings = loadSettings();
  // ?seed=N opens a specific world (shareable links).
  const urlSeed = new URLSearchParams(location.search).get("seed");
  if (urlSeed !== null && Number.isFinite(Number(urlSeed))) settings.seed = Math.max(0, Math.floor(Number(urlSeed)));
  // The world seed must be in place before anything samples the landscape.
  setWorldSeed(settings.seed);
  gradeSunflowerField();
  gpu.onError((e) => {
    console.error(e);
    showError(String((e as Error).message ?? e));
  });

  const globals = new Globals(gpu, GOLDEN_HOUR);
  const renderer = new Renderer(gpu, canvas, globals.uniforms, {
    renderScale: settings.renderScale,
    msaa: true,
  });
  const life = new LifeMap(gpu);
  stage("shaping the hills…");
  const [pebbles, rock, scree, forestFloor, mountains] = await Promise.all([
    track(loadTexture(gpu, "assets/textures/pebbles.jpg", { srgb: true })),
    track(loadTexture(gpu, "assets/textures/rock_diff.jpg", { srgb: true })),
    track(loadTexture(gpu, "assets/textures/scree_diff.jpg", { srgb: true })),
    track(loadTexture(gpu, "assets/textures/forest_floor.jpg", { srgb: true })),
    track(loadMountains(gpu)),
  ]);
  const terrain = new Terrain(gpu, globals.uniforms, life.buffer, pebbles, mountains, rock, scree, forestFloor);
  const grass = new Grass(gpu, globals.uniforms, life.buffer, mountains, settings.grass);
  const flowers = new Flowers(gpu, globals.uniforms);
  const fireflies = new Fireflies(gpu, globals.uniforms, mountains);
  const stream = new PetalStream(gpu, globals.uniforms);
  const camera = new Camera();
  const input = new Input(canvas);
  const audio = new Audio();
  stage("planting trees and flowers…");
  const [trees, beds, water, undergrowth, sunflowers, lanterns, torii] = await Promise.all([
    track(Trees.load(gpu, globals.uniforms)),
    track(FlowerBeds.load(gpu, globals.uniforms, life.buffer)),
    track(Water.load(gpu, globals.uniforms)),
    track(Undergrowth.load(gpu, globals.uniforms, life.buffer)),
    track(Sunflowers.load(gpu, globals.uniforms)),
    track(Lanterns.load(gpu, globals.uniforms)),
    track(Torii.load(gpu, globals.uniforms)),
  ]);
  const player = new Player(0, 30, 0.4);
  const freecam = new FreeCam(canvas);
  let explore = false;
  let hintTimer = 0;
  lanterns.writePositions(globals.lampPos);
  const chainEl = document.createElement("div");
  chainEl.id = "chain";
  (document.getElementById("hud") ?? document.body).append(chainEl);
  let chainTimer = 0;
  const SEASONS = ["summer", "autumn", "winter", "spring"] as const;
  let season = 0;
  try {
    season = Number(localStorage.getItem("endless-field-season")) % 4 || 0;
  } catch {
    // Storage unavailable: start in summer.
  }
  let atmSeason = -1;
  /** Exposure of the current atmosphere (time of day, season, weather). */
  let atmExposure = GOLDEN_HOUR.exposure * 0.8;
  let shownSeason = "";
  let seasonSave = 0;
  let rain = 0;
  let wet = 0;
  let raining = false;
  let weatherTimer = 150 + Math.random() * 150;
  const precipitation = new Rain(gpu, globals.uniforms);
  const seasonEl = document.createElement("div");
  seasonEl.id = "season";
  (document.getElementById("hud") ?? document.body).append(seasonEl);
  let seasonTimer = 0;
  stream.add(player.pos, PALETTES[0]);

  stage("lighting the lanterns…");
  await Promise.all(
    [renderer.sky, terrain.draw, ...grass.draws, ...flowers.draws, stream.draw, ...trees.draws, ...beds.draws, ...undergrowth.draws, ...sunflowers.draws, ...lanterns.draws, ...torii.draws, ...precipitation.draws, fireflies.draw, water.draw].map((d) => track(d.compile(renderer.scene))),
  );


  let playing = false;
  const start = () => {
    if (playing) return;
    playing = true;
    titleEl.classList.add("gone");
    controlsEl.classList.add("show");
    setTimeout(() => controlsEl.classList.remove("show"), 9000);
    audio.start();
  };
  canvas.addEventListener("pointerdown", start);
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" || e.code === "Enter") start();
  });

  let current = settings;
  let onFpsTarget: ((fps: number) => void) | undefined;
  const applySettings = (s: Settings) => {
    // A new seed is a new world: regenerate everything from scratch.
    if (s.seed !== current.seed) {
      location.reload();
      return;
    }
    const fpsChanged = s.fpsTarget !== current.fpsTarget;
    current = s;
    if (fpsChanged) onFpsTarget?.(s.fpsTarget);
    renderer.setRenderScale(s.renderScale);
    renderer.setBloom(s.bloom);
    grass.setQuality(s.grass);
    audio.setMusic(s.music);
  };
  const panel = new SettingsPanel(settings, applySettings);
  const worldMap = new WorldMap(
    document.getElementById("hud") ?? document.body,
    [
      { label: "spawn", x: 0, z: 30, yaw: 0.4 },
      {
        label: "sunflowers",
        x: SUNFLOWERS.x - SUNFLOWERS.dir[0] * (SUNFLOWERS.rx + 6),
        z: SUNFLOWERS.z - SUNFLOWERS.dir[1] * (SUNFLOWERS.rx + 6),
        yaw: Math.atan2(SUNFLOWERS.dir[0], -SUNFLOWERS.dir[1]),
        at: [SUNFLOWERS.x, SUNFLOWERS.z],
      },
      { label: "river", x: 120, z: riverCenter(120) + 30, yaw: Math.PI, at: [300, riverCenter(300) - 30] },
      { label: "lantern path", x: PATH.x0 - 4, z: pathZ(PATH.x0), yaw: Math.PI / 2, at: [200, pathZ(200) + 25] },
      { label: "forest edge", x: 520, z: 120, yaw: Math.PI / 2, at: [470, 260] },
      { label: "autumn forest", x: 900, z: 120, yaw: Math.PI / 2 },
      { label: "deep forest", x: 1650, z: 150, yaw: Math.PI / 2 },
    ],
    (x, z, yaw) => {
      start();
      const y = terrainHeight(x, z);
      if (explore) {
        freecam.pos.splice(0, 3, x, y + 1.65, z);
        freecam.yaw = yaw;
        freecam.pitch = 0;
      } else {
        player.teleport([x, y + 1.7, z], yaw);
        stream.regroup(player.pos);
      }
    },
  );
  applySettings(settings);
  const gpuTimer = canTime ? timer(gpu) : undefined;
  const spans = gpuTimer ? { scene: gpuTimer.span("scene"), post: gpuTimer.span("post") } : undefined;
  let gpuMs = 0;
  gpuTimer?.onResults((r) => {
    gpuMs = gpuMs * 0.85 + ((r.scene ?? 0) + (r.post ?? 0)) * 0.15;
  });
  let cpuMs = 0;
  let night = settings.night ? 1 : 0;
  let mist = 0;
  let canopy = 0;
  let mistBase = 0;
  let mistTimer = 0;
  let atmosphereDirty = true;
  const debug = {
    fixedCamera: null as null | { pos: Vec3; target: Vec3 },
    fixedTime: null as null | number,
    hide: { fireflies: false, grass: false, terrain: false, trees: false, water: false, beds: false, sunflowers: false },
    start,
    player,
    trees,
    camera,
    lanterns,
    sunflowers,
    beds,
    height: terrainHeight,
    mountainZone,
    biome,
    river: { center: riverCenter, water: riverWater, halfWidth: riverHalfWidth },
    grassCounts: () => grass.counts(),
    bloom: (x: number, z: number, r: number) => life.bloom(x, z, r, 0.01),
    /** HDR scene texels at normalized (u, v), for debugging tone and exposure. */
    sceneAt: async (u: number, v: number) => {
      const [w, h] = renderer.scene.size;
      const x = Math.min(w - 1, Math.floor(u * w));
      const y = Math.min(h - 1, Math.floor(v * h));
      const f = await renderer.scene.color.readFloats({ mipLevel: 0, region: { origin: [x, y, 0], size: [1, 1, 1] } });
      return Array.from(f).map((n) => +n.toFixed(3));
    },
  };
  (window as unknown as { __ef: typeof debug }).__ef = debug;

  const time = clock(gpu);
  let fpsAccum = 0;
  let fpsFrames = 0;
  let fps = 0;
  let windAngle = 0.6;
  let lastTrail: [number, number] = [player.pos[0], player.pos[2]];

  // Smoothed simulation step: frame-time spikes do not jerk the camera or the stream.
  let dtSmooth = 1 / 60;
  // Dynamic resolution: render scale follows the GPU budget with hysteresis.
  let dynScale = 1;
  let overBudget = 0;
  let underBudget = 0;
  let lastScaleChange = 0;
  let loop: FrameLoopHandle | undefined;
  const startLoop = (fpsTarget: number) => {
    loop?.stop();
    loop = frameLoop(gpu, tick, fpsTarget > 0 ? { fps: fpsTarget } : undefined);
  };

  /**
   * Resolution changes wait for the start of the next frame: resizing destroys the scene
   * textures, and this frame's (already encoded) passes still reference them.
   */
  let pendingScale: number | null = null;
  function adaptResolution(now: number): void {
    if (!current.autoResolution || !gpuTimer) {
      if (dynScale !== 1) {
        dynScale = 1;
        pendingScale = current.renderScale;
      }
      return;
    }
    const budget = 1000 / (current.fpsTarget > 0 ? current.fpsTarget : 60);
    if (gpuMs > budget * 0.85) overBudget++;
    else overBudget = 0;
    if (gpuMs < budget * 0.55) underBudget++;
    else underBudget = 0;
    if (now - lastScaleChange < 1.0) return;
    let next = dynScale;
    if (overBudget > 20) next = Math.max(0.5, dynScale - 0.1);
    else if (underBudget > 180) next = Math.min(1, dynScale + 0.05);
    if (next !== dynScale) {
      dynScale = next;
      lastScaleChange = now;
      overBudget = underBudget = 0;
      pendingScale = current.renderScale * dynScale;
    }
  }

  let framesShown = 0;
  function tick(frame: Frame): void {
    if (pendingScale !== null) {
      renderer.setRenderScale(pendingScale);
      pendingScale = null;
    }
    // The loading screen fades once the world has drawn a few frames (streaming warmed up).
    if (++framesShown === 20) {
      stage("");
      loadingEl?.classList.add("done");
    }
    const t = debug.fixedTime ?? time.time;
    const rawDt = Math.min(time.deltaTime, 1 / 15);
    dtSmooth += (rawDt - dtSmooth) * 0.2;
    const dt = dtSmooth;
    input.suspended = panel.open || worldMap.open;
    input.update(rawDt);
    if (input.wasPressed("k")) worldMap.toggle();
    worldMap.update(explore ? freecam.pos[0] : player.pos[0], explore ? freecam.pos[2] : player.pos[2], explore ? freecam.yaw : player.yaw);
    const frameStart = performance.now();
    if (input.wasPressed("f")) panel.set({ showStats: !current.showStats });
    if (input.wasPressed("o")) panel.toggle();
    if (input.wasPressed("n")) panel.set({ night: !current.night });
    if (input.wasPressed("m")) {
      explore = !explore;
      if (explore) {
        start();
        freecam.enter(camera);
      } else {
        freecam.exit();
        // The wind picks up where you were exploring.
        player.teleport(freecam.pos, freecam.yaw);
        stream.regroup(player.pos);
      }
      controlsEl.textContent = explore ? EXPLORE_HINT : WIND_HINT;
      controlsEl.classList.add("show");
      clearTimeout(hintTimer);
      hintTimer = window.setTimeout(() => controlsEl.classList.remove("show"), 7000);
    }

    // Seasons: the year turns in ~20 minutes, each bloom nudges it on; or one is held.
    if (input.wasPressed("y")) {
      // Next season: held seasons step to the next one; a turning year jumps ahead.
      if (current.seasonMode === "cycle") season = (Math.floor(season + 0.5) + 1) % 4;
      else panel.set({ seasonMode: SEASONS[(SEASONS.indexOf(current.seasonMode) + 1) % 4] });
    }
    if (current.seasonMode === "cycle") season = (season + dt * (4 / 1200)) % 4;
    else {
      const target = SEASONS.indexOf(current.seasonMode);
      let d = target - season;
      d -= Math.round(d / 4) * 4;
      season = (season + Math.sign(d) * Math.min(Math.abs(d), dt * 0.5) + 4) % 4;
    }
    const seasonName = SEASONS[Math.floor(season + 0.5) % 4];
    if (seasonName !== shownSeason) {
      if (shownSeason) {
        seasonEl.textContent = seasonName;
        seasonEl.classList.add("show");
        clearTimeout(seasonTimer);
        seasonTimer = window.setTimeout(() => seasonEl.classList.remove("show"), 4000);
      }
      shownSeason = seasonName;
    }
    seasonSave += dt;
    if (seasonSave > 5) {
      seasonSave = 0;
      try {
        localStorage.setItem("endless-field-season", String(season));
      } catch {
        // Non-essential.
      }
    }
    if (Math.abs(season - atmSeason) > 0.004) atmosphereDirty = true;

    // Weather: showers of a minute or two every few minutes (or held clear / raining). Rain
    // fades in and out over ~10 s; the ground wets within ~25 s and dries over ~90 s.
    if (input.wasPressed("r")) {
      if (current.weather !== "auto") panel.set({ weather: "auto" });
      raining = !raining;
      weatherTimer = raining ? 90 : 240;
    }
    weatherTimer -= dt;
    if (weatherTimer <= 0) {
      raining = !raining;
      weatherTimer = raining ? 60 + Math.random() * 90 : 150 + Math.random() * 240;
    }
    const rainTarget = current.weather === "rain" ? 1 : current.weather === "clear" ? 0 : raining ? 1 : 0;
    const prevRain = rain;
    rain += Math.sign(rainTarget - rain) * Math.min(Math.abs(rainTarget - rain), dt / 10);
    wet += ((rain > 0.15 ? 1 : 0) - wet) * Math.min(1, dt / (rain > 0.15 ? 25 : 90));
    if (Math.abs(rain - prevRain) > 1e-5) atmosphereDirty = true;
    audio.setRain(rain * (1 - seasonWeightsTs(season)[2]));

    // Day/night eases over ~4 s; the atmosphere, bloom and sound follow.
    const nightTarget = current.night ? 1 : 0;
    if (night !== nightTarget || atmosphereDirty) {
      atmosphereDirty = false;
      night = nightTarget > night ? Math.min(1, night + dt / 4) : Math.max(0, night - dt / 4);
      const k = night * night * (3 - 2 * night);
      atmSeason = season;
      const atm = weatherAtmosphere(mixAtmosphere(GOLDEN_HOUR, NIGHT, k), season, rain);
      globals.setAtmosphere(atm);
      atmExposure = atm.exposure * 0.8;
      renderer.setPost({ bloomStrength: 0.1 + 0.12 * k });
      audio.setNight(k);
    }

    const obstacles = trees
      .near(camera.position[0], camera.position[2], 14)
      .map((tr) => ({ x: tr.x, z: tr.z, r: trees.trunkRadius(tr) + 0.6 }));
    if (explore) {
      freecam.update(dt, obstacles);
      flowers.update(dt, freecam.pos[0], freecam.pos[1], freecam.pos[2], 0);
    } else {
      // Before the first click the wind wanders toward flowers on its own.
      const target = flowers.nearestClosed(player.pos[0], player.pos[2]);
      player.update(dt, input, playing ? null : { toward: target ? [target.x, target.z] : undefined });

      // Slide around trunks instead of passing through them.
      for (const tr of trees.near(player.pos[0], player.pos[2], 12)) {
        const r = trees.trunkRadius(tr) + 0.6;
        const dx = player.pos[0] - tr.x;
        const dz = player.pos[2] - tr.z;
        const d = Math.hypot(dx, dz);
        if (d < r && player.pos[1] < tr.y + 3 * tr.scale) {
          player.pos[0] = tr.x + (dx / (d || 1)) * r;
          player.pos[2] = tr.z + (dz / (d || 1)) * r;
        }
      }

      const touched = flowers.update(dt, player.pos[0], player.pos[1], player.pos[2], 2.4 + Math.min(stream.count, 60) * 0.02 + player.gust);
      for (const f of touched) {
        stream.add([f.x, f.y + f.height, f.z], f.color);
        life.bloom(f.x, f.z, 9 + Math.random() * 4);
        audio.bloom();
        if (current.seasonMode === "cycle") season = (season + 0.02) % 4;
      }
      for (const c of flowers.completedClusters(touched)) {
        life.bloom(c.x, c.z, 34, 7);
        audio.cluster();
      }
      stream.update(dt, player.pos, player.forward, player.gust);
      // Once carrying collected petals, the stream leaves a faint wake of new growth.
      if (stream.count > 1 && player.altitude < 6) {
        if (Math.hypot(player.pos[0] - lastTrail[0], player.pos[2] - lastTrail[1]) > 2.5) {
          lastTrail = [player.pos[0], player.pos[2]];
          life.bloom(player.pos[0], player.pos[2], 2.5 + Math.min(stream.count, 40) * 0.06, 0.8, 0.45);
        }
      }
    }

    if (debug.fixedCamera) {
      camera.position.splice(0, 3, ...debug.fixedCamera.pos);
      camera.target.splice(0, 3, ...debug.fixedCamera.target);
    } else if (explore) {
      freecam.apply(camera);
    } else {
      player.updateCamera(camera, dt, stream.length(player.gust));
    }
    camera.aspect = renderer.aspect;
    camera.update();

    windAngle += Math.sin(t * 0.05) * 0.02 * dt;
    // Mist: dense in the forest and over wet hollows, thicker at night; eased so it rolls in.
    mistTimer -= dt;
    if (mistTimer <= 0) {
      mistTimer = 0.25;
      const eco = ecology(camera.position[0], camera.position[2]);
      const target = Math.min(1.2, eco.forest * 0.9 + Math.max(0, eco.moisture - 0.68) * 1.6) * (1 + night * 0.6);
      mist += (target - mist) * 0.35;
      mistBase += (eco.height - mistBase) * (mistBase === 0 ? 1 : 0.35);
      const deep = Math.min(1, Math.max(0, (camera.position[0] - 1000) / 900));
      const canopyTarget = eco.forest * deep * (1 - Math.min(1, Math.max(0, (camera.position[1] - eco.height - 12) / 20)));
      canopy += (canopyTarget - canopy) * 0.3;
      // Eyes adapt only partly: the deep forest should feel darker (applied with the rest of
      // the exposure, below; a single writer, so weather and canopy never fight).
    }
    // Light shafts toward the sun (only when it is in front of the camera).
    {
      const sx = camera.position[0] + globals.sunDir[0] * 1000;
      const sy = camera.position[1] + globals.sunDir[1] * 1000;
      const sz = camera.position[2] + globals.sunDir[2] * 1000;
      const m = camera.viewProj;
      const cx = m[0] * sx + m[4] * sy + m[8] * sz + m[12];
      const cy = m[1] * sx + m[5] * sy + m[9] * sz + m[13];
      const cw = m[3] * sx + m[7] * sy + m[11] * sz + m[15];
      if (cw > 0) {
        const u = (cx / cw) * 0.5 + 0.5;
        const v = 0.5 - (cy / cw) * 0.5;
        const onScreen = Math.max(0, 1 - Math.max(Math.abs(u - 0.5), Math.abs(v - 0.5)) / 0.9);
        renderer.setShafts([u, v], (0.18 + mist * 0.55) * onScreen * (1 - night));
      } else {
        renderer.setShafts([0.5, 0.5], 0);
      }
    }
    if (explore) freecam.writeTrail(globals.trail);
    else player.writeTrail(globals.trail);
    globals.updateFrame(camera, t, renderer.viewport, {
      night: night * night * (3 - 2 * night),
      mist,
      mistBase,
      canopy,
      season,
      rain,
      wet,
      explore: explore ? 1 : 0,
      playerPos: explore ? freecam.pos : player.pos,
      playerSpeed: explore ? 0 : player.speed,
      gust: explore ? 0 : player.gust,
      windDir: [Math.cos(windAngle), Math.sin(windAngle)],
      windStrength: 0.68 + 0.17 * Math.sin(t * 0.13),
    });
    life.update(dt);
    terrain.update(camera.position[0], camera.position[2]);
    grass.update(camera.position[0], camera.position[2]);
    fireflies.update(camera.position[0], camera.position[2]);
    water.update(camera.position[0], camera.position[2]);
    trees.update(camera.position, camera.frustum, current.drawDistance);
    beds.update(camera.position, camera.frustum, 140 * current.drawDistance);
    undergrowth.update(camera.position, camera.frustum, current.drawDistance);
    sunflowers.update(dt, camera.position, camera.frustum);
    lanterns.update(camera.position);
    const lanternEvent = lanterns.touch(explore ? freecam.pos : player.pos, t);
    if (lanternEvent) {
      audio.lantern(lanternEvent.chain);
      if (lanternEvent.complete) audio.lanternsComplete();
      const p = lanterns.progress;
      chainEl.textContent = lanternEvent.complete
        ? `every lantern, in one breath`
        : lanternEvent.chain > 1
          ? `${lanternEvent.chain} lanterns in a row · ${p.lit} / ${p.total}`
          : `${p.lit} / ${p.total} lanterns`;
      chainEl.classList.add("show");
      clearTimeout(chainTimer);
      chainTimer = window.setTimeout(() => chainEl.classList.remove("show"), lanternEvent.complete ? 6000 : 2500);
    }
    lanterns.animate(dt, globals.lamps, t);
    renderer.setPost({ time: t, exposure: atmExposure * (1 - canopy * 0.45) });
    audio.update(explore ? 0.15 : (player.speed - 7.5) / 13.5, explore ? 2 : player.altitude);

    renderer.render(frame, (pass) => {
      // Rough front-to-back for early depth rejection: blades and props first, ground last.
      if (!debug.hide.grass) grass.encode(pass);
      if (!debug.hide.trees) trees.encode(pass);
      if (!debug.hide.beds) beds.encode(pass);
      if (!debug.hide.sunflowers) sunflowers.encode(pass);
      lanterns.encode(pass);
      torii.encode(pass);
      undergrowth.encode(pass);
      flowers.encode(pass);
      if (!explore) stream.encode(pass);
      if (!debug.hide.water) water.encode(pass);
      if (!debug.hide.terrain) terrain.encode(pass);
      flowers.encodeGlow(pass);
      if (!debug.hide.fireflies) fireflies.encode(pass, night);
      // Transparent, depth-tested but not depth-writing: after everything opaque.
      precipitation.encode(pass, rain);
    }, spans);

    petalsEl.textContent = explore ? (freecam.fly ? "free roam · flying" : "free roam") : stream.count > 1 ? `${stream.count} petals` : "";
    cpuMs = cpuMs * 0.9 + (performance.now() - frameStart) * 0.1;
    fpsAccum += time.deltaTime;
    fpsFrames++;
    if (fpsAccum > 0.5) {
      fps = fpsFrames / fpsAccum;
      fpsAccum = 0;
      fpsFrames = 0;
      statsEl.textContent = current.showStats
        ? `${fps.toFixed(0)} fps  ${(1000 / Math.max(fps, 1)).toFixed(1)} ms\n` +
          `cpu ${cpuMs.toFixed(1)} ms${gpuTimer ? `  gpu ${gpuMs.toFixed(1)} ms` : ""}\n` +
          `${renderer.viewport.join("x")}  ${current.preset}${current.fpsTarget ? `  cap ${current.fpsTarget}` : ""}`
        : "";
    }
    adaptResolution(time.time);
    input.endFrame();
  }
  startLoop(current.fpsTarget);
  onFpsTarget = startLoop;
}

main().catch((e: unknown) => {
  console.error(e);
  showError(String((e as Error)?.message ?? e));
});
