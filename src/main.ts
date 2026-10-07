import { clock, frameLoop, init, timer, type Frame, type FrameLoopHandle } from "vgpu";
import { FILTERS, SettingsPanel, TIMES, loadSettings, type Filter, type Settings } from "./ui/settings";

const FILTER_NAMES: Record<Filter, string> = { none: "natural", painterly: "painterly", watercolor: "watercolor", film: "film", miniature: "miniature", ink: "ink wash", cozy: "cozy" };
import { WorldMap } from "./ui/map";
import { Camera, type Vec3 } from "./engine/camera";
import { GOLDEN_HOUR, Globals, dayAtmosphere, seasonWeights as seasonWeightsTs, weatherAtmosphere } from "./engine/globals";
import { Renderer } from "./engine/renderer";
import { loadTexture, setTextureMaxSize } from "./engine/textures";
import { Audio } from "./game/audio";
import { Input } from "./game/input";
import { Motes } from "./game/motes";
import { Birds } from "./world/birds";
import { Secrets } from "./world/secrets";
import { placeLandmarks } from "./world/layout";
import { Nest } from "./world/nest";
import { RiverRace } from "./world/river-race";
import { SkyLanterns } from "./world/sky-lanterns";
import { RiverFish } from "./world/fish";
import { PlayerBird } from "./game/player-bird";
import { Insects } from "./world/insects";
import { THERMAL_TOP, Thermals } from "./world/thermals";
import { WindTrail } from "./game/wind-trail";
import { Player } from "./game/player";
import { FreeCam } from "./game/freecam";
import { Flowers } from "./world/flowers";
import { Grass } from "./world/grass";
import { LifeMap } from "./world/life";
import { Terrain } from "./world/terrain";
import { Fireflies } from "./world/fireflies";
import { Water } from "./world/water";
import { setWorldSeed, mountainZone, riverCenter, riverInfo, riverHalfWidth, riverWater, terrainHeightM as terrainHeight } from "./world/height";
import { biome } from "./world/biome";
import { ecology } from "./world/ecology";
import { Trees } from "./world/trees";
import { loadMountains } from "./world/mountains";
import { FlowerBeds } from "./world/beds";
import { Undergrowth } from "./world/undergrowth";
import { SUNFLOWERS, Sunflowers, gradeSunflowerField, sunflowerField } from "./world/sunflowers";
import { Lanterns } from "./world/lanterns";
import { Torii, toriiGates } from "./world/torii";
import { Discoveries } from "./game/discoveries";
import { Kind } from "./game/motes";
import { Rain } from "./world/rain";
import { PATH, pathNear, pathZ } from "./world/lantern-path";

const canvas = document.getElementById("view") as HTMLCanvasElement;
const errorBox = document.getElementById("error")!;
const titleEl = document.getElementById("title")!;
const controlsEl = document.getElementById("controls")!;
const petalsEl = document.getElementById("petals")!;
const statsEl = document.getElementById("stats")!;
const WIND_HINT = "<kbd>WASD</kbd> steer · <kbd>Space</kbd> gust · <kbd>↑</kbd><kbd>↓</kbd> rise / dive · <kbd>E</kbd> loop · <kbd>←</kbd><kbd>←</kbd> roll · <kbd>M</kbd> free roam · <kbd>K</kbd> map · <kbd>O</kbd> menu";
const EXPLORE_HINT = "<kbd>WASD</kbd> move · <kbd>←</kbd><kbd>→</kbd> look · <kbd>Shift</kbd> sprint · <kbd>V</kbd> fly · <kbd>Space</kbd><kbd>C</kbd> up / down · <kbd>M</kbd> wind";
const TOUCH = matchMedia("(pointer: coarse)").matches;
const TOUCH_HINT = "Drag to steer · hold <kbd>◎</kbd> to gust, double-tap to loop · tilt steering in Menu → System";
controlsEl.innerHTML = TOUCH ? TOUCH_HINT : WIND_HINT;
if (TOUCH) {
  document.body.classList.add("touch");
  const hint = document.getElementById("start-hint");
  if (hint) hint.textContent = "tap to begin";
}

const params = new URLSearchParams(location.search);

/** A small, non-blocking problem report (pipelines this device could not build, etc.). */
const notices: string[] = [];
function notice(message: string): void {
  if (notices.includes(message) || notices.length > 6) return;
  notices.push(message);
  let el = document.getElementById("notice");
  if (!el) {
    el = document.createElement("div");
    el.id = "notice";
    el.addEventListener("click", () => el?.remove());
    document.body.append(el);
  }
  el.textContent = `Some effects are off on this device (tap to hide):\n${notices.join("\n")}`;
}

function showError(message: string): void {
  // Shown above everything, the loading screen included (a failure while loading must not
  // look like loading forever).
  if (errorBox.parentElement !== document.body) document.body.append(errorBox);
  errorBox.hidden = false;
  errorBox.textContent = message;
}
// Stray errors are reported, not fatal (loading failures are caught where they happen).
window.addEventListener("error", (e) => notice(`Error: ${e.message}`));
window.addEventListener("unhandledrejection", (e) => notice(`Error: ${String((e.reason as Error)?.message ?? e.reason)}`));

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
  const pending = new Set<string>();
  let loadSeq = 0;
  const track = <T,>(p: Promise<T>, label = `item ${loadSeq + 1}`): Promise<T> => {
    loadTotal++;
    const id = `${label}#${loadSeq++}`;
    pending.add(id);
    return p.then((v) => {
      pending.delete(id);
      loadDone++;
      if (loadBar) loadBar.style.width = `${Math.round((loadDone / Math.max(loadTotal, 1)) * 100)}%`;
      return v;
    });
  };
  // If loading stalls, say what is still pending (on phones, a pipeline can take very long or
  // never come back).
  const watchdog = window.setTimeout(() => {
    if (!loadStage || loadingEl?.classList.contains("done")) return;
    const left = [...pending].map((p) => p.split("#")[0]);
    loadStage.textContent = `still waiting on ${left.length} of ${loadTotal}: ${left.slice(0, 6).join(", ")}${left.length > 6 ? "…" : ""}`;
  }, 30000);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const canTime = adapter?.features.has("timestamp-query") ?? false;
  const gpu = await init({ requiredFeatures: canTime ? ["timestamp-query"] : [] });
  const settings: Settings = loadSettings();
  // Phones: smaller textures and a sparser sunflower field, to fit their graphics memory.
  if (TOUCH) {
    setTextureMaxSize(1024);
    Sunflowers.thin = 2;
  }
  // ?seed=N opens a specific world (shareable links).
  const urlSeed = new URLSearchParams(location.search).get("seed");
  if (urlSeed !== null && Number.isFinite(Number(urlSeed))) settings.seed = Math.max(0, Math.floor(Number(urlSeed)));
  // The world seed must be in place before anything samples the landscape.
  setWorldSeed(settings.seed);
  // Landmarks (sunflower field, lantern path, river race) placed for this world.
  placeLandmarks(settings.seed);
  gradeSunflowerField();
  void gpu.device.gpu.lost.then((info) => {
    if (info.reason !== "destroyed") showError(`The GPU stopped (${info.message || info.reason}).\nThis device may not have enough graphics memory: try Menu → System → Quality → Low.`);
  });
  gpu.onError((e) => {
    console.error(e);
    // Not fatal by itself: report it in a small banner and keep the world running.
    notice(String((e as Error).message ?? e).slice(0, 220));
  });

  const globals = new Globals(gpu, GOLDEN_HOUR);
  const camera = new Camera();
  const renderer = new Renderer(gpu, canvas, globals.uniforms, {
    renderScale: settings.renderScale,
    msaa: true,
  });
  const life = new LifeMap(gpu);
  stage("shaping the hills…");
  const [pebbles, rock, scree, forestFloor, mountains] = await Promise.all([
    track(loadTexture(gpu, "assets/textures/pebbles.jpg", { srgb: true }), "pebbles"),
    track(loadTexture(gpu, "assets/textures/rock_diff.jpg", { srgb: true }), "rock_diff"),
    track(loadTexture(gpu, "assets/textures/scree_diff.jpg", { srgb: true }), "scree_diff"),
    track(loadTexture(gpu, "assets/textures/forest_floor.jpg", { srgb: true }), "forest_floor"),
    track(loadMountains(gpu), "mountains"),
  ]);
  const terrain = new Terrain(gpu, globals.uniforms, life.buffer, pebbles, mountains, rock, scree, forestFloor);
  const grass = new Grass(gpu, globals.uniforms, life.buffer, mountains, settings.grass);
  const flowers = new Flowers(gpu, globals.uniforms);
  const fireflies = new Fireflies(gpu, globals.uniforms, mountains);
  const motes = new Motes(gpu, globals.uniforms);
  const windTrail = new WindTrail(gpu, globals.uniforms);
  const skyLanterns = new SkyLanterns(gpu, globals.uniforms);
  const insects = new Insects(gpu, globals.uniforms);
  const thermals = new Thermals(gpu, globals.uniforms);
  // Catching insects on the wing: each catch sparks, plucks a note and gives a little speed;
  // catches close together climb the scale.
  let combo = 0;
  let lastCatch = -10;
  let skimTimer = 0;
  let wasPerched = false;
  // Weather moments and sound: a rainbow after a daytime shower, frogs at night, rain plinks.
  let rainbowT = 0;
  let rainPeak = 0;
  let frogTimer = 2;
  let plinkTimer = 0;
  let lastDip = 0;
  let skimMotifAt = -20;
  let restingHome = false;
  // Threading the needle: through a torii, or between two trunks under the canopy.
  let threadChain = 0;
  let lastThread = -10;
  const threadSeen = new Map<string, number>();
  const prevWind: Vec3 = [0, 0, 0];
  const thread = (at: Vec3, key: string, now: number) => {
    if ((threadSeen.get(key) ?? -10) > now - 2) return;
    threadSeen.set(key, now);
    threadChain = now - lastThread < 6 ? threadChain + 1 : 1;
    lastThread = now;
    audio.lantern(threadChain);
    player.boost(1.6);
    motes.puff(at, [1, 0.95, 0.75]);
  };
  const crosses = (a: Vec3, b: Vec3, p: [number, number], q: [number, number]): boolean => {
    // Does the wind's step a->b cross the segment p-q (seen from above)?
    const cross = (ox: number, oz: number, ax: number, az: number, bx: number, bz: number) => (ax - ox) * (bz - oz) - (az - oz) * (bx - ox);
    const d1 = cross(p[0], p[1], q[0], q[1], a[0], a[2]);
    const d2 = cross(p[0], p[1], q[0], q[1], b[0], b[2]);
    const d3 = cross(a[0], a[2], b[0], b[2], p[0], p[1]);
    const d4 = cross(a[0], a[2], b[0], b[2], q[0], q[1]);
    return d1 * d2 < 0 && d3 * d4 < 0;
  };
  const input = new Input(canvas);
  const audio = new Audio();
  stage("planting trees and flowers…");
  const [trees, beds, water, undergrowth, sunflowers, lanterns, torii, birds, secrets, race, fish, swallow] = await Promise.all([
    track(Trees.load(gpu, globals.uniforms), "trees"),
    track(FlowerBeds.load(gpu, globals.uniforms, life.buffer), "flowerbeds"),
    track(Water.load(gpu, globals.uniforms), "water"),
    track(Undergrowth.load(gpu, globals.uniforms, life.buffer), "undergrowth"),
    track(Sunflowers.load(gpu, globals.uniforms), "sunflowers"),
    track(Lanterns.load(gpu, globals.uniforms), "lanterns"),
    track(Torii.load(gpu, globals.uniforms), "torii"),
    track(Birds.load(gpu, globals.uniforms), "birds"),
    track(Secrets.load(gpu, globals.uniforms, settings.seed, [Math.cos(0.6), Math.sin(0.6)]), "secrets"),
    track(RiverRace.load(gpu, globals.uniforms), "riverrace"),
    track(RiverFish.load(gpu, globals.uniforms), "fish"),
    track(PlayerBird.load(gpu, globals.uniforms), "swallow"),
  ]);
  const nest = new Nest(gpu, globals.uniforms);
  /** Stereo position (-1 left .. 1 right) and distance of a point, from the camera. */
  const heard = (at: readonly number[]): [number, number] => {
    const fx = camera.target[0] - camera.position[0];
    const fz = camera.target[2] - camera.position[2];
    const dx = at[0] - camera.position[0];
    const dz = at[2] - camera.position[2];
    const d = Math.hypot(dx, dz) || 1;
    return [(dx * -fz + dz * fx) / (d * (Math.hypot(fx, fz) || 1)), d];
  };
  birds.onTakeoff = (at, n) => {
    const [pan, d] = heard(at);
    audio.birds(n, pan, d);
    if (d < 25) discoveries.find("sparrows");
  };
  birds.onChirp = (at) => audio.chirp(...heard(at));
  insects.onCatch = (at) => {
    const now = performance.now() / 1000;
    combo = now - lastCatch < 1.6 ? combo + 1 : 1;
    lastCatch = now;
    audio.catchInsect(combo);
    motes.puff(at, [1, 0.85, 0.45]);
    player.boost(0.5 + Math.min(combo, 8) * 0.12);
    nest.feed();
  };
  fish.onSplash = (at, size) => {
    motes.splash(at, size);
    const [pan, d] = heard(at);
    audio.splash(size, pan, d);
  };
  // The kingfisher's course on the river: follow it through every gate.
  race.onStart = () => audio.kingfisher(...heard([race.start[0], 0, race.start[1]]));
  race.onGate = (i) => audio.lantern(i + 1);
  race.onFail = () => audio.raceLost();
  race.onComplete = () => {
    audio.lanternsComplete();
    audio.discovery();
    // The banks along the course burst into flower and a lantern rises from every ring.
    for (const g of race.gates) life.bloom(g.base[0], g.base[2], 26, 6);
    skyLanterns.release(race.gates.map((g) => [g.centre[0], g.centre[1] - 1.4, g.centre[2]] as Vec3));
  };
  // Startled birds lead the way to somewhere the wind has not found yet, preferring places
  // roughly in the direction they flee, not too far.
  birds.guide = (from, away) => {
    const places: [number, number][] = secrets.hidden.map((s) => [s.x, s.z] as [number, number]);
    if (!race.done) places.push(race.start);
    if (!discoveries.has("lantern")) places.push([PATH.x0 + 6, pathZ(PATH.x0 + 6)]);
    if (!discoveries.has("pollen")) places.push([SUNFLOWERS.x - SUNFLOWERS.dir[0] * SUNFLOWERS.rx, SUNFLOWERS.z - SUNFLOWERS.dir[1] * SUNFLOWERS.rx]);
    if (!discoveries.has("spray")) places.push([from[0], riverCenter(from[0])]);
    if (!discoveries.has("leaves")) places.push([480, 250]);
    let best: [number, number] | null = null;
    let bestScore = -Infinity;
    for (const [x, z] of places) {
      const dx = x - from[0];
      const dz = z - from[2];
      const d = Math.hypot(dx, dz);
      if (d < 40 || d > 450) continue;
      const align = (dx * Math.sin(away) + dz * Math.cos(away)) / d;
      const score = align * 1.2 - d / 300;
      if (score > bestScore) {
        bestScore = score;
        best = [x, z];
      }
    }
    return bestScore > -1.2 ? best : null;
  };
  // Secrets: woken by the wind, each answers with a jingle and the meadow around it bursts
  // into flower; bells ring and pinwheels whirr where they are.
  secrets.onWake = (s) => {
    audio.discovery();
    life.bloom(s.x, s.z, s.kind === "pinwheels" ? 32 : 24, 6);
  };
  secrets.onBell = (at, strength, bell) => {
    const [pan, d] = heard(at);
    audio.windBell(bell, strength, pan, d);
  };
  let whirr = 0;
  secrets.onWhirr = (at, level) => {
    const [, d] = heard(at);
    whirr = Math.max(whirr, level * Math.min(1, 8 / d));
  };
  const player = new Player(0, 30, 0.4);
  const freecam = new FreeCam(canvas);
  let explore = false;
  let hintTimer = 0;
  lanterns.writePositions(globals.lampPos);
  const chainEl = document.createElement("div");
  chainEl.id = "chain";
  (document.getElementById("hud") ?? document.body).append(chainEl);
  // Discoveries: a little jingle the first time each living thing is found.
  const discoveries = new Discoveries();
  discoveries.onFind = () => audio.discovery();
  motes.onPickup = (kind) => {
    const id = { [Kind.Leaf]: "leaves", [Kind.Pollen]: "pollen", [Kind.Spray]: "spray", [Kind.Firefly]: "fireflies", [Kind.Snow]: "snow" }[kind as number];
    if (id) discoveries.find(id);
  };
  const gates = toriiGates();
  /** Named places: a banner (and a soft sting) the first time the wind enters each one. */
  const REGIONS: { name: string; inside: (x: number, z: number) => boolean }[] = [
    { name: "Lantern Path", inside: (x, z) => pathNear(x, z, 10) },
    { name: "Sunflower Field", inside: (x, z) => sunflowerField(x, z) > 0.2 },
    { name: "The River", inside: (x, z) => { const [d, , hw] = riverInfo(x, z); return d < hw + 8; } },
    { name: "Eastern Wood", inside: (x, z) => ecology(x, z).forest > 0.55 },
  ];
  const regionsSeen = new Set<string>();
  let regionCheck = 0;
  /** Entering a place for the first time: a soft chord, no text. */
  const showRegion = () => audio.region();
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
  // The first shower comes within a minute or so; then every few minutes.
  let weatherTimer = 40 + Math.random() * 40;
  const precipitation = new Rain(gpu, globals.uniforms);
  const seasonEl = document.createElement("div");
  seasonEl.id = "season";
  (document.getElementById("hud") ?? document.body).append(seasonEl);
  let seasonTimer = 0;

  stage("lighting the lanterns…");
  await Promise.all(
    [renderer.sky, terrain.draw, ...grass.draws, ...flowers.draws, motes.draw, windTrail.draw, insects.draw, thermals.draw, skyLanterns.draw, ...birds.draws, ...race.draws, fish.draw, nest.draw, swallow.draw, swallow.haloDraw, ...secrets.draws, secrets.glintDraw, ...trees.draws, ...beds.draws, ...undergrowth.draws, ...sunflowers.draws, ...lanterns.draws, ...torii.draws, ...precipitation.draws, fireflies.draw, water.draw].map((d) =>
      track(
        d.compile(renderer.scene).catch((e: unknown) => {
          // Keep going without it; say which one (so it can be fixed for this device).
          renderer.broken.add(d);
          notice(`couldn't build ${(d as { label?: string }).label ?? "a shader"}: ${String((e as Error)?.message ?? e).slice(0, 160)}`);
        }),
        `shader ${(d as { label?: string }).label ?? "?"}`,
      ),
    ),
  );


  let playing = false;
  const start = () => {
    if (playing) return;
    playing = true;
    titleEl.classList.add("gone");
    controlsEl.classList.add("show");
    setTimeout(() => controlsEl.classList.remove("show"), 9000);
    audio.start();
    if (current.steering === "tilt") void enableTilt();
  };
  // Phones: a gust button (hold) and a map button; tilt steering if chosen.
  if (TOUCH) {
    const hud = document.getElementById("hud") ?? document.body;
    const gustBtn = document.createElement("button");
    gustBtn.id = "gust-btn";
    gustBtn.textContent = "◎";
    gustBtn.setAttribute("aria-label", "Gust");
    const hold = (on: boolean) => (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
      input.touchGust = on;
      gustBtn.classList.toggle("on", on);
    };
    // Double-tap the gust button: a vertical loop.
    let lastTap = 0;
    gustBtn.addEventListener("pointerdown", (e) => {
      const now = performance.now();
      if (now - lastTap < 320) player.startLoop();
      lastTap = now;
      hold(true)(e);
    });
    gustBtn.addEventListener("pointerup", hold(false));
    gustBtn.addEventListener("pointercancel", hold(false));
    gustBtn.addEventListener("pointerleave", hold(false));
    const mapBtn = document.createElement("button");
    mapBtn.id = "map-btn";
    mapBtn.textContent = "MAP";
    mapBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
    mapBtn.addEventListener("click", () => worldMap.toggle());
    hud.append(gustBtn, mapBtn);
  }
  /** Tilt: the phone's lean steers (relative to how it was held when tilt began). */
  let tiltRef: [number, number] | null = null;
  let tiltOn = false;
  const onOrient = (e: DeviceOrientationEvent) => {
    if (e.beta === null || e.gamma === null) return;
    // In landscape the phone's axes swap.
    const angle = (screen.orientation?.angle ?? (window as unknown as { orientation?: number }).orientation ?? 0) as number;
    let side = e.gamma;
    let fwd = e.beta;
    if (angle === 90) [side, fwd] = [e.beta, -e.gamma];
    else if (angle === 270 || angle === -90) [side, fwd] = [-e.beta, e.gamma];
    if (!tiltRef) tiltRef = [side, fwd];
    const clamp = (v: number) => Math.max(-1, Math.min(1, v));
    input.tilt = current.steering === "tilt" ? [clamp((side - tiltRef[0]) / 22), clamp((fwd - tiltRef[1]) / 22)] : null;
  };
  async function enableTilt(): Promise<void> {
    if (tiltOn) return;
    // iOS asks permission, and only from a tap.
    const D = DeviceOrientationEvent as unknown as { requestPermission?: () => Promise<string> };
    try {
      if (D.requestPermission && (await D.requestPermission()) !== "granted") return;
    } catch {
      return;
    }
    tiltOn = true;
    tiltRef = null;
    window.addEventListener("deviceorientation", onOrient);
  }
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
    renderer.setStyle(Math.max(0, FILTERS.indexOf(s.filter)));
    const sub = document.querySelector("#title .sub");
    if (sub) sub.textContent = s.avatar === "swallow" ? "a swallow on the wind" : "be the wind";
    if (s.steering === "tilt") {
      tiltRef = null;
      void enableTilt();
    } else input.tilt = null;
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
      // The kingfisher's course: its first ring (start just upstream of it, facing along).
      { label: "river rings", x: race.gates[0].centre[0] - 14, z: race.gates[0].centre[2], yaw: Math.PI / 2, at: [race.gates[0].centre[0], race.gates[0].centre[2]] },
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
        motes.reset();
        windTrail.reset();
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
  let night = 0;
  let underwater = 0;
  let waterY = -1000;
  /** Time of day in [0, 1): 0 dawn, ~0.2 midday, ~0.5 golden hour, ~0.62 dusk, ~0.8 night. */
  let dayPhase = 0.46;
  /** A requested jump ahead (N): the day fast-forwards to this phase. */
  let dayTarget: number | null = null;
  const DAY_LENGTH = 600;
  const TIME_PHASE: Record<string, number> = { dawn: 0.02, day: 0.2, golden: 0.48, night: 0.8 };
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
    renderer,
    nest,
    rainbowNow: () => {
      rainbowT = 90;
    },
    motes,
    birds,
    thermals,
    insects,
    swallow,
    get panel() {
      return panel;
    },
    fish,
    secrets,
    race,
    skyLanterns,
    torii,
    setDay: (p: number) => {
      dayPhase = p;
      dayTarget = null;
    },
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
  // Pause: the world holds still (time, wind, creatures, sound); the view can still be admired.
  let paused = false;
  let simTime = 0;
  const pauseEl = document.createElement("div");
  pauseEl.id = "pause";
  pauseEl.hidden = true;
  pauseEl.innerHTML = `<div class="pz-box"><div class="pz-tag">Paused</div>
    <button data-a="resume">Resume</button><button data-a="menu">Settings</button>
    <div class="pz-hint"><kbd>P</kbd> resume</div></div>`;
  (document.getElementById("hud") ?? document.body).append(pauseEl);
  const setPaused = (on: boolean) => {
    if (on === paused) return;
    paused = on;
    pauseEl.hidden = !on;
    audio.setPaused(on);
    if (!on) panel.toggle(false);
  };
  pauseEl.addEventListener("pointerdown", (e) => e.stopPropagation());
  pauseEl.addEventListener("click", (e) => {
    const a = (e.target as HTMLElement).dataset.a;
    if (a === "resume") setPaused(false);
    if (a === "menu") panel.toggle(true);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && playing) setPaused(true);
  });
  const pauseBtn = document.createElement("button");
  pauseBtn.id = "pause-btn";
  pauseBtn.title = "Pause (P)";
  pauseBtn.textContent = "❚❚";
  pauseBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  pauseBtn.addEventListener("click", () => setPaused(!paused));
  (document.getElementById("hud") ?? document.body).append(pauseBtn);
  function tick(frame: Frame): void {
    if (pendingScale !== null) {
      renderer.setRenderScale(pendingScale);
      pendingScale = null;
    }
    // The loading screen fades once the world has drawn a few frames (streaming warmed up).
    if (++framesShown === 20) {
      stage("");
      loadingEl?.classList.add("done");
      clearTimeout(watchdog);
    }
    if (input.wasPressed("p") && playing) setPaused(!paused);
    const rawDt = paused ? 0 : Math.min(time.deltaTime, 1 / 15);
    simTime += rawDt;
    const t = debug.fixedTime ?? simTime;
    if (!paused) dtSmooth += (rawDt - dtSmooth) * 0.2;
    const dt = paused ? 0 : dtSmooth;
    input.suspended = panel.open || worldMap.open || paused;
    input.update(rawDt);
    if (input.wasPressed("k")) worldMap.toggle();
    if (worldMap.open) worldMap.setFlocks(birds.feedingSpots().map((h) => [h[0], h[2]] as [number, number]));
    worldMap.update(explore ? freecam.pos[0] : player.pos[0], explore ? freecam.pos[2] : player.pos[2], explore ? freecam.yaw : player.yaw);
    const frameStart = performance.now();
    if (input.wasPressed("f")) panel.set({ showStats: !current.showStats });
    if (input.wasPressed("o")) panel.toggle();
    if (input.wasPressed("n")) {
      if (current.time === "cycle") {
        // Skip ahead to the next part of the day.
        const next = [0.02, 0.2, 0.48, 0.8].find((p) => p > dayPhase + 0.01) ?? 1.02;
        dayTarget = next;
      } else panel.set({ time: TIMES[(TIMES.indexOf(current.time) + 1) % TIMES.length] });
    }
    if (input.wasPressed("m")) {
      explore = !explore;
      if (explore) {
        start();
        freecam.enter(camera);
      } else {
        freecam.exit();
        // The wind picks up where you were exploring.
        player.teleport(freecam.pos, freecam.yaw);
        motes.reset();
        windTrail.reset();
      }
      controlsEl.innerHTML = explore ? EXPLORE_HINT : TOUCH ? TOUCH_HINT : WIND_HINT;
      controlsEl.classList.add("show");
      clearTimeout(hintTimer);
      hintTimer = window.setTimeout(() => controlsEl.classList.remove("show"), 7000);
    }

    // A vertical loop.
    if (input.wasPressed("e") && !explore && playing) player.startLoop();
    // Barrel rolls: double-tap left or right.
    // ...and a roll near a flock invites it to fly along (a murmuration).
    const rollL = input.wasPressed("roll-left");
    const rollR = input.wasPressed("roll-right");
    if ((rollL || rollR) && !explore && playing) {
      player.startRoll(rollL ? -1 : 1);
      if (birds.invite(player.pos, 30) > 0) audio.birds(6, 0, 6);
    }
    // Let go of everything the wind is carrying.
    if (input.wasPressed("x") && !explore) motes.release();
    if (input.wasPressed("l")) {
      const next = FILTERS[(FILTERS.indexOf(current.filter) + 1) % FILTERS.length];
      panel.set({ filter: next });
      seasonEl.textContent = FILTER_NAMES[next];
      seasonEl.classList.add("show");
      clearTimeout(seasonTimer);
      seasonTimer = window.setTimeout(() => seasonEl.classList.remove("show"), 2500);
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
      weatherTimer = raining ? 60 + Math.random() * 60 : 120 + Math.random() * 120;
    }
    const rainTarget = current.weather === "rain" ? 1 : current.weather === "clear" ? 0 : raining ? 1 : 0;
    const prevRain = rain;
    rain += Math.sign(rainTarget - rain) * Math.min(Math.abs(rainTarget - rain), dt / 10);
    wet += ((rain > 0.15 ? 1 : 0) - wet) * Math.min(1, dt / (rain > 0.15 ? 25 : 90));
    if (Math.abs(rain - prevRain) > 1e-5) atmosphereDirty = true;
    audio.setRain(rain * (1 - seasonWeightsTs(season)[2]));
    // A shower that clears by day leaves a rainbow for a minute and a half.
    rainPeak = Math.max(rainPeak * Math.exp(-dt / 120), rain);
    if (rainPeak > 0.6 && rain < 0.12 && night < 0.3 && rainbowT <= 0) {
      rainbowT = 90;
      rainPeak = 0;
      audio.rainbow();
    }
    rainbowT = Math.max(0, rainbowT - dt);

    // The day turns in ~10 minutes (or holds one time); a skip fast-forwards smoothly.
    {
      const held = current.time === "cycle" ? null : TIME_PHASE[current.time];
      if (held !== null && held !== undefined) {
        let d = held - dayPhase;
        d -= Math.floor(d);
        dayTarget = d < 1e-4 ? null : dayPhase + d;
      }
      if (restingHome && current.time === "cycle") dayTarget = dayPhase > 0.5 ? 1.03 : dayTarget;
      if (dayTarget !== null) {
        const step = Math.min(dayTarget - dayPhase, dt * (restingHome ? 0.02 : 0.06));
        dayPhase += step;
        if (dayTarget - dayPhase < 1e-4) dayTarget = null;
      } else if (current.time === "cycle") dayPhase += dt / DAY_LENGTH;
      if (dayPhase >= 1) {
        dayPhase -= 1;
        if (dayTarget !== null) dayTarget -= 1;
      }
      const day = dayAtmosphere(dayPhase);
      night = day.night;
      atmSeason = season;
      atmosphereDirty = false;
      const atm = weatherAtmosphere(day.atm, season, rain, day.night);
      globals.setAtmosphere(atm);
      atmExposure = atm.exposure * 0.8;
      renderer.setPost({ bloomStrength: 0.1 + 0.12 * night });
      audio.setNight(night);
    }

    const obstacles = trees
      .near(camera.position[0], camera.position[2], 14)
      .map((tr) => ({ x: tr.x, z: tr.z, r: trees.trunkRadius(tr) + 0.6 }));
    if (explore) {
      freecam.update(dt, obstacles);
      // Free roam lifts nothing, but splashes and falling things still play out.
      motes.update(dt, freecam.pos, [0, 0, 1], 0, 0, { altitude: 99, forest: 0, river: 0, sunflowers: 0, season: [1, 0, 0, 0], night, rain }, [Math.cos(windAngle), Math.sin(windAngle)]);
      flowers.update(dt, freecam.pos[0], freecam.pos[1], freecam.pos[2], 0);
    } else {
      // Before the first click the wind wanders toward flowers on its own.
      const target = flowers.nearestClosed(player.pos[0], player.pos[2]);
      player.update(dt, input, playing ? null : { toward: target ? [target.x, target.z] : undefined });
      input.poked = false;

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

      const touched = flowers.update(dt, player.pos[0], player.pos[1], player.pos[2], 2.4 + player.gust);
      if (touched.length) discoveries.find("flowers");
      for (const f of touched) {
        motes.puff([f.x, f.y + f.height, f.z], f.color);
        motes.petals([f.x, f.y + f.height, f.z], f.color, 3);
        life.bloom(f.x, f.z, 9 + Math.random() * 4);
        audio.bloom();
        if (current.seasonMode === "cycle") season = (season + 0.02) % 4;
      }
      for (const c of flowers.completedClusters(touched)) {
        life.bloom(c.x, c.z, 34, 7);
        // A whole bed opened: its petals swirl up around the wind with a rising chime.
        motes.swirl(player.pos, touched.length ? touched.map((f) => f.color) : [[1, 0.7, 0.8]], 40, player.forward.map((v) => v * player.speed) as Vec3);
        audio.petalBank();
      }
      const eco = ecology(player.pos[0], player.pos[2]);
      const [riverDist, , riverHw] = riverInfo(player.pos[0], player.pos[2]);
      motes.update(dt, player.pos, player.forward, player.speed, player.gust, {
        altitude: player.altitude,
        forest: Math.min(1, eco.forest * 1.4),
        river: 1 - Math.min(1, Math.max(0, (riverDist - riverHw) / 4)),
        sunflowers: sunflowerField(player.pos[0], player.pos[2]) > 0.3 ? 1 : 0,
        season: seasonWeightsTs(season),
        night,
        rain,
      }, [Math.cos(windAngle), Math.sin(windAngle)]);
      windTrail.update(dt, player.pos, player.speed, player.gust);
      // Thermals by day: rising air the wind can circle up in.
      thermals.update(t, player.pos, player.yaw, (1 - night) * (1 - rain));
      player.thermal = thermals.lift(player.pos, (1 - night) * (1 - rain));
      birds.soarAt(thermals.columns);
      // Flying into a thermal: carried up it, to music.
      if (!player.riding && !player.looping && playing) {
        const col = thermals.at(player.pos, (1 - night) * (1 - rain));
        if (col) {
          player.ride(col, col.ground + THERMAL_TOP);
          audio.thermal();
        }
      }
      const [wd, wy, whw] = riverInfo(player.pos[0], player.pos[2]);
      // Pause and look: left alone for a while, the swallow finds a perch and settles there.
      if (playing && current.avatar === "swallow" && input.idle > 6 && !player.perched && !player.landing && !player.looping && !player.riding) {
        let perch: Vec3 | null = null;
        let best = 35;
        // At dusk and night, home: the nest under the torii beam, if it is within reach.
        if (night > 0.25 && Math.hypot(nest.pos[0] - player.pos[0], nest.pos[2] - player.pos[2]) < 90) {
          perch = nest.perch;
          best = 0;
        }
        for (const lp of lanterns.positions) {
          const d = Math.hypot(lp[0] - player.pos[0], lp[2] - player.pos[2]);
          if (d < best) {
            best = d;
            perch = [lp[0], lp[1] + 1.52, lp[2]];
          }
        }
        if (!perch) {
          // A tall grass stem a little ahead.
          const ax = player.pos[0] + Math.sin(player.yaw) * 9;
          const az = player.pos[2] - Math.cos(player.yaw) * 9;
          perch = [ax, terrainHeight(ax, az) + 0.95, az];
        }
        player.landAt(perch);
      }
      if (player.perched && !wasPerched) audio.rest();
      // Frogs along the river at night, from somewhere along its banks near you.
      frogTimer -= dt;
      if (frogTimer <= 0) {
        frogTimer = 0.8 + Math.random() * 2.2;
        const [fd] = riverInfo(player.pos[0], player.pos[2]);
        if (night > 0.4 && fd < 80) {
          const fx = player.pos[0] + (Math.random() - 0.5) * 60;
          audio.frog(...heard([fx, 0, riverCenter(fx) + (Math.random() - 0.5) * 8]));
        }
      }
      // Rain on the water plinks; under trees it patters duller.
      plinkTimer -= dt;
      if (rain > 0.1 && plinkTimer <= 0) {
        plinkTimer = 0.04 + Math.random() * 0.12;
        const [pd, , phw] = riverInfo(player.pos[0], player.pos[2]);
        if (pd < phw + 15) audio.plink(rain);
      }
      audio.setRainShelter(Math.min(1, trees.near(player.pos[0], player.pos[2], 7).length / 3));
      // Starting to slice the water: a little rising piano figure (not too often).
      if (swallow.dip > 0.5 && lastDip <= 0.5 && t - skimMotifAt > 12) {
        skimMotifAt = t;
        audio.skimMotif();
      }
      lastDip = swallow.dip;
      nest.update(player.pos);
      // Resting in the nest at night: the night passes quickly (to dawn), then flows again.
      restingHome = player.perched && Math.hypot(player.pos[0] - nest.perch[0], player.pos[2] - nest.perch[2]) < 0.5 && (night > 0.05 || dayPhase > 0.5);
      wasPerched = player.perched;
      swallow.update(dt, player, current.avatar === "swallow" && wd < whw * 0.9 ? wy : null);
      if (playing) {
        const a = prevWind;
        const b = player.pos;
        const now = t;
        // Under a torii's lintel, between its pillars.
        for (const [gi, g] of gates.entries()) {
          const dx = g.cos;
          const dz = -g.sin;
          const sa = (a[0] - g.x) * dx + (a[2] - g.z) * dz;
          const sb = (b[0] - g.x) * dx + (b[2] - g.z) * dz;
          const lateral = Math.abs((b[0] - g.x) * -dz + (b[2] - g.z) * dx);
          if (sa * sb < 0 && lateral < 1.9 && b[1] - terrainHeight(g.x, g.z) < 4.3) thread(b, `torii${gi}`, now);
        }
        // Between two trunks, under the canopy.
        if (player.altitude < 3.4) {
          const near = trees.near(b[0], b[2], 9).slice(0, 14);
          for (let i = 0; i < near.length; i++) {
            for (let j = i + 1; j < near.length; j++) {
              const ti = near[i];
              const tj = near[j];
              const gap = Math.hypot(ti.x - tj.x, ti.z - tj.z);
              if (gap < 2.2 || gap > 7.5) continue;
              if (crosses(a, b, [ti.x, ti.z], [tj.x, tj.z])) thread(b, `${Math.round(ti.x)},${Math.round(ti.z)}|${Math.round(tj.x)},${Math.round(tj.z)}`, now);
            }
          }
        }
      }
      prevWind.splice(0, 3, player.pos[0], player.pos[1], player.pos[2]);
      insects.update(dt, t, player.pos, player.yaw, current.avatar === "swallow", rain);
      // Skimming the river: the swallow drinks on the wing, its wake splashing behind.
      {
        const [rd, , rhw] = riverInfo(player.pos[0], player.pos[2]);
        skimTimer -= dt;
        if (swallow.dip > 0.5 && skimTimer <= 0) {
          // The wingtip slicing the water: a continuous fine spray off it.
          skimTimer = 0.07;
          motes.splash([swallow.tip[0], swallow.tip[1], swallow.tip[2]], 0.06);
          if (Math.random() < 0.3) audio.skim();
        } else if (rd < rhw * 0.95 && player.altitude < 0.95 && skimTimer <= 0) {
          skimTimer = 0.11;
          motes.splash([player.pos[0], player.pos[1] - player.altitude, player.pos[2]], 0.12);
          if (Math.random() < 0.35) audio.skim();
        }
      }
      player.followScale += ((current.avatar === "swallow" ? 0.52 : 1) - player.followScale) * Math.min(1, dt * 2);
      birds.update(dt, player.pos, player.altitude, night, player.yaw);
      if (playing && !paused) {
        regionCheck -= dt;
        if (regionCheck <= 0) {
          regionCheck = 0.5;
          const [px, , pz] = player.pos;
          for (const r of REGIONS) {
            if (!regionsSeen.has(r.name) && r.inside(px, pz)) {
              regionsSeen.add(r.name);
              showRegion();
              break;
            }
          }
          if (gates.some((g) => Math.hypot(g.x - px, g.z - pz) < 4)) discoveries.find("torii");
        }
      }
      {
        const low = 1 - Math.min(1, Math.max(0, (player.altitude - 1.5) / 4));
        let leafy = 0;
        for (const tr of trees.near(player.pos[0], player.pos[2], 8)) {
          const d = Math.hypot(tr.x - player.pos[0], tr.z - player.pos[2]);
          leafy = Math.max(leafy, 1 - d / 8);
        }
        const over = sunflowerField(player.pos[0], player.pos[2]) > 0.3 ? 0.8 : 1;
        const water = 1 - Math.min(1, Math.max(0, (riverDist - riverHw) / 6));
        audio.setSurroundings(low * (1 - water) * over, leafy, water * low, Math.min(1, player.speed / 21));
      }
    }

    if (debug.fixedCamera) {
      camera.position.splice(0, 3, ...debug.fixedCamera.pos);
      camera.target.splice(0, 3, ...debug.fixedCamera.target);
    } else if (explore) {
      freecam.apply(camera);
    } else {
      player.updateCamera(camera, dt, 2 + Math.sqrt(motes.carried) * 0.4);
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
    // Under water? (the camera below the river's surface, over its bed)
    {
      const [cx, cy, cz] = camera.position;
      const [d, wy, hw] = riverInfo(cx, cz);
      waterY = wy;
      const target = d < hw * 1.6 && cy < wy - 0.02 && terrainHeight(cx, cz) < wy ? 1 : 0;
      underwater += (target - underwater) * Math.min(1, rawDt * 10);
      if (Math.abs(target - underwater) < 0.01) underwater = target;
      renderer.setUnderwater(underwater);
      audio.setUnderwater(underwater);
    }
    globals.updateFrame(camera, t, renderer.viewport, {
      underwater,
      waterY,
      rainbow: Math.min(1, rainbowT / 10) * Math.min(1, (90 - rainbowT) / 6),
      playerGlow: swallow.glow,
      dip: swallow.visible ? swallow.dip : 0,
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
      discoveries.find("lantern");
      if (lanternEvent.complete) {
        discoveries.find("chain");
        // The reward: a paper lantern rises from every stone lantern, and the gates light up.
        skyLanterns.release(lanterns.positions.map((p) => [p[0], p[1], p[2]] as Vec3));
      }
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
    torii.setGlow(discoveries.has("chain") ? 1 : 0, dt);
    skyLanterns.update(dt, [Math.cos(windAngle), Math.sin(windAngle)]);
    // A faint glow over feeding flocks (by day), so they can be spotted and flown to.
    {
      const glints: [number, number, number, number][] = night > 0.5 ? [] : birds.feedingSpots().map((h) => [h[0], h[1] + 1.4, h[2], 0.55]);
      const kf = race.waitingAt;
      if (kf) glints.push([kf[0], kf[1] + 0.6, kf[2], 1]);
      secrets.setExtraGlints(glints);
    }
    secrets.update(dt, explore ? freecam.pos : player.pos, explore ? 4 : player.speed, camera.position);
    race.update(dt, explore ? freecam.pos : player.pos, explore ? 6 : player.speed);
    fish.update(dt, camera.position, t);
    audio.setWhirr(whirr);
    whirr = 0;
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
      nest.encode(pass);
      undergrowth.encode(pass);
      flowers.encode(pass);
      motes.encode(pass);
      if (!explore) windTrail.encode(pass);
      swallow.visible = current.avatar === "swallow" && !explore;
      const darkGlow = Math.min(1, 0.06 + night * 0.75 + rain * 0.2 + canopy * 0.3);
      swallow.glow = swallow.visible ? darkGlow : 0;
      race.glow = darkGlow;
      swallow.encode(pass);
      birds.encode(pass);
      secrets.encode(pass);
      race.encode(pass);
      if (!debug.hide.terrain) terrain.encode(pass);
      fish.encode(pass);
      // The water blends over what lies beneath it, so it comes after the bed and the fish.
      if (!debug.hide.water) water.encode(pass);
      flowers.encodeGlow(pass);
      if (!debug.hide.fireflies) fireflies.encode(pass, night);
      // Transparent, depth-tested but not depth-writing: after everything opaque.
      secrets.encodeGlints(pass);
      if (!explore) {
        insects.encode(pass);
        thermals.encode(pass);
      }
      skyLanterns.encode(pass);
      precipitation.encode(pass, rain);
    }, spans);

    petalsEl.textContent = explore ? (freecam.fly ? "free roam · flying" : "free roam") : "";
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
