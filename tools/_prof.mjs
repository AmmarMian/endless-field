import { chromium } from "playwright";
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
await p.goto("http://localhost:5173/");
await p.waitForFunction(() => window.__ef, null, { timeout: 90000 });
await p.evaluate(() => { __ef.start(); __ef.panel.set({ autoResolution: false, renderScale: 1 }); __ef.player.teleport([950, __ef.height(950,120)+1.7, 120], Math.PI/2); });
await p.waitForTimeout(4000);
const cdp = await p.context().newCDPSession(p);
await cdp.send("Profiler.enable");
await cdp.send("Profiler.setSamplingInterval", { interval: 200 });
await cdp.send("Profiler.start");
await p.waitForTimeout(4000);
const { profile } = await cdp.send("Profiler.stop");
const self = new Map();
const byId = new Map(profile.nodes.map((n) => [n.id, n]));
const dt = profile.timeDeltas;
const counts = new Map();
profile.samples.forEach((id, i) => counts.set(id, (counts.get(id) ?? 0) + (dt[i] ?? 0)));
for (const [id, t] of counts) {
  const n = byId.get(id);
  const f = n.callFrame;
  const key = `${f.functionName || "(anon)"} ${f.url.split("/").pop()}:${f.lineNumber}`;
  self.set(key, (self.get(key) ?? 0) + t);
}
// Who is spending time in the height functions: attribute to the first caller outside them.
const parent = new Map();
for (const n of profile.nodes) for (const c of n.children ?? []) parent.set(c, n.id);
const callers = new Map();
for (const [id, t] of counts) {
  let n = byId.get(id);
  if (!/height\.ts|ecology\.ts|biome\.ts/.test(n.callFrame.url)) continue;
  while (n && /height\.ts|ecology\.ts|biome\.ts/.test(n.callFrame.url)) n = byId.get(parent.get(n.id));
  const key = n ? `${n.callFrame.functionName || "(anon)"} ${n.callFrame.url.split("/").pop().split("?")[0]}:${n.callFrame.lineNumber}` : "?";
  callers.set(key, (callers.get(key) ?? 0) + t);
}
console.log("-- height-function time by caller --");
[...callers.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).forEach(([k, t]) => console.log((t / 1000).toFixed(0).padStart(6) + " ms  " + k));
const total = [...self.values()].reduce((a, b) => a + b, 0);
[...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18).forEach(([k, t]) => console.log((t / total * 100).toFixed(1).padStart(5) + "%  " + k));
await b.close();
