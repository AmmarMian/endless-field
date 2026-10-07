// Frame-time benchmark across scenes (dev server on :5173): node tools/bench.mjs
import { chromium } from "playwright";
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
await p.goto("http://localhost:5173/");
await p.waitForFunction(() => window.__ef, null, { timeout: 90000 });
await p.evaluate(() => { __ef.start(); __ef.panel.set({ autoResolution: false, renderScale: 1, filter: "cozy" }); });
const scenes = {
  meadow: "",
  river: "const x=150, rc=__ef.river.center(x), w=__ef.river.water(x); __ef.player.teleport([x,w+1.6,rc], Math.PI/2);",
  forest: "__ef.player.teleport([950, __ef.height(950,120)+1.7, 120], Math.PI/2);",
  sunflowers: "__ef.player.teleport([-62-0.39*95, __ef.height(-99,-313)+2, -400+0.92*95], 0);",
  rainNight: "const x=150, rc=__ef.river.center(x), w=__ef.river.water(x); __ef.player.teleport([x,w+1.6,rc], Math.PI/2); __ef.setDay(0.8); __ef.panel.set({weather:'rain'});",
};
for (const [name, js] of Object.entries(scenes)) {
  await p.evaluate((js) => eval(js), js);
  await p.waitForTimeout(6000);
  const st = await p.evaluate(() => document.getElementById("stats").textContent);
  console.log(name.padEnd(11), st.replace(/\n/g, " | "));
}
await b.close();
