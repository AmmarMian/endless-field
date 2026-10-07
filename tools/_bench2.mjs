import { chromium } from "playwright";
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
await p.goto("http://localhost:5173/");
await p.waitForFunction(() => window.__ef, null, { timeout: 90000 });
await p.evaluate(() => { __ef.start(); __ef.panel.set({ autoResolution: false, renderScale: 1 }); __ef.player.teleport([950, __ef.height(950,120)+1.7, 120], Math.PI/2); const pos=[...__ef.player.pos]; __ef.fixedCamera={pos:[pos[0]-5,pos[1]+1.5,pos[2]],target:[pos[0]+10,pos[1],pos[2]]}; });
await p.waitForTimeout(3000);
const read = async (label) => { await p.waitForTimeout(3500); console.log(label.padEnd(14), (await p.evaluate(() => document.getElementById("stats").textContent)).split("\n")[1]); };
await read("all");
for (const k of ["trees", "grass", "terrain", "water", "fireflies"]) {
  await p.evaluate((k) => { __ef.hide[k] = true; }, k);
  await read(`no ${k}`);
  await p.evaluate((k) => { __ef.hide[k] = false; }, k);
}
await b.close();
