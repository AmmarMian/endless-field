// node tools/dev/popdiff.mjs <out prefix> <step m> [height] [yaw]
// Two frames, the camera moved `step` meters forward, time frozen: what changes is LOD popping.
import { chromium } from "playwright";
const [out, step, hgt, yawArg] = process.argv.slice(2);
const b = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
await p.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
await p.goto("http://localhost:5199/", { waitUntil: "domcontentloaded" });
await p.waitForFunction(() => window.__ef, null, { timeout: 60000 });
const yaw = +(yawArg ?? 4.0), h = +(hgt ?? 3);
const place = (d) => p.evaluate(([d, yaw, h]) => {
  const e = window.__ef;
  const x = Math.sin(yaw) * d, z = -Math.cos(yaw) * d;
  const gy = e.height(x, z);
  e.fixedCamera = { pos: [x, gy + h, z], target: [x + Math.sin(yaw) * 200, gy + h - 6, z - Math.cos(yaw) * 200] };
}, [d, yaw, h]);
await p.evaluate(() => { const e = window.__ef; e.start(); e.setDay(0.25); e.fixedTime = 40; document.getElementById("hud").style.display = "none"; });
await place(0);
await p.waitForTimeout(3000);
await p.screenshot({ path: `${out}-a.png` });
await place(+step);
await p.waitForTimeout(600);
await p.screenshot({ path: `${out}-b.png` });
await b.close();
