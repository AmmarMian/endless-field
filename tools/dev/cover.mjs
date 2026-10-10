// node tools/dev/cover.mjs <prefix> : low flight view, grass on/off (time frozen).
import { chromium } from "playwright";
const [out] = process.argv.slice(2);
const b = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
await p.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
await p.goto("http://localhost:5199/", { waitUntil: "domcontentloaded" });
await p.waitForFunction(() => window.__ef, null, { timeout: 60000 });
await p.evaluate(() => { const e = window.__ef; e.start(); e.setDay(0.25); e.fixedTime = 40; document.getElementById("hud").style.display = "none";
  const yaw = 4.0, x = 0, z = 0, gy = e.height(x, z);
  e.fixedCamera = { pos: [x, gy + 3, z], target: [x + Math.sin(yaw) * 200, gy - 3, z - Math.cos(yaw) * 200] }; });
await p.waitForTimeout(3000);
await p.screenshot({ path: `${out}-on.png` });
await p.evaluate(() => (window.__ef.hide.grass = true));
await p.waitForTimeout(600);
await p.screenshot({ path: `${out}-off.png` });
await b.close();
