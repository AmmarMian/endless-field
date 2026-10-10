// node tools/dev/lodpop.mjs <prefix> <shift m> [height] [yaw]: same view, rings' centre moved
// `shift` m forward (as if flown there): the difference is LOD popping alone.
import { chromium } from "playwright";
const [out, shift, hgt, yawArg] = process.argv.slice(2);
const b = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
await p.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
await p.goto("http://localhost:5199/", { waitUntil: "domcontentloaded" });
await p.waitForFunction(() => window.__ef, null, { timeout: 60000 });
const yaw = +(yawArg ?? 4.0), h = +(hgt ?? 3);
await p.evaluate(([yaw, h]) => { const e = window.__ef; e.start(); e.setDay(0.25); e.fixedTime = 40; document.getElementById("hud").style.display = "none";
  const gy = e.height(0, 0); e.fixedCamera = { pos: [0, gy + h, 0], target: [Math.sin(yaw) * 200, gy + h - 8, -Math.cos(yaw) * 200] }; }, [yaw, h]);
await p.waitForTimeout(3000);
// Pause: the swallow, birds and its trail stop, so only the grass rings change.
await p.keyboard.press("p");
await p.waitForTimeout(400);
await p.screenshot({ path: `${out}-a.png` });
await p.evaluate(([yaw, s]) => { window.__ef.grass.lodShift = [Math.sin(yaw) * s, -Math.cos(yaw) * s]; }, [yaw, +shift]);
await p.waitForTimeout(600);
await p.screenshot({ path: `${out}-b.png` });
await b.close();
