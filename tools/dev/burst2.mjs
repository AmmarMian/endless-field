// node tools/dev/burst2.mjs <outdir> <setupJs> <startMs> <count> <stepMs> [probeJs]
import { chromium } from "playwright";
const [out, setup, start, count, step, probe] = process.argv.slice(2);
const b = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: +(process.env.W ?? 640), height: +(process.env.H ?? 400) } });
await p.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
await p.goto("http://localhost:5199/", { waitUntil: "domcontentloaded" });
await p.waitForFunction(() => window.__ef, null, { timeout: 60000 });
await p.waitForTimeout(500);
await p.evaluate(setup);
await p.waitForTimeout(+start);
for (let i = 0; i < +count; i++) {
  await p.screenshot({ path: `${out}/f${String(i).padStart(2, "0")}.png` });
  if (probe) console.log(i, JSON.stringify(await p.evaluate(probe)));
  await p.waitForTimeout(+step);
}
await b.close();
