// Usage: node tools/shot.mjs <out.png> [--url URL] [--wait ms] [--eval "js"] [--size WxH]
// After the wait, `window.__ef.probe()` (if the eval defined one) is reported as `probe`.
// Captures the running app with WebGPU in Chromium, prints console errors and pixel stats.
import { chromium } from "playwright";

const args = process.argv.slice(2);
const out = args[0] ?? "tools/out/shot.png";
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const url = opt("url", "http://localhost:5199/");
const wait = Number(opt("wait", "4000"));
const evalJs = opt("eval", "");
const [w, h] = opt("size", "1280x720").split("x").map(Number);

const browser = await chromium.launch({
  channel: "chromium",
  headless: opt("headed", "0") !== "1",
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
const logs = [];
page.on("console", (m) => {
  if (m.type() === "error" || m.type() === "warning") logs.push(`[${m.type()}] ${m.text()}`);
});
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
page.on("response", (r) => {
  if (r.status() >= 400) logs.push(`[http ${r.status()}] ${r.url()}`);
});
await page.goto(url);
const t0 = Date.now();
await page.waitForFunction(() => window.__ef || !document.getElementById("error")?.hidden, null, { timeout: 60000 });
console.log(`ready in ${Date.now() - t0} ms`);
await page.waitForTimeout(500);
if (evalJs) {
  const r = await page.evaluate(evalJs);
  if (r !== undefined) console.log("eval:", JSON.stringify(r));
}
await page.waitForTimeout(wait);
await page.screenshot({ path: out });
const stats = await page.evaluate(async () => {
  const c = document.querySelector("canvas");
  const err = document.getElementById("error");
  const ef = window.__ef;
  return {
    canvas: c ? [c.width, c.height] : null,
    error: err && !err.hidden ? err.textContent : null,
    grass: ef?.grassCounts ? await ef.grassCounts() : null,
    stats: document.getElementById("stats")?.textContent ?? "",
    mode: document.getElementById("petals")?.textContent ?? "",
    probe: ef?.probe ? ef.probe() : undefined,
  };
});
console.log(JSON.stringify(stats));
for (const l of logs.slice(0, 30)) console.log(l);
await browser.close();
