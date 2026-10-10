import { webkit, devices } from "playwright";
const url = process.argv[2] ?? "http://localhost:5199/";
const b = await webkit.launch();
const ctx = await b.newContext({ ...devices["iPhone 15"] });
const p = await ctx.newPage();
const logs = [];
p.on("console", (m) => logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`));
p.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
await p.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
const t0 = Date.now();
await p.goto(url, { waitUntil: "domcontentloaded" });
console.log("gpu?", await p.evaluate(() => !!navigator.gpu));
try {
  await p.waitForFunction(() => window.__ef || !document.getElementById("error")?.hidden, null, { timeout: 90000 });
  console.log("ready ms", Date.now() - t0);
  await p.evaluate(() => window.__ef?.start());
  await p.waitForTimeout(5000);
} catch (e) { console.log("TIMEOUT", e.message.slice(0, 100)); }
const st = await p.evaluate(() => ({ err: document.getElementById("error")?.hidden ? null : document.getElementById("error")?.textContent, notice: document.getElementById("notice")?.textContent ?? null, stage: document.querySelector(".ld-stage")?.textContent }));
console.log(JSON.stringify(st));
await p.screenshot({ path: "tools/out/wk.png" });
console.log(logs.filter((l) => !l.includes("Biquad")).slice(0, 30).join("\n"));
await b.close();
