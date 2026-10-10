// node tools/dev/mem.mjs [url] : phone-emulated load, tallies GPU buffer/texture bytes by label.
import { chromium, devices } from "playwright";
const url = process.argv[2] ?? "http://localhost:5199/";
const b = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist", "--enable-precise-memory-info"] });
const ctx = await b.newContext({ ...devices["iPhone 15"], deviceScaleFactor: 3 });
const p = await ctx.newPage();
await p.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
await p.addInitScript(() => {
  const tally = (window.__gpuMem = { buffers: {}, textures: {}, total: 0 });
  const fmtBytes = { rgba16float: 8, rgba8unorm: 4, "rgba8unorm-srgb": 4, depth32float: 4, bgra8unorm: 4, rgba32float: 16, r32float: 4, rg16float: 4, r16float: 2, r8unorm: 1, rg8unorm: 2, "depth24plus": 4, "depth24plus-stencil8": 4 };
  const ob = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function (d) {
    const k = d.label || `? ${d.size} u${d.usage}`;
    tally.buffers[k] = (tally.buffers[k] || 0) + d.size;
    tally.total += d.size;
    return ob.call(this, d);
  };
  const ot = GPUDevice.prototype.createTexture;
  GPUDevice.prototype.createTexture = function (d) {
    const s = Array.isArray(d.size) ? d.size : [d.size.width, d.size.height ?? 1, d.size.depthOrArrayLayers ?? 1];
    const mips = d.mipLevelCount ?? 1;
    const bytes = (s[0] * s[1] * (s[2] ?? 1) * (fmtBytes[d.format] ?? 4) * (d.sampleCount ?? 1) * (mips > 1 ? 1.33 : 1)) | 0;
    const k = (d.label || "?") + ` ${d.format} ${s.join("x")}${d.sampleCount > 1 ? " msaa" + d.sampleCount : ""}`;
    tally.textures[k] = (tally.textures[k] || 0) + bytes;
    tally.total += bytes;
    return ot.call(this, d);
  };
});
let downloaded = 0;
p.on("response", async (r) => {
  try {
    const len = (await r.body().catch(() => Buffer.alloc(0))).length;
    downloaded += len;
  } catch {}
});
const t0 = Date.now();
await p.goto(url, { waitUntil: "domcontentloaded" });
await p.waitForFunction(() => window.__ef || !document.getElementById("error")?.hidden, null, { timeout: 120000 });
console.log("ready ms", Date.now() - t0);
await p.evaluate(() => window.__ef?.start());
await p.waitForTimeout(4000);
const r = await p.evaluate(() => {
  const m = window.__gpuMem;
  const top = (o) => Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, 25).map(([k, v]) => `${(v / 1e6).toFixed(1)} MB  ${k}`);
  const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0) / 1e6;
  return { totalMB: (m.total / 1e6).toFixed(1), buffersMB: sum(m.buffers).toFixed(1), texturesMB: sum(m.textures).toFixed(1), heapMB: (performance.memory?.usedJSHeapSize / 1e6).toFixed(1), stats: document.getElementById("stats")?.textContent, error: document.getElementById("error")?.hidden ? null : document.getElementById("error")?.textContent, notice: document.getElementById("notice")?.textContent ?? null, topBuffers: top(m.buffers), topTextures: top(m.textures) };
});
console.log("downloaded MB", (downloaded / 1e6).toFixed(1));
console.log(JSON.stringify(r, null, 1));
await p.screenshot({ path: "tools/out/phone.png" });
await b.close();
