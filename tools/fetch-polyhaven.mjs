// Downloads a Poly Haven (CC0) glTF model with its textures into assets-src/<id>/.
// Usage: node tools/fetch-polyhaven.mjs <asset-id> [resolution=1k]
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const [id, res = "1k"] = process.argv.slice(2);
if (!id) throw new Error("usage: fetch-polyhaven.mjs <asset-id> [1k|2k|4k]");
const files = await (await fetch(`https://api.polyhaven.com/files/${id}`)).json();
const entry = files.gltf?.[res]?.gltf;
if (!entry) throw new Error(`no gltf ${res} for ${id}`);
const outDir = join("assets-src", id);
const jobs = [[entry.url, `${id}.gltf`], ...Object.entries(entry.include ?? {}).map(([p, v]) => [v.url, p])];
for (const [url, rel] of jobs) {
  const dest = join(outDir, rel);
  await mkdir(dirname(dest), { recursive: true });
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  await writeFile(dest, Buffer.from(await r.arrayBuffer()));
  console.log("ok", rel);
}
