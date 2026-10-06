// Downloads a Poly Haven (CC0) PBR texture set (diffuse, GL normal, ARM) as JPG.
// Usage: node tools/fetch-texture.mjs <asset-id> [resolution=1k]
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const [id, res = "1k"] = process.argv.slice(2);
const files = await (await fetch(`https://api.polyhaven.com/files/${id}`)).json();
const outDir = join("assets-src", "textures", id);
await mkdir(outDir, { recursive: true });
for (const [key, name] of [["Diffuse", "diff"], ["nor_gl", "nor"], ["arm", "arm"]]) {
  const entry = files[key]?.[res]?.jpg;
  if (!entry) {
    console.log("missing", key);
    continue;
  }
  const r = await fetch(entry.url);
  await writeFile(join(outDir, `${name}.jpg`), Buffer.from(await r.arrayBuffer()));
  console.log("ok", id, name);
}
