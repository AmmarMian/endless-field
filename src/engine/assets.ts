/**
 * Which assets have compressed web copies (tools/web-assets.py): images as WebP, meshes as
 * gzip. Loaded once at start; without it (or for anything unlisted) the originals load.
 */
let web: { webp: Set<string>; gz: Set<string> } = { webp: new Set(), gz: new Set() };

export async function loadAssetIndex(): Promise<void> {
  try {
    const j = (await (await fetch("assets/web.json")).json()) as { webp: string[]; gz: string[] };
    web = { webp: new Set(j.webp), gz: new Set(j.gz) };
  } catch {
    // Originals only.
  }
}

/** The URL to fetch for an image (its WebP copy when there is one). */
export function imageUrl(url: string): string {
  return web.webp.has(url.replace(/\.m(\.\w+)$/, "$1")) ? url.replace(/\.(png|jpg)$/, ".webp") : url;
}

/**
 * A binary asset (a mesh), from its gzip copy when there is one. Servers differ: some send
 * the .gz file as is, some mark it gzip-encoded so the browser has already unpacked it; the
 * gzip header tells which.
 */
export async function loadBin(url: string): Promise<ArrayBuffer> {
  if (web.gz.has(url) && typeof DecompressionStream !== "undefined") {
    try {
      const r = await fetch(`${url}.gz`);
      if (r.ok) {
        const data = await r.arrayBuffer();
        const head = new Uint8Array(data, 0, Math.min(2, data.byteLength));
        if (head[0] !== 0x1f || head[1] !== 0x8b) return data;
        return await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
      }
    } catch {
      // Fall back to the original below.
    }
  }
  return (await fetch(url)).arrayBuffer();
}
