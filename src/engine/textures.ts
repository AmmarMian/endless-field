import { texture, type Gpu, type Texture } from "vgpu";
import { imageUrl } from "./assets";

export interface TextureLoadOptions {
  srgb: boolean;
  /** Keep straight (non-premultiplied) alpha, e.g. for alpha-tested foliage. */
  alpha?: boolean;
}

/**
 * Loads an image into a mipmapped GPU texture. Mip levels are produced by the browser's
 * high-quality image resampler, so no extra GPU pass is needed.
 */
let maxSize = 8192;
/** Phone-sized copies (tools/phone-textures.py), by original url, when in use. */
let small: Set<string> | null = null;
/** Decodes in flight at once (phones run out of memory decoding many large images together). */
let slots = 6;
const waiting: (() => void)[] = [];

/** Caps loaded textures (longest side, px): phones keep their graphics memory for the scene. */
export function setTextureMaxSize(px: number): void {
  maxSize = px;
}

/** Phones: load the small copies listed in assets/phone-textures.json, one or two at a time. */
export async function usePhoneTextures(concurrency = 2): Promise<void> {
  slots = concurrency;
  try {
    small = new Set((await (await fetch("assets/phone-textures.json")).json()) as string[]);
  } catch {
    small = null;
  }
}

async function slot<T>(work: () => Promise<T>): Promise<T> {
  // A freed slot passes straight to the next waiter.
  if (slots > 0) slots--;
  else await new Promise<void>((r) => waiting.push(r));
  try {
    return await work();
  } finally {
    const next = waiting.shift();
    if (next) next();
    else slots++;
  }
}

export function loadTexture(gpu: Gpu, url: string, opts: TextureLoadOptions): Promise<Texture> {
  return slot(() => decodeTexture(gpu, url, opts));
}

async function decodeTexture(gpu: Gpu, url: string, opts: TextureLoadOptions): Promise<Texture> {
  const src = small?.has(url) ? url.replace(/(\.\w+)$/, ".m$1") : url;
  const blob = await (await fetch(imageUrl(src))).blob();
  const bitmapOpts: ImageBitmapOptions = {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  };
  let base = await createImageBitmap(blob, bitmapOpts);
  const longest = Math.max(base.width, base.height);
  if (longest > maxSize) {
    const k = maxSize / longest;
    const full = base;
    base = await createImageBitmap(full, { ...bitmapOpts, resizeWidth: Math.round(full.width * k), resizeHeight: Math.round(full.height * k), resizeQuality: "high" });
    full.close();
  }
  const levels = Math.floor(Math.log2(Math.max(base.width, base.height))) + 1;
  const tex = texture(gpu, {
    kind: "2d",
    size: [base.width, base.height],
    format: opts.srgb ? "rgba8unorm-srgb" : "rgba8unorm",
    usage: ["texture_binding", "copy_dst", "render_attachment"],
    mipLevelCount: levels,
    label: url,
  });
  const queue = gpu.device.gpu.queue;
  // Each level from the one above it. The bitmaps are released only once the queue has taken
  // the copies (closing one right after queueing its copy can upload nothing).
  const mips = [base];
  queue.copyExternalImageToTexture({ source: base }, { texture: tex.gpu, mipLevel: 0 }, [base.width, base.height]);
  for (let level = 1; level < levels; level++) {
    const w = Math.max(1, base.width >> level);
    const h = Math.max(1, base.height >> level);
    const bmp = await createImageBitmap(mips[level - 1], { ...bitmapOpts, resizeWidth: w, resizeHeight: h, resizeQuality: "high" });
    queue.copyExternalImageToTexture({ source: bmp }, { texture: tex.gpu, mipLevel: level }, [w, h]);
    mips.push(bmp);
  }
  await queue.onSubmittedWorkDone();
  for (const m of mips) m.close();
  return tex;
}
