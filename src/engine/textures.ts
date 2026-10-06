import { texture, type Gpu, type Texture } from "vgpu";

export interface TextureLoadOptions {
  srgb: boolean;
  /** Keep straight (non-premultiplied) alpha, e.g. for alpha-tested foliage. */
  alpha?: boolean;
}

/**
 * Loads an image into a mipmapped GPU texture. Mip levels are produced by the browser's
 * high-quality image resampler, so no extra GPU pass is needed.
 */
export async function loadTexture(gpu: Gpu, url: string, opts: TextureLoadOptions): Promise<Texture> {
  const blob = await (await fetch(url)).blob();
  const bitmapOpts: ImageBitmapOptions = {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  };
  const base = await createImageBitmap(blob, bitmapOpts);
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
  const mips = [base];
  for (let level = 1; level < levels; level++) {
    const w = Math.max(1, base.width >> level);
    const h = Math.max(1, base.height >> level);
    mips.push(await createImageBitmap(base, { ...bitmapOpts, resizeWidth: w, resizeHeight: h, resizeQuality: "high" }));
  }
  mips.forEach((bmp, level) => {
    queue.copyExternalImageToTexture({ source: bmp }, { texture: tex.gpu, mipLevel: level }, [bmp.width, bmp.height]);
    bmp.close();
  });
  return tex;
}
