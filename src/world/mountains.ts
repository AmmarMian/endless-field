import { sampler, texture, type Gpu, type Texture } from "vgpu";
import { setMountains } from "./height";

export interface MountainSet {
  texture: Texture;
  sampler: GPUSampler;
}

function floatToHalf(v: number): number {
  const f = new Float32Array([v]);
  const u = new Uint32Array(f.buffer)[0];
  const sign = (u >> 16) & 0x8000;
  let exp = ((u >> 23) & 0xff) - 127 + 15;
  const mant = u & 0x7fffff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | (mant >> 13);
}

/** 2x2 box-filtered mip chain (heights stay in their natural units). */
function mipChain(base: Float32Array, n: number): { size: number; data: Float32Array }[] {
  const out = [{ size: n, data: base }];
  let cur = base;
  let size = n;
  while (size > 1) {
    const half = size >> 1;
    const next = new Float32Array(half * half);
    for (let y = 0; y < half; y++) {
      for (let x = 0; x < half; x++) {
        const i = y * 2 * size + x * 2;
        next[y * half + x] = (cur[i] + cur[i + 1] + cur[i + size] + cur[i + size + 1]) * 0.25;
      }
    }
    out.push({ size: half, data: next });
    cur = next;
    size = half;
  }
  return out;
}

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

/**
 * Loads the Blender-generated massif heightfields: one r16float array layer each on the GPU,
 * and decoded floats for the CPU height mirror (both sample them identically).
 */
export async function loadMountains(gpu: Gpu, base = "/assets/mountains"): Promise<MountainSet> {
  const meta = (await (await fetch(`${base}/mountains.json`)).json()) as { size: number; massifs: string[] };
  const n = meta.size;
  const raw = await Promise.all(meta.massifs.map(async (f) => new Uint16Array(await (await fetch(`${base}/${f}`)).arrayBuffer())));
  const levels = Math.log2(n) + 1;
  const tex = texture(gpu, {
    kind: "2d-array",
    size: [n, n],
    layers: raw.length,
    format: "r16float",
    usage: ["texture_binding", "copy_dst"],
    mipLevelCount: levels,
    label: "mountains",
  });
  const floats = raw.map((d) => Float32Array.from(d, halfToFloat));
  floats.forEach((data, layer) => {
    mipChain(data, n).forEach(({ size, data: level }, mip) => {
      const halves = Uint16Array.from(level, floatToHalf);
      // Rows must be 256-byte aligned only for buffer copies; writeTexture has no such rule.
      gpu.device.gpu.queue.writeTexture({ texture: tex.gpu, origin: [0, 0, layer], mipLevel: mip }, halves, { bytesPerRow: size * 2, rowsPerImage: size }, [size, size, 1]);
    });
  });
  setMountains(floats, n);
  return {
    texture: tex,
    sampler: sampler(gpu, { minFilter: "linear", magFilter: "linear", mipmapFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" }),
  };
}
