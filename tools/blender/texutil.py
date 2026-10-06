"""Image helpers for the asset pipelines (run inside Blender: uses bpy for PNG/JPG IO)."""
import os
import bpy
import numpy as np


def img_np(path):
    im = bpy.data.images.load(os.path.abspath(path))
    im.colorspace_settings.name = "Non-Color"
    a = np.zeros(im.size[0] * im.size[1] * 4, np.float32)
    im.pixels.foreach_get(a)
    a = a.reshape(im.size[1], im.size[0], 4)
    bpy.data.images.remove(im)
    return a


def save_png(arr, path, colorspace="sRGB"):
    h, w = arr.shape[:2]
    im = bpy.data.images.new(os.path.basename(path), w, h, alpha=True)
    im.colorspace_settings.name = colorspace
    im.pixels.foreach_set(np.ascontiguousarray(arr, np.float32).ravel())
    im.filepath_raw = os.path.abspath(path)
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)


def bleed(rgba):
    """Pull-push: fill transparent texels with nearby opaque colors so mips never fringe dark."""
    a = (rgba[..., 3:4] > 0.02).astype(np.float32)
    levels = [(rgba[..., :3] * a, a)]
    while levels[-1][0].shape[0] > 1:
        c, w = levels[-1]
        h2, w2 = c.shape[0] // 2, c.shape[1] // 2
        c = c[: h2 * 2, : w2 * 2].reshape(h2, 2, w2, 2, 3).sum((1, 3))
        w = w[: h2 * 2, : w2 * 2].reshape(h2, 2, w2, 2, 1).sum((1, 3))
        levels.append((c, w))
    fill = levels[-1][0] / np.maximum(levels[-1][1], 1e-6)
    for c, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(fill, 2, 0), 2, 1)
        up = np.pad(up, ((0, c.shape[0] - up.shape[0]), (0, c.shape[1] - up.shape[1]), (0, 0)), mode="edge")
        fill = np.where(w > 0, c / np.maximum(w, 1e-6), up)
    out = rgba.copy()
    out[..., :3] = fill
    return out
