"""
Models the sunflower (Helianthus annuus) in Blender and exports it for the wind-simulated field.

  blender -b --factory-startup --python tools/blender/model_sunflower.py -- <out dir> [--preview <png>]

Anatomy, each part built as its own mesh and joined per variant:
  - stem: tapered, slightly bristly tube; the top "neck" arcs over (shepherd's crook) under the
    weight of a mature head, so the head faces forward / down
  - leaves: alternate (opposite at the base), on petioles, cordate-ovate blades with a folded
    midrib, gravity droop and wavy margins; the outline (with serration) is in the alpha
  - head: domed disk painted with ~1400 florets in Fibonacci phyllotaxis (golden angle),
    a calyx cup behind it and two rings of pointed green bracts (phyllaries)
  - ray petals: two staggered rings, cupped across and arched along, each with its own twist

Variants differ in age (head size, nod angle, height, leaf count). Each variant is built at
three detail levels. Vertex attributes carry what the wind simulation needs:
  p.w  part (0 stem, 1 leaf, 2 disk, 3 petal, 4 calyx/bract)
  e.x  position along the stem in [0, 1] (leaves: where they attach; head parts: 1)
  e.y  flutter weight (how freely a leaf / petal vertex moves)
  e.z  ambient occlusion
  e.w  per-organ random phase
Output: sunflower.bin (vertices then indices), sunflower.json (variants x lods, pivots),
albedo.png (RGBA atlas). Model space is Y-up, heads facing +Z.
"""
import json, math, os, random, sys
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(__file__))
from texutil import bleed, save_png  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None
os.makedirs(OUT, exist_ok=True)
ATLAS = 2048
GOLDEN = math.pi * (3 - math.sqrt(5))

# Atlas regions (u0, v0, w, h) in UV space, v pointing down the image.
R_DISK = (0.0, 0.0, 0.5, 0.5)
R_CALYX = (0.5, 0.0, 0.25, 0.25)
R_STEM = (0.75, 0.0, 0.25, 0.25)
R_PETAL = [(0.5, 0.25, 0.125, 0.25), (0.625, 0.25, 0.125, 0.25)]
R_BRACT = (0.75, 0.25, 0.125, 0.25)
R_FACE = (0.875, 0.25, 0.125, 0.125)  # whole flower face for the far LOD
R_LEAF = [(0.0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)]  # fresh, old

# ---------------------------------------------------------------------------------------------
# Texture painting (numpy). Each painter works in the part's own (a, b) coordinates, the same
# ones the geometry uses for UVs, so outlines line up exactly.


def region_coords(reg):
    u0, v0, w, h = reg
    W, H = int(w * ATLAS), int(h * ATLAS)
    v = (np.arange(H) + 0.5) / H
    u = (np.arange(W) + 0.5) / W
    return np.meshgrid(u, v)  # u across, v down


def put(img, reg, rgba):
    u0, v0, w, h = reg
    x0, y0 = int(u0 * ATLAS), int(v0 * ATLAS)
    img[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]] = rgba


def vnoise(x, y, seed=0):
    """Smooth value noise for texture mottling."""
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    xf, yf = x - xi, y - yi

    def h(i, j):
        n = (i * 374761393 + j * 668265263 + seed * 2246822519) & 0xFFFFFFFF
        n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
        return (n & 0xFFFF) / 65535.0

    sx, sy = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf)
    a = h(xi, yi) + (h(xi + 1, yi) - h(xi, yi)) * sx
    b = h(xi, yi + 1) + (h(xi + 1, yi + 1) - h(xi, yi + 1)) * sx
    return a + (b - a) * sy


def fbm(x, y, seed=0, octaves=4):
    s, amp, tot = 0.0, 1.0, 0.0
    for o in range(octaves):
        s = s + vnoise(x * 2**o, y * 2**o, seed + o) * amp
        tot += amp
        amp *= 0.5
    return s / tot


def leaf_halfwidth(a):
    """Cordate-ovate outline: lobed base, widest at ~35%, long acuminate tip; with serration."""
    a = np.clip(a, 0, 1)
    body = np.sin(np.pi * np.clip(a * 1.08, 0, 1) ** 0.72) ** 0.85 * (1 - a) ** 0.18
    lobe = 0.22 * np.exp(-((a - 0.05) / 0.07) ** 2)
    teeth = 1 - 0.045 * (0.5 + 0.5 * np.sin(a * 2 * np.pi * 16)) * (a > 0.06)
    return np.clip((body + lobe) * 0.98 * teeth, 0, 1)


def paint_leaf(old):
    u, v = region_coords(R_LEAF[0])
    a, b = 1 - v, u * 2 - 1
    hw = leaf_halfwidth(a)
    inside = np.abs(b) < hw
    edge = np.clip((hw - np.abs(b)) * 140, 0, 1)
    base_col = np.array([0.16, 0.30, 0.07]) if not old else np.array([0.36, 0.33, 0.09])
    m = fbm(u * 18, v * 18, 3 if old else 1)
    col = base_col[None, None] * (0.82 + 0.36 * m[..., None])
    # Secondary veins leave the midrib and sweep toward the tip; areoles between them.
    bn = np.abs(b) / np.maximum(hw, 1e-3)
    vein_t = (a - 0.4 * np.abs(b) - 0.04) / 0.105
    sec = np.exp(-(((vein_t - np.round(vein_t)) * 0.105) / 0.006) ** 2) * (bn < 0.92) * (a > 0.03)
    mid = np.exp(-(b / 0.016) ** 2) * (a < 0.97)
    tert = 0.5 + 0.5 * np.sin(u * 260 + np.sin(v * 90) * 2) * np.sin(v * 230 + np.sin(u * 70) * 2)
    col = col * (0.93 + 0.07 * tert[..., None])
    vein_col = np.array([0.42, 0.55, 0.22]) if not old else np.array([0.55, 0.5, 0.2])
    col = col + (vein_col - col) * np.clip(sec * 0.55 + mid * 0.85, 0, 1)[..., None]
    # Margins dry first; old leaves brown in patches.
    dry = np.clip((bn - 0.8) * 4, 0, 1) * (0.25 if not old else 0.7)
    if old:
        dry = np.clip(dry + np.clip(fbm(u * 7, v * 7, 9) * 1.6 - 0.75, 0, 1), 0, 1)
    col = col + (np.array([0.42, 0.28, 0.1]) - col) * dry[..., None]
    rgba = np.concatenate([col, (edge * inside)[..., None]], -1)
    return rgba


def petal_halfwidth(a):
    a = np.clip(a, 0, 1)
    w = 0.92 * np.sin(np.pi * np.clip(a, 0, 1) ** 0.62) ** 0.55
    notch = 1 - 0.18 * np.exp(-((a - 0.985) / 0.03) ** 2)
    return w * np.where(a > 0.9, 1 - ((a - 0.9) / 0.1) ** 2 * 0.55, 1) * notch


def paint_petal(k):
    u, v = region_coords(R_PETAL[0])
    a, b = 1 - v, u * 2 - 1
    hw = petal_halfwidth(a)
    edge = np.clip((hw - np.abs(b)) * 90, 0, 1)
    deep = np.array([0.93, 0.42, 0.02]) if k == 0 else np.array([0.95, 0.5, 0.03])
    main = np.array([1.0, 0.74, 0.06]) if k == 0 else np.array([1.0, 0.8, 0.12])
    tip = np.array([1.0, 0.86, 0.3])
    t = np.clip(a / 0.35, 0, 1)[..., None]
    col = deep + (main - deep) * t
    col = col + (tip - col) * np.clip((a[..., None] - 0.75) / 0.25, 0, 1) * 0.5
    # Longitudinal veins converge to the base.
    vb = b / np.maximum(hw, 0.05)
    veins = 0.5 + 0.5 * np.cos(vb * np.pi * 9 + fbm(u * 6, v * 40, 4 + k) * 2.5)
    col = col * (0.86 + 0.14 * veins[..., None]) * (0.9 + 0.1 * fbm(u * 10, v * 30, 7 + k))[..., None]
    return np.concatenate([col, edge[..., None]], -1)


def paint_disk():
    u, v = region_coords(R_DISK)
    x, y = u * 2 - 1, v * 2 - 1
    r = np.sqrt(x * x + y * y)
    H, W = u.shape
    height = np.zeros_like(r)
    pollen = np.zeros_like(r)
    n = 1500
    for i in range(1, n):
        rr = math.sqrt(i / n) * 0.985
        th = i * GOLDEN
        cx, cy = rr * math.cos(th), rr * math.sin(th)
        # Florets touch their neighbours, so the Fibonacci spirals (21/34/55 parastichies) read.
        size = 0.03 + 0.008 * rr
        px, py = int((cx * 0.5 + 0.5) * W), int((cy * 0.5 + 0.5) * H)
        k = int(size * W * 1.2) + 2
        ys, xs = slice(max(py - k, 0), min(py + k, H)), slice(max(px - k, 0), min(px + k, W))
        dx, dy = x[ys, xs] - cx, y[ys, xs] - cy
        d2 = (dx * dx + dy * dy) / (size * size)
        bump = np.clip(1 - d2, 0, 1) ** 0.5 * (0.75 + 0.25 * ((i * 2654435761) % 997) / 997)
        height[ys, xs] = np.maximum(height[ys, xs], bump)
        if rr > 0.72:
            # Open disc florets: a five-lobed corolla tipped with pollen.
            ang = np.arctan2(dy, dx)
            star = np.clip(1 - d2 * (1.6 - 0.6 * np.cos(ang * 5)), 0, 1)
            pollen[ys, xs] = np.maximum(pollen[ys, xs], star * (0.6 + 0.4 * ((i * 7) % 5) / 4))
    center = np.array([0.2, 0.17, 0.06])
    inner = np.array([0.11, 0.06, 0.025])
    outer = np.array([0.26, 0.1, 0.03])
    t1 = np.clip(r / 0.2, 0, 1)[..., None]
    col = center + (inner - center) * t1
    col = col + (outer - col) * np.clip((r[..., None] - 0.5) / 0.35, 0, 1)
    # Each floret is a small dome with a lit crown and dark crevices between neighbours.
    col = col * (0.3 + 0.95 * height[..., None] ** 0.8)
    col = col + (np.array([0.78, 0.55, 0.08]) - col) * (pollen * 0.85)[..., None]
    alpha = np.clip((1.0 - r) * 200, 0, 1)
    return np.concatenate([col, alpha[..., None]], -1)


def paint_calyx():
    u, v = region_coords(R_CALYX)
    # u = angle around the head, v = from the stem (0) to the rim (1): overlapping scales.
    row = v * 6
    shift = np.floor(row) * 0.5
    cell = (u * 34 + shift) % 1
    scale = np.clip(1 - np.abs(cell - 0.5) * 2 - (row % 1) * 0.9, 0, 1)
    col = np.array([0.18, 0.3, 0.08])[None, None] * (0.6 + 0.5 * scale[..., None]) * (0.85 + 0.3 * fbm(u * 30, v * 10, 11)[..., None])
    return np.concatenate([col, np.ones_like(u)[..., None]], -1)


def paint_stem():
    u, v = region_coords(R_STEM)
    stripes = 0.5 + 0.5 * np.sin(u * 2 * np.pi * 9 + fbm(u * 4, v * 3, 12) * 3)
    col = np.array([0.22, 0.34, 0.09])[None, None] * (0.8 + 0.25 * stripes[..., None])
    base = np.clip((v - 0.75) / 0.25, 0, 1)[..., None]  # v = 1 at the ground
    col = col + (np.array([0.28, 0.2, 0.1]) - col) * base * 0.6
    hairs = (vnoise(u * 300, v * 300, 13) > 0.93) * 0.25
    col = col + hairs[..., None] * 0.6
    return np.concatenate([col, np.ones_like(u)[..., None]], -1)


def paint_bract():
    u, v = region_coords(R_BRACT)
    a, b = 1 - v, u * 2 - 1
    hw = 0.9 * np.sin(np.pi * np.clip(a, 0, 1) ** 0.5) * (1 - a) ** 0.7 + 0.04 * (a < 0.98)
    edge = np.clip((hw - np.abs(b)) * 60, 0, 1)
    col = np.array([0.2, 0.33, 0.09])[None, None] * (0.85 + 0.25 * fbm(u * 8, v * 20, 14)[..., None])
    col = col * (0.9 + 0.2 * np.exp(-(b / 0.08) ** 2))[..., None]
    return np.concatenate([col, edge[..., None]], -1)


def paint_face(disk, petal):
    """Disk + a ring of petals in one card, sampled from the painted disk / petal textures."""
    u, v = region_coords(R_FACE)
    x, y = u * 2 - 1, v * 2 - 1
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    rd = 0.58  # disk radius relative to the card (petals ~0.72 R long)
    H, W = disk.shape[:2]
    dx = np.clip(((x / rd) * 0.5 + 0.5) * W, 0, W - 1).astype(int)
    dy = np.clip(((y / rd) * 0.5 + 0.5) * H, 0, H - 1).astype(int)
    col = disk[dy, dx, :3]
    out = np.zeros(u.shape + (4,), np.float32)
    out[..., :3] = col
    out[..., 3] = (r < rd).astype(np.float32)
    PH, PW = petal.shape[:2]
    for ring, (count, off, ln) in enumerate([(34, 0.0, 1.0), (21, 0.5, 0.9)]):
        k = (th / (2 * np.pi) * count - off) % 1.0
        b = (k - 0.5) * 2  # across one petal sector
        a = (r - rd * 0.95) / ((1 - rd * 0.95) * ln)
        inside = (a >= 0) & (a <= 1)
        # Sector width at radius r vs petal width: petals ~ fill their sector near the base.
        bb = b * (2 * np.pi * r / count) / (0.44 * (1 - rd) * ln) * 1.1
        px = np.clip(((bb * 0.5 + 0.5)) * PW, 0, PW - 1).astype(int)
        py = np.clip((1 - a) * PH, 0, PH - 1).astype(int)
        pc = petal[py, px]
        m = inside & (np.abs(bb) <= 1) & (pc[..., 3] > 0.5) & (out[..., 3] < 0.5)
        out[m, :3] = pc[m, :3] * (0.9 if ring else 1.0)
        out[m, 3] = 1.0
    return out


def paint_atlas():
    img = np.zeros((ATLAS, ATLAS, 4), np.float32)
    disk = paint_disk()
    put(img, R_DISK, disk)
    put(img, R_CALYX, paint_calyx())
    put(img, R_STEM, paint_stem())
    petal0 = paint_petal(0)
    put(img, R_PETAL[0], petal0)
    put(img, R_FACE, paint_face(disk, petal0))
    put(img, R_PETAL[1], paint_petal(1))
    put(img, R_BRACT, paint_bract())
    put(img, R_LEAF[0], paint_leaf(False))
    put(img, R_LEAF[1], paint_leaf(True))
    return bleed(img)


# ---------------------------------------------------------------------------------------------
# Geometry (Blender Z-up while building; front = +Y). Every part is a grid of vertices.


class Builder:
    def __init__(self):
        self.v, self.uv, self.f, self.attr = [], [], [], []

    def grid(self, pts, uvs, attrs, closed_u=False):
        """pts/uvs/attrs: arrays [nu][nv]; adds quads between neighbors."""
        nu, nv = len(pts), len(pts[0])
        base = len(self.v)
        for i in range(nu):
            for j in range(nv):
                self.v.append(pts[i][j])
                self.uv.append(uvs[i][j])
                self.attr.append(attrs[i][j])
        for i in range(nu - 1):
            for j in range(nv - 1):
                a = base + i * nv + j
                self.f.append((a, a + nv, a + nv + 1, a + 1))
        return base


def reg_uv(reg, s, t):
    """s across [0, 1], t down [0, 1] within an atlas region."""
    u0, v0, w, h = reg
    return (u0 + w * min(max(s, 0.002), 0.998), v0 + h * min(max(t, 0.002), 0.998))


def frame_from(t):
    t = t.normalized()
    side = t.cross(Vector((0, 0, 1)))
    if side.length < 1e-4:
        side = Vector((1, 0, 0))
    side.normalize()
    nrm = side.cross(t).normalized()
    return side, nrm


class Plant:
    def __init__(self, H, nod, head_r, leaves, seed, young=False):
        self.H, self.nod, self.R, self.nleaves, self.seed, self.young = H, nod, head_r, leaves, seed, young
        rnd = random.Random(seed)
        self.lean = Vector((rnd.uniform(-0.04, 0.04), rnd.uniform(-0.03, 0.02), 0))
        self.wob = rnd.uniform(0, 6.28)
        self.neck = 0.16 if not young else 0.1
        self.rnd = rnd

    def tangent(self, s):
        """Stem tangent: nearly vertical, a gentle S-wobble, then the neck arcs forward."""
        ang = 0.0
        if s > 1 - self.neck:
            k = (s - (1 - self.neck)) / self.neck
            ang = self.nod * k * k * (3 - 2 * k) * 0.82
        w = 0.03 * math.sin(s * 5.5 + self.wob)
        return Vector((math.sin(w) + self.lean.x, math.sin(ang) + self.lean.y * s, math.cos(ang))).normalized()

    def centerline(self, n=200):
        pts = [Vector((0, 0, 0))]
        for i in range(n):
            s = (i + 0.5) / n
            pts.append(pts[-1] + self.tangent(s) * (self.H / n))
        return pts

    def at(self, s):
        cl = self._cl
        x = s * (len(cl) - 1)
        i = min(int(x), len(cl) - 2)
        return cl[i].lerp(cl[i + 1], x - i)

    def radius(self, s):
        r0, r1 = (0.024, 0.012) if not self.young else (0.016, 0.008)
        return r0 + (r1 - r0) * s ** 0.8 + 0.004 * math.exp(-((s - 0.02) / 0.03) ** 2)

    def build_far(self):
        """~10 triangles: crossed stem quads, two leaf cards, one face card."""
        self._cl = self.centerline()
        b = Builder()
        rnd = random.Random(self.seed * 31 + 7)
        top = self.at(0.97)
        for ang in (0.0, math.pi / 2):
            side = Vector((math.cos(ang), math.sin(ang), 0)) * 0.022
            pts = [[Vector((0, 0, 0)) - side, top - side], [Vector((0, 0, 0)) + side, top + side]]
            uvs = [[reg_uv(R_STEM, 0, 1), reg_uv(R_STEM, 0, 0)], [reg_uv(R_STEM, 1, 1), reg_uv(R_STEM, 1, 0)]]
            ats = [[(0, 0, 0, 0.6, 0), (0, 0.97, 0, 1, 0)], [(0, 0, 0, 0.6, 0), (0, 0.97, 0, 1, 0)]]
            b.grid(pts, uvs, ats)
        for k, s in enumerate((0.35, 0.6)):
            az = 0.7 + k * 2.6
            out = Vector((math.cos(az), math.sin(az), 0))
            across = Vector((-out.y, out.x, 0))
            base = self.at(s)
            L = 0.42 - 0.12 * k
            tip = base + out * L + Vector((0, 0, -0.08))
            pts = [[base - across * L * 0.45, tip - across * L * 0.45], [base + across * L * 0.45, tip + across * L * 0.45]]
            uvs = [[reg_uv(R_LEAF[0], 0, 1), reg_uv(R_LEAF[0], 0, 0)], [reg_uv(R_LEAF[0], 1, 1), reg_uv(R_LEAF[0], 1, 0)]]
            ats = [[(1, s, 0.2, 0.8, k * 0.5), (1, s, 1, 0.8, k * 0.5)], [(1, s, 0.2, 0.8, k * 0.5), (1, s, 1, 0.8, k * 0.5)]]
            b.grid(pts, uvs, ats)
        # Face card perpendicular to the head axis.
        self.head(Builder(), 2, rnd)  # sets pivot / axis
        side, up = frame_from(self.axis)
        c = self.pivot + self.axis * (0.045 * self.R / 0.15 + 0.01)
        rc = self.R * 0.96 / 0.58
        pts = [[c - side * rc - up * rc, c - side * rc + up * rc], [c + side * rc - up * rc, c + side * rc + up * rc]]
        uvs = [[reg_uv(R_FACE, 0, 0), reg_uv(R_FACE, 0, 1)], [reg_uv(R_FACE, 1, 0), reg_uv(R_FACE, 1, 1)]]
        ats = [[(2, 1, 0, 1, 0)] * 2, [(2, 1, 0, 1, 0)] * 2]
        b.grid(pts, uvs, ats)
        return b

    def build(self, lod):
        if lod == 3:
            return self.build_far()
        self._cl = self.centerline()
        b = Builder()
        rnd = random.Random(self.seed * 31 + 7)
        nR, nS = [(10, 44), (4, 10), (3, 6)][lod]
        # --- stem
        pts, uvs, ats = [], [], []
        for j in range(nR + 1):
            th = j / nR * 2 * math.pi
            col_p, col_uv, col_a = [], [], []
            for i in range(nS + 1):
                s = i / nS
                c = self.at(s)
                side, nrm = frame_from(self.tangent(s))
                r = self.radius(s)
                col_p.append(c + (side * math.cos(th) + nrm * math.sin(th)) * r)
                col_uv.append(reg_uv(R_STEM, j / nR, 1 - s))
                col_a.append((0, s, 0, 0.5 + 0.5 * s, 0))
            pts.append(col_p)
            uvs.append(col_uv)
            ats.append(col_a)
        b.grid(pts, uvs, ats)
        self.leaves(b, lod, rnd)
        self.head(b, lod, rnd)
        return b

    def leaves(self, b, lod, rnd):
        n = self.nleaves
        keep = range(n) if lod < 2 else [i for i in range(n) if i % 2 == 1 or i == 0]
        ga, gb = [(11, 9), (3, 2), (2, 2)][lod]
        for i in keep:
            t = i / max(n - 1, 1)
            s = 0.08 + 0.74 * t ** 0.92
            # Opposite pairs low on the stem, then alternate on a spiral.
            az = (math.pi * (i % 2) + 0.4 * i) if i < 4 else (i * 2.4 + 0.3)
            old = i < 2 and not self.young
            L = (0.46 - 0.27 * t) * (0.85 + 0.3 * rnd.random()) * (0.75 if self.young else 1.0)
            W = L * 0.92
            pet = L * (0.45 - 0.15 * t)
            base = self.at(s)
            out = Vector((math.cos(az), math.sin(az), 0))
            phase = rnd.random()
            # Petiole: rises steeply, then arcs over into the blade.
            rise = math.radians(55 - 15 * t)
            p0 = base + out * self.radius(s)
            pdir = (out * math.cos(rise) + Vector((0, 0, math.sin(rise)))).normalized()
            if lod < 2:
                pts, uvs, ats = [], [], []
                nP = [6, 2][lod]
                for j in range(4):
                    th = j / 3 * 2 * math.pi
                    cp, cu, ca = [], [], []
                    for k in range(nP + 1):
                        q = k / nP
                        d = (pdir * (1 - q) + (out * math.cos(-0.2) + Vector((0, 0, math.sin(-0.2)))) * q).normalized()
                        c = p0 + pdir * pet * q + (d - pdir) * pet * q * 0.4
                        side, nrm = frame_from(d)
                        rr = 0.005 * (1 - 0.4 * q)
                        cp.append(c + (side * math.cos(th) + nrm * math.sin(th)) * rr)
                        cu.append(reg_uv(R_STEM, j / 3, 0.3))
                        ca.append((1, s, q * 0.25, 0.7, phase))
                    pts.append(cp)
                    uvs.append(cu)
                    ats.append(ca)
                b.grid(pts, uvs, ats)
            tip0 = p0 + pdir * pet * 0.92
            # Blade frame: along (droops progressively), across, and up.
            pitch0 = math.radians(18 - 30 * (1 - t)) if not old else math.radians(-25)
            droop = (0.9 + 0.8 * (1 - t)) * (1.6 if old else 1.0)
            across = Vector((-out.y, out.x, 0))
            twist = rnd.uniform(-0.25, 0.25)
            fold = (0.22 if not old else 0.1) * (1 if lod < 2 else 0.5)
            pts, uvs, ats = [], [], []
            for jb in range(gb + 1):
                bb = jb / gb * 2 - 1
                cp, cu, ca = [], [], []
                pos = tip0.copy()
                for ia in range(ga + 1):
                    a = ia / ga
                    pitch = pitch0 - droop * a * a
                    along = out * math.cos(pitch) + Vector((0, 0, math.sin(pitch)))
                    up = across.cross(along).normalized()
                    if ia > 0:
                        prev_a = (ia - 1) / ga
                        pp = pitch0 - droop * ((a + prev_a) / 2) ** 2
                        pos = pos + (out * math.cos(pp) + Vector((0, 0, math.sin(pp)))) * (L / ga)
                    hw = float(leaf_halfwidth(np.array(min(a, 0.999)))) if lod == 2 else 1.0
                    x = bb * W * 0.5 * (1 if lod < 2 else max(hw, 0.25))
                    ax = across * math.cos(twist * a) + up * math.sin(twist * a)
                    wave = 0.012 * math.sin(a * 14 + bb * 3 + phase * 6) * abs(bb) * (lod < 2)
                    p = pos + ax * x + up * (abs(bb) * W * 0.5 * fold + wave)
                    cp.append(p)
                    reg = R_LEAF[1] if old else R_LEAF[0]
                    cu.append(reg_uv(reg, bb * 0.5 + 0.5, 1 - a))
                    flut = min(1.0, 0.25 + a * 0.8 + abs(bb) * 0.3)
                    ca.append((1, s, flut, 0.65 + 0.35 * min(1, abs(bb) + a), phase))
                pts.append(cp)
                uvs.append(cu)
                ats.append(ca)
            b.grid(pts, uvs, ats)

    def head(self, b, lod, rnd):
        top = self.at(1.0)
        axis = self.tangent(1.0)
        # The head turns a little further than the neck: the face looks forward and down.
        axis = (axis + Vector((0, 0.35 * math.sin(self.nod), -0.15 * math.sin(self.nod)))).normalized()
        self.pivot, self.axis = top, axis
        side, up = frame_from(axis)
        R = self.R
        back = 0.045 * R / 0.15
        center = top + axis * back
        dome = 0.022 * R / 0.15
        # --- disk (front face)
        nr, nt = [(10, 40), (2, 12), (1, 10)][lod]
        pts, uvs, ats = [], [], []
        for j in range(nt + 1):
            th = j / nt * 2 * math.pi
            cp, cu, ca = [], [], []
            for i in range(nr + 1):
                rr = (i / nr) ** 0.9 if i else 0.0
                dirv = side * math.cos(th) + up * math.sin(th)
                p = center + dirv * rr * R + axis * dome * (1 - rr * rr)
                cp.append(p)
                cu.append(reg_uv(R_DISK, 0.5 + 0.5 * rr * math.cos(th), 0.5 + 0.5 * rr * math.sin(th)))
                ca.append((2, 1.0, 0.0, 1.0, 0.0))
            pts.append(cp)
            uvs.append(cu)
            ats.append(ca)
        b.grid(pts, uvs, ats)
        # --- calyx: a shallow cup from the neck to the rim, behind the disk
        nc, ntc = [(4, 40), (1, 12), (1, 8)][lod]
        pts, uvs, ats = [], [], []
        for j in range(ntc + 1):
            th = j / ntc * 2 * math.pi
            cp, cu, ca = [], [], []
            for i in range(nc + 1):
                q = i / nc
                dirv = side * math.cos(th) + up * math.sin(th)
                rr = 0.012 + (R * 1.02 - 0.012) * math.sin(q * math.pi / 2)
                p = top + axis * (back * q ** 1.6) + dirv * rr
                cp.append(p)
                cu.append(reg_uv(R_CALYX, j / ntc, q))
                ca.append((4, 1.0, 0.0, 0.6 + 0.4 * q, 0.0))
            pts.append(list(reversed(cp)))
            uvs.append(list(reversed(cu)))
            ats.append(list(reversed(ca)))
        b.grid(pts, uvs, ats)
        # --- bracts (phyllaries): pointed green leaflets curling back from the rim
        if lod < 1:
            rings = [(22, 0.55, 0.0), (22, 0.45, 0.5)]
            for count, ln, off in rings:
                for k in range(count):
                    th = (k + off) / count * 2 * math.pi + rnd.uniform(-0.05, 0.05)
                    dirv = side * math.cos(th) + up * math.sin(th)
                    tang = dirv.cross(axis).normalized()
                    L = R * ln * rnd.uniform(0.8, 1.1)
                    nA = [4, 2][lod]
                    pts, uvs, ats = [], [], []
                    ph = rnd.random()
                    for jb in range(2):
                        bb = jb * 2 - 1
                        cp, cu, ca = [], [], []
                        for ia in range(nA + 1):
                            a = ia / nA
                            curl = 0.6 + 1.1 * a
                            d = dirv * math.cos(curl) - axis * math.sin(curl)
                            p = center + dirv * R * 0.9 - axis * back * 0.3 + d * L * a + tang * bb * 0.11 * L * (1 - a * 0.5)
                            cp.append(p)
                            cu.append(reg_uv(R_BRACT, bb * 0.5 + 0.5, 1 - a))
                            ca.append((4, 1.0, a * 0.3, 0.75, ph))
                        pts.append(cp)
                        uvs.append(cu)
                        ats.append(ca)
                    b.grid(pts, uvs, ats)
        # --- ray petals: two staggered rings
        rings = [(34, 1.0, 0.0, 0), (21, 0.9, 0.5, 1)] if lod == 0 else ([(26, 1.0, 0.0, 0)] if lod == 1 else [(21, 1.0, 0.0, 0)])
        na, nb = [(6, 3), (2, 1), (1, 1)][lod]
        plen = R * (0.72 if not self.young else 0.8)
        for count, lscale, off, tex in rings:
            for k in range(count):
                th = (k + off + rnd.uniform(-0.12, 0.12)) / count * 2 * math.pi
                dirv = side * math.cos(th) + up * math.sin(th)
                tang = dirv.cross(axis).normalized()
                L = plen * lscale * rnd.uniform(0.85, 1.12)
                Wp = L * 0.44
                lift = rnd.uniform(-0.1, 0.25) + (0.25 if self.young else -0.05) - 0.12 * tex
                arch = rnd.uniform(0.25, 0.6)
                twist = rnd.uniform(-0.35, 0.35)
                cup = rnd.uniform(0.15, 0.3)
                ph = rnd.random()
                pts, uvs, ats = [], [], []
                for jb in range(nb + 1):
                    bb = jb / nb * 2 - 1
                    cp, cu, ca = [], [], []
                    for ia in range(na + 1):
                        a = ia / na
                        ang = lift - arch * a * a
                        d = dirv * math.cos(ang) + axis * math.sin(ang)
                        w = tang * math.cos(twist * a) + axis * math.sin(twist * a)
                        base = center + dirv * R * 0.96 - axis * (0.006 + 0.004 * tex)
                        p = base + d * L * a + w * bb * Wp * 0.5 + axis * cup * abs(bb) * Wp * 0.5 * (1 - a * 0.3)
                        cp.append(p)
                        cu.append(reg_uv(R_PETAL[tex], bb * 0.5 + 0.5, 1 - a))
                        ca.append((3, 1.0, a, 0.75 + 0.25 * a, ph))
                    pts.append(cp)
                    uvs.append(cu)
                    ats.append(ca)
                b.grid(pts, uvs, ats)


VARIANTS = [
    # height, nod (radians from vertical), head radius, leaves, seed, young
    dict(H=2.05, nod=math.radians(100), head_r=0.155, leaves=14, seed=1),
    dict(H=1.85, nod=math.radians(85), head_r=0.14, leaves=13, seed=2),
    dict(H=1.6, nod=math.radians(70), head_r=0.125, leaves=12, seed=3),
    dict(H=1.35, nod=math.radians(38), head_r=0.095, leaves=10, seed=4, young=True),
]


def to_mesh(name, b):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in b.v], [], b.f)
    me.validate()
    uvl = me.uv_layers.new(name="UV")
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            uvl.data[li].uv = (b.uv[vi][0], 1 - b.uv[vi][1])
    names = ["part", "sat", "flut", "ao", "phase"]
    for k, nm in enumerate(names):
        at = me.attributes.new(nm, "FLOAT", "POINT")
        at.data.foreach_set("value", [a[k] for a in b.attr])
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def export_object(ob):
    """Triangulates and packs one variant/LOD: returns (vertex records, indices)."""
    me = ob.data
    me.calc_loop_triangles()
    n = len(me.vertices)
    co = np.zeros(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nr = np.zeros(n * 3, np.float32)
    me.vertex_normals.foreach_get("vector", nr)
    nr = nr.reshape(-1, 3)
    uv = np.zeros((n, 2), np.float32)
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            uv[vi] = me.uv_layers["UV"].data[li].uv
    at = {}
    for nm in ["part", "sat", "flut", "ao", "phase"]:
        a = np.zeros(n, np.float32)
        me.attributes[nm].data.foreach_get("value", a)
        at[nm] = a
    tris = np.zeros(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tris)
    # Z-up (front +Y) -> Y-up (front +Z).
    pos = np.stack([co[:, 0], co[:, 2], co[:, 1]], 1)
    nrm = np.stack([nr[:, 0], nr[:, 2], nr[:, 1]], 1)
    tris = tris.reshape(-1, 3)[:, [0, 2, 1]]  # the axis swap mirrors: restore winding
    v = np.zeros(n, dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
    v["p"][:, :3] = pos
    v["p"][:, 3] = at["part"]
    v["n"][:, :3] = np.clip(np.round(nrm * 127), -127, 127)
    v["t"] = np.stack([uv[:, 0], 1 - uv[:, 1]], 1)
    v["e"] = np.clip(np.round(np.stack([at["sat"], at["flut"], at["ao"], at["phase"]], 1) * 255), 0, 255)
    return v, tris.reshape(-1)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    atlas = paint_atlas()
    albedo_path = os.path.join(OUT, "albedo.png")
    save_png(atlas[::-1], albedo_path)
    vbytes, ibytes, variants = [], [], []
    vbase = ioff = 0
    preview = []
    for vi, cfg in enumerate(VARIANTS):
        plant = Plant(cfg["H"], cfg["nod"], cfg["head_r"], cfg["leaves"], cfg["seed"], cfg.get("young", False))
        lods = []
        for lod in range(4):
            b = plant.build(lod)
            ob = to_mesh(f"sunflower_{vi}_lod{lod}", b)
            v, idx = export_object(ob)
            vbytes.append(v.tobytes())
            ibytes.append((idx + vbase).astype("<u4").tobytes())
            lods.append({"firstIndex": ioff, "indexCount": int(idx.size)})
            print(f"[sunflower] variant {vi} lod {lod}: {idx.size // 3} tris, {len(v)} verts", flush=True)
            vbase += len(v)
            ioff += idx.size
            if lod == 0:
                preview.append(ob)
            else:
                ob.hide_render = True
        piv = plant.pivot
        ax = plant.axis
        variants.append({
            "height": round(cfg["H"], 3),
            # Y-up: top of the stem (head pivot) and the head's facing axis.
            "pivot": [round(piv.x, 4), round(piv.z, 4), round(piv.y, 4)],
            "axis": [round(ax.x, 4), round(ax.z, 4), round(ax.y, 4)],
            "lods": lods,
        })
    with open(os.path.join(OUT, "sunflower.bin"), "wb") as f:
        f.write(b"".join(vbytes))
        f.write(b"".join(ibytes))
    manifest = {
        "vertexCount": vbase,
        "indexCount": ioff,
        "vertexBytes": vbase * 20,
        "variants": variants,
        "credit": "Modeled in Blender (tools/blender/model_sunflower.py)",
    }
    json.dump(manifest, open(os.path.join(OUT, "sunflower.json"), "w"), indent=2)
    if PREVIEW:
        render_preview(preview, albedo_path)


def render_preview(obs, albedo_path):
    """A small studio render of the four variants side by side (for checking the model)."""
    scene = bpy.context.scene
    mat = bpy.data.materials.new("sunflower")
    mat.use_nodes = True
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.abspath(albedo_path))
    bsdf = nt.nodes["Principled BSDF"]
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.7
    try:
        bsdf.inputs["Subsurface Weight"].default_value = 0.15
    except KeyError:
        pass
    mat.surface_render_method = "DITHERED"
    for i, ob in enumerate(obs):
        ob.data.materials.append(mat)
        ob.location.x = (i - 1.5) * 0.9
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4.0
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(55), 0, math.radians(200))
    scene.collection.objects.link(so)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.62, 0.75, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    scene.world = world
    cam = bpy.data.cameras.new("cam")
    cam.lens = 50
    co = bpy.data.objects.new("cam", cam)
    co.location = (0.3, 4.4, 1.45)
    direction = Vector((0, 0, 1.15)) - co.location
    co.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 1000
    scene.render.filepath = os.path.abspath(PREVIEW)
    bpy.ops.render.render(write_still=True)
    print("[sunflower] preview", PREVIEW, flush=True)


main()
