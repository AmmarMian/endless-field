"""
Models, rigs and animates the game's birds in Blender, and bakes their animation for the GPU:

  blender -b --factory-startup --python tools/blender/model_birds.py -- <out dir> \
      [--species swallow,sparrow,kingfisher,hawk] [--preview <dir>] [--blend <file.blend>]

Every species is built from the same parts, sized by its own parameters (SPECIES below):

  body     metaballs (breast, belly, rump, neck, head) remeshed into one smooth skin
  bill     a lofted cone of elliptical sections (stout, wide-gaped, dagger or hooked)
  wings    a lofted airfoil from deep inside the flank to the tip: planform stations give the
           leading edge, chord and thickness; the hand's trailing edge can be scalloped, and
           broad-winged birds get separate primary "fingers"
  tail     a fanned sheet (square, notched or forked) and, for the swallow, two streamers
  legs     tapered tarsi and four toes each (three forward, one back)

and the same skeleton (14 bones): body (root, at the body's centre), neck, head, tail, and
upperarm / forearm / hand / leg / foot on each side. Skin weights are assigned from the
geometry (span along the wing, position along the body), two bones per vertex at most.

Animations are Blender actions keyed on the pose bones (Bezier interpolation, so every key is
eased); the clips are authored as key poses below (CLIPS). Each clip is then baked at its own
frame rate to skinning matrices (bone pose * inverse bind) for every bone and frame.

Output per species (Y-up, facing +Z, real size in meters; the game scales it):
  <species>.bin   vertices (24 bytes each), then uint32 indices, then the baked matrices
                  (per frame, per bone: 3 rows x 4 as float16)
  <species>.json  counts, bone names, clips {start, frames, fps, loop}, perchHeight (body
                  origin to the soles in the perching pose)

Vertex layout: p float16x4 (xyz, part: 0 body 1 bill 2 eye 3 wing 4 tail 5 leg);
n snorm8x4 (normal, 1 on upper surfaces); t float16x2 (wing: span 0..1 and chord 0..1;
tail: across -1..1 and along 0..1); e unorm8x4 (sRGB albedo, AO); j uint8x4 (bone 0, bone 1,
weight of bone 0 * 255, gloss * 255).
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector, Euler, noise

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


ONLY = opt("--species", "swallow,sparrow,kingfisher,hawk").split(",")
PREVIEW = opt("--preview")
BLEND = opt("--blend")
CLIPS_ONLY = opt("--clips", "").split(",") if opt("--clips") else None
os.makedirs(OUT, exist_ok=True)

BONES = ["body", "neck", "head", "tail", "upperarm.R", "forearm.R", "hand.R", "upperarm.L", "forearm.L", "hand.L", "leg.R", "foot.R", "leg.L", "foot.L"]
BI = {b: i for i, b in enumerate(BONES)}

# ---------------------------------------------------------------------------------------
# Species. Lengths are in "units" of the body length L (bill base to rump), so one design
# language covers a 14 cm sparrow and a 50 cm buzzard. Blender axes: x right, y forward, z up.

SPECIES = {
    "swallow": dict(
        L=0.12,
        girth=0.92, head=1.0, headPos=(0.38, 0.075), neck=0.0,
        bill=dict(len=0.07, w=0.1, d=0.045, hook=0.0),
        shoulder=(0.1, 0.15, 0.06),
        # Arm (upper + fore) and the long hand: scythe-shaped, swept, pointed.
        upperarm=0.12, forearm=0.26,
        # (span from the shoulder, leading edge y, chord, thickness)
        wing=[(-0.1, 0.03, 0.36, 0.09), (0.0, 0.03, 0.36, 0.075), (0.12, 0.03, 0.34, 0.06), (0.38, 0.0, 0.28, 0.045),
              (0.62, -0.1, 0.24, 0.032), (0.88, -0.25, 0.18, 0.022), (1.1, -0.42, 0.1, 0.014), (1.25, -0.56, 0.02, 0.008)],
        scallop=(9, 0.06), fingers=0,
        tail=dict(len=0.36, w0=0.07, w1=0.2, fork=0.45, streamers=0.72, spot=True),
        leg=0.1, toe=0.08, legw=0.012,
        perchPitch=12, clips="swallow",
    ),
    "sparrow": dict(
        L=0.11,
        girth=1.12, head=1.1, headPos=(0.36, 0.09), neck=0.0,
        bill=dict(len=0.11, w=0.09, d=0.085, hook=0.0),
        shoulder=(0.13, 0.13, 0.08),
        upperarm=0.16, forearm=0.3,
        wing=[(-0.1, 0.03, 0.4, 0.1), (0.0, 0.03, 0.4, 0.085), (0.16, 0.03, 0.42, 0.07), (0.46, 0.0, 0.42, 0.05),
              (0.66, -0.06, 0.36, 0.035), (0.82, -0.14, 0.26, 0.024), (0.92, -0.22, 0.14, 0.014), (0.96, -0.28, 0.05, 0.008)],
        scallop=(8, 0.16), fingers=0,
        tail=dict(len=0.5, w0=0.08, w1=0.17, fork=0.06, streamers=0.0, spot=False),
        leg=0.17, toe=0.12, legw=0.016,
        perchPitch=22, clips="sparrow",
        fold=dict(upperarm=(-46, 0, -82), forearm=(0, 0, 34), hand=(-6, 0, -40), arm_loc=(0, -0.025, 0.035)),
    ),
    "kingfisher": dict(
        L=0.12,
        girth=1.1, head=1.32, headPos=(0.36, 0.1), neck=0.0,
        bill=dict(len=0.38, w=0.07, d=0.075, hook=0.0),
        shoulder=(0.13, 0.12, 0.08),
        upperarm=0.15, forearm=0.27,
        wing=[(-0.1, 0.03, 0.4, 0.1), (0.0, 0.03, 0.4, 0.085), (0.15, 0.03, 0.42, 0.07), (0.42, 0.0, 0.4, 0.05),
              (0.6, -0.05, 0.34, 0.035), (0.76, -0.12, 0.24, 0.024), (0.86, -0.19, 0.13, 0.014), (0.9, -0.24, 0.05, 0.008)],
        scallop=(8, 0.1), fingers=0,
        tail=dict(len=0.24, w0=0.07, w1=0.12, fork=-0.05, streamers=0.0, spot=False),
        leg=0.08, toe=0.1, legw=0.014,
        perchPitch=30, clips="kingfisher",
        fold=dict(upperarm=(-46, 0, -82), forearm=(0, 0, 34), hand=(-6, 0, -40), arm_loc=(0, -0.025, 0.035)),
    ),
    "hawk": dict(
        L=0.36,
        girth=1.08, head=0.95, headPos=(0.38, 0.07), neck=0.02,
        bill=dict(len=0.12, w=0.07, d=0.085, hook=0.5),
        shoulder=(0.12, 0.13, 0.07),
        upperarm=0.2, forearm=0.55,
        # Broad, long, with fingered tips (the primaries spread apart at the end).
        wing=[(-0.1, 0.03, 0.5, 0.1), (0.0, 0.03, 0.5, 0.08), (0.2, 0.03, 0.52, 0.065), (0.6, 0.0, 0.52, 0.05),
              (0.95, -0.03, 0.46, 0.035), (1.2, -0.06, 0.36, 0.024), (1.32, -0.08, 0.28, 0.016)],
        scallop=(0, 0.0), fingers=5,
        tail=dict(len=0.55, w0=0.1, w1=0.3, fork=-0.12, streamers=0.0, spot=False),
        leg=0.2, toe=0.12, legw=0.03,
        perchPitch=20, clips="hawk",
    ),
}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def catmull(points, n):
    """Resamples control rows (k values each) to n rows along a Catmull-Rom spline."""
    P = np.array(points, float)
    m = len(P)
    out = []
    for i in range(n):
        u = i / (n - 1) * (m - 1)
        k = min(int(u), m - 2)
        t = u - k
        p0, p1, p2, p3 = P[max(k - 1, 0)], P[k], P[k + 1], P[min(k + 2, m - 1)]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


def mesh_obj(name, verts, faces, part, attrs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    set_attrs(ob, part, attrs or {})
    return ob


def set_attrs(ob, part, attrs):
    me = ob.data
    n = len(me.vertices)
    a = me.attributes.new("part", "INT", "POINT")
    a.data.foreach_set("value", np.full(n, part, np.int32))
    for k in ("fs", "fc", "up"):
        arr = np.asarray(attrs.get(k, np.zeros(n)), np.float32)
        at = me.attributes.new(k, "FLOAT", "POINT")
        at.data.foreach_set("value", arr)


# ---------------------------------------------------------------------------------------
# Parts


def build_body(sp, L):
    g, h = sp["girth"], sp["head"]
    hy, hz = sp["headPos"]
    mb = bpy.data.metaballs.new("skin")
    mb.resolution = mb.render_resolution = 0.022 * L
    mb.threshold = 0.6
    ob = bpy.data.objects.new("body", mb)
    bpy.context.scene.collection.objects.link(ob)

    def el(co, r, s):
        e = mb.elements.new(type="ELLIPSOID")
        e.co = Vector(co) * L
        e.radius = r * L
        e.stiffness = 2.0
        e.size_x, e.size_y, e.size_z = s

    el((0, 0.1, 0.0), 0.22 * g, (0.95, 1.45, 0.9))
    el((0, -0.15, -0.005), 0.18 * g, (0.85, 1.55, 0.8))
    el((0, -0.36, 0.015), 0.09 * g, (0.85, 1.9, 0.55))
    el((0, 0.26 + sp["neck"] * 0.5, 0.035 + hz * 0.3), 0.13 * g, (0.95, 1.1, 0.95))
    el((0, hy + sp["neck"], hz), 0.165 * h, (1.0, 1.08, 0.92))
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.convert(target="MESH")
    ob = bpy.context.view_layer.objects.active
    for mod, kw in (("REMESH", dict(mode="VOXEL", voxel_size=0.018 * L)), ("SMOOTH", dict(iterations=10, factor=0.7)), ("DECIMATE", dict(ratio=0.28))):
        m = ob.modifiers.new(mod, mod)
        for k, v in kw.items():
            setattr(m, k, v)
        bpy.ops.object.modifier_apply(modifier=m.name)
    set_attrs(ob, 0, {})
    return ob


def head_centre(sp, L):
    return Vector((0, sp["headPos"][0] + sp["neck"], sp["headPos"][1])) * L


def build_bill(sp, L, skin):
    b = sp["bill"]
    hc = head_centre(sp, L)
    r = 0.165 * sp["head"] * L
    # Base: just inside the front of the head, a little below its centre.
    ok, base, nrm, _ = skin.closest_point_on_mesh(hc + Vector((0, r * 1.2, -r * 0.18)))
    base = base - Vector((0, r * 0.15, 0))
    rings, segs = 9, 10
    verts, faces = [], []
    for i in range(rings):
        u = i / (rings - 1)
        taper = (1 - u) ** 0.85
        w = b["w"] * L * 0.5 * taper + 0.0004
        d = b["d"] * L * 0.5 * taper + 0.0003
        # Hooked bills curl down at the tip.
        y = base.y + b["len"] * L * u
        z = base.z - b["hook"] * b["d"] * L * (u ** 3) * 0.9 - 0.15 * b["d"] * L * u
        for k in range(segs):
            a = 2 * math.pi * k / segs
            verts.append((base.x + w * math.cos(a), y, z + d * math.sin(a)))
    tip = len(verts)
    verts.append((base.x, base.y + b["len"] * L * 1.04, base.z - b["hook"] * b["d"] * L * 0.95 - 0.15 * b["d"] * L))
    for i in range(rings - 1):
        for k in range(segs):
            a, c = i * segs + k, i * segs + (k + 1) % segs
            faces.append((a, c, c + segs, a + segs))
    for k in range(segs):
        faces.append(((rings - 1) * segs + k, (rings - 1) * segs + (k + 1) % segs, tip))
    return mesh_obj("bill", verts, faces, 1)


def build_eyes(sp, L, skin):
    hc = head_centre(sp, L)
    r = 0.165 * sp["head"] * L
    bm = bmesh.new()
    for s in (-1, 1):
        ok, co, n, _ = skin.closest_point_on_mesh(hc + Vector((s * r * 1.2, r * 0.45, r * 0.22)))
        rr = r * 0.2
        res = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=7, radius=rr)
        bmesh.ops.translate(bm, verts=res["verts"], vec=co - n * rr * 0.45)
    me = bpy.data.meshes.new("eyes")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("eyes", me)
    bpy.context.scene.collection.objects.link(ob)
    set_attrs(ob, 2, {})
    return ob


def wing_frame(sp, L):
    """Shoulder (Blender, right side) and the resampled planform: rows of (s, le, chord, th)."""
    sx, sy, sz = sp["shoulder"]
    rows = catmull(sp["wing"], 44)
    return Vector((sx, sy, sz)) * L, rows


def build_wing(sp, L, side):
    S, rows = wing_frame(sp, L)
    span = sp["wing"][-1][0]
    m = 9
    verts, faces, fs, fc, up = [], [], [], [], []
    n_sc, depth = sp["scallop"]
    wrist = sp["upperarm"] + sp["forearm"]
    for s, le, chord, th in rows:
        # The hand's trailing edge: one rounded feather tip after another.
        if n_sc and s > wrist:
            u = (s - wrist) / (span - wrist)
            g = 1 - (2 * ((u * n_sc) % 1.0) - 1) ** 2
            chord = chord * (1 - depth * (1 - math.sqrt(max(g, 0))) * min(1, u * 4))
        for k in range(2 * m):
            upper = k < m
            q = (k / (m - 1)) if upper else ((2 * m - 1 - k) / (m - 1))
            q = q * q
            half = 5 * th * (0.2969 * math.sqrt(q) - 0.126 * q - 0.3516 * q * q + 0.2843 * q ** 3 - 0.1036 * q ** 4)
            camber = th * 0.8 * 4 * q * (1 - q)
            z = camber + (half if upper else -half)
            verts.append((side * (S.x + s * L), S.y + (le - q * chord) * L, S.z + z * L))
            fs.append(max(0.0, s) / span)
            fc.append(q)
            up.append(1.0 if upper else 0.0)
    n = 2 * m
    for i in range(len(rows) - 1):
        for k in range(n):
            a, b, c, d = i * n + k, i * n + (k + 1) % n, (i + 1) * n + (k + 1) % n, (i + 1) * n + k
            faces.append((a, b, c, d) if side > 0 else (d, c, b, a))
    faces.append(tuple(range(n))[::-1] if side > 0 else tuple(range(n)))
    last = (len(rows) - 1) * n
    faces.append(tuple(range(last, last + n)) if side > 0 else tuple(range(last, last + n))[::-1])
    ob = mesh_obj(f"wing{side}", verts, faces, 3, dict(fs=fs, fc=fc, up=up))
    parts = [ob]
    if sp["fingers"]:
        parts.append(build_fingers(sp, L, side, rows))
    return parts


def blade(verts, faces, attrs, root, direction, length, width, th, rows=8, droop=0.0):
    """A tapered, cambered feather (closed thin solid) from `root` along `direction`."""
    d = Vector(direction).normalized()
    up = Vector((0, 0, 1))
    side = d.cross(up).normalized()
    base = len(verts)
    for i in range(rows + 1):
        u = i / rows
        w = width * (0.55 + 0.45 * math.sin(math.pi * min(1.0, 0.25 + u * 0.8))) * (1 - u ** 6 * 0.8) * 0.5
        for j, (a, zz) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            p = root + d * (u * length) + side * (a * w) + up * (zz * th * (1 - u * 0.7) * 0.5 + 0.3 * th * (1 - a * a) - droop * u * u * length)
            verts.append(tuple(p))
            attrs["fs"].append(u)
            attrs["fc"].append(a * 0.5 + 0.5)
            attrs["up"].append(1.0 if zz >= 0 else 0.0)
    for i in range(rows):
        for j in range(4):
            a = base + i * 4 + j
            b = base + i * 4 + (j + 1) % 4
            faces.append((a, b, b + 4, a + 4))
    tip = base + rows * 4
    faces.append((tip, tip + 1, tip + 2, tip + 3))
    faces.append((base + 3, base + 2, base + 1, base))


def build_fingers(sp, L, side, rows):
    S, _ = wing_frame(sp, L)
    span = sp["wing"][-1][0]
    verts, faces, attrs = [], [], {"fs": [], "fc": [], "up": []}
    k = sp["fingers"]
    tip = rows[-1]
    for i in range(k):
        f = i / (k - 1)
        s = span - 0.04 - f * 0.1
        r = catmull(sp["wing"], 200)
        row = r[np.argmin(np.abs(r[:, 0] - s))]
        root = Vector((side * (S.x + s * L), S.y + (row[1] - row[2] * (0.15 + f * 0.6)) * L, S.z + 0.004 * L))
        ang = math.radians(4 - f * 38)
        d = Vector((side * math.cos(ang), math.sin(ang), 0.02))
        blade(verts, faces, attrs, root, d, (0.3 - f * 0.08) * L, 0.075 * L, 0.012 * L, droop=-0.02)
    if side < 0:
        faces = [f[::-1] for f in faces]
    attrs["fs"] = [0.85 + 0.15 * v for v in attrs["fs"]]
    return mesh_obj(f"fingers{side}", verts, faces, 3, attrs)


def build_tail(sp, L):
    t = sp["tail"]
    root = Vector((0, -0.34, 0.03)) * L
    cols, rws = 13, 10
    verts, faces, fs, fc, up = [], [], [], [], []
    for j in range(rws + 1):
        v = j / rws
        for i in range(cols):
            u = i / (cols - 1) * 2 - 1
            length = t["len"] * (1 + t["fork"] * (abs(u) ** 2.2) * 2.2)
            hw = t["w0"] + (t["w1"] - t["w0"]) * v
            y = -v * length
            x = u * hw
            z = 0.02 * (1 - u * u) * hw * 2 - 0.05 * v * v * length
            verts.append((root.x + x * L, root.y + y * L, root.z + z * L))
            fs.append(u)
            fc.append(v)
            up.append(1.0)
    for j in range(rws):
        for i in range(cols - 1):
            a = j * cols + i
            faces.append((a, a + 1, a + cols + 1, a + cols))
    obs = [mesh_obj("tail", verts, faces, 4, dict(fs=fs, fc=fc, up=up))]
    if t["streamers"]:
        verts, faces, attrs = [], [], {"fs": [], "fc": [], "up": []}
        for s in (-1, 1):
            length = t["len"] * (1 + t["fork"] * 2.2)
            start = root + Vector((s * t["w1"] * 0.9, -length * 0.85, -0.02)) * L
            d = Vector((s * 0.12, -1, -0.04))
            blade(verts, faces, attrs, start, d, t["streamers"] * L, 0.03 * L, 0.008 * L, rows=10)
        attrs["fs"] = [(-1 if i < len(verts) // 2 else 1) * 1.0 for i in range(len(verts))]
        obs.append(mesh_obj("streamers", verts, faces, 4, attrs))
    return obs


def leg_points(sp, L, side):
    hip = Vector((side * 0.075, -0.02, -0.08)) * L
    ankle = hip + Vector((0, 0, -(sp["leg"] + 0.12))) * L
    return hip, ankle


def build_legs(sp, L):
    verts, faces = [], []
    seg = 6

    def tube(a, b, r0, r1):
        base = len(verts)
        d = (b - a).normalized()
        o = d.orthogonal().normalized()
        o2 = d.cross(o)
        for i, (p, r) in enumerate(((a, r0), (b, r1))):
            for k in range(seg):
                an = 2 * math.pi * k / seg
                verts.append(tuple(p + (o * math.cos(an) + o2 * math.sin(an)) * r))
        for k in range(seg):
            faces.append((base + k, base + (k + 1) % seg, base + seg + (k + 1) % seg, base + seg + k))
        faces.append(tuple(base + seg + k for k in range(seg)))

    w = sp["legw"] * L
    for s in (-1, 1):
        hip, ankle = leg_points(sp, L, s)
        tube(hip, ankle, w * 1.4, w)
        t = sp["toe"] * L
        for dx, dy in ((-0.45, 0.9), (0.0, 1.0), (0.45, 0.9), (0.0, -0.6)):
            tip = ankle + Vector((s * dx * t * 0.6, dy * t, -w * 0.4))
            tube(ankle + Vector((0, 0, w * 0.3)), tip, w * 0.75, w * 0.35)
    return mesh_obj("legs", verts, faces, 5)


# ---------------------------------------------------------------------------------------
# Skeleton and weights


def bone_layout(sp, L):
    S, rows = wing_frame(sp, L)
    def le_at(s):
        r = rows[np.argmin(np.abs(rows[:, 0] - s))]
        return (r[1] - 0.12 * r[2]) * L
    ua, fa = sp["upperarm"], sp["forearm"]
    span = sp["wing"][-1][0]
    hc = head_centre(sp, L)
    lay = {
        "body": (Vector((0, -0.05, 0)) * L, Vector((0, 0.15, 0)) * L),
        "neck": (Vector((0, 0.2, 0.02)) * L, hc - Vector((0, 0.02, 0.0)) * L),
        "head": (hc - Vector((0, 0.02, 0.0)) * L, hc + Vector((0, 0.18, 0)) * L),
        "tail": (Vector((0, -0.33, 0.03)) * L, Vector((0, -0.33 - sp["tail"]["len"], 0.03)) * L),
    }
    for side, sfx in ((1, "R"), (-1, "L")):
        p = [Vector((side * (S.x + s * L), S.y + le_at(s), S.z)) for s in (0.0, ua, ua + fa, span)]
        lay[f"upperarm.{sfx}"] = (p[0], p[1])
        lay[f"forearm.{sfx}"] = (p[1], p[2])
        lay[f"hand.{sfx}"] = (p[2], p[3])
        hip, ankle = leg_points(sp, L, side)
        lay[f"leg.{sfx}"] = (hip, ankle)
        lay[f"foot.{sfx}"] = (ankle, ankle + Vector((0, sp["toe"] * L, 0)))
    return lay


PARENT = {"neck": "body", "head": "neck", "tail": "body", "leg.R": "body", "leg.L": "body", "foot.R": "leg.R", "foot.L": "leg.L"}
for _s in "RL":
    PARENT[f"upperarm.{_s}"] = "body"
    PARENT[f"forearm.{_s}"] = f"upperarm.{_s}"
    PARENT[f"hand.{_s}"] = f"forearm.{_s}"


def build_armature(sp, L):
    arm = bpy.data.armatures.new("rig")
    ob = bpy.data.objects.new("rig", arm)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    lay = bone_layout(sp, L)
    for name in BONES:
        eb = arm.edit_bones.new(name)
        eb.head, eb.tail = lay[name]
        # Predictable local axes (see clip authoring): wings, body, neck, head, tail keep Z
        # up; legs and feet keep Z forward.
        eb.align_roll(Vector((0, 1, 0)) if name.startswith(("leg", "foot")) else Vector((0, 0, 1)))
    for name, parent in PARENT.items():
        arm.edit_bones[name].parent = arm.edit_bones[parent]
        arm.edit_bones[name].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in ob.pose.bones:
        pb.rotation_mode = "YZX" if pb.name.startswith(("upperarm", "forearm", "hand", "leg", "foot")) else "YXZ"
    return ob


def weights(sp, L, pos, part, fs):
    """Two bones per vertex: (bone0, bone1, w0)."""
    n = len(pos)
    W = np.zeros((n, len(BONES)), np.float32)
    x, y, z = pos[:, 0] / L, pos[:, 1] / L, pos[:, 2] / L
    side = np.where(x >= 0, "R", "L")
    # Body skin, bill, eyes: along the body (tail - body - neck - head).
    hy = sp["headPos"][0] + sp["neck"]
    h = smoothstep(hy - 0.2, hy - 0.08, y)
    k = smoothstep(hy - 0.1, hy - 0.02, y)
    t = 1 - smoothstep(-0.4, -0.26, y)
    core = (part == 0) | (part == 1) | (part == 2)
    head_only = (part == 1) | (part == 2)
    W[core, BI["tail"]] = t[core]
    W[core, BI["body"]] = ((1 - t) * (1 - h))[core]
    W[core, BI["neck"]] = (h * (1 - k))[core]
    W[core, BI["head"]] = (h * k)[core]
    W[head_only] = 0
    W[head_only, BI["head"]] = 1
    # Wings: along the span.
    S, _ = wing_frame(sp, L)
    s = np.abs(x) - S.x / L
    ua, fa = sp["upperarm"], sp["forearm"]
    a = smoothstep(-0.06, 0.03, s)
    b = smoothstep(ua - 0.05, ua + 0.05, s)
    c = smoothstep(ua + fa - 0.07, ua + fa + 0.05, s)
    wing = part == 3
    for i in np.nonzero(wing)[0]:
        sfx = side[i]
        W[i, BI["body"]] = 1 - a[i]
        W[i, BI[f"upperarm.{sfx}"]] = a[i] * (1 - b[i])
        W[i, BI[f"forearm.{sfx}"]] = a[i] * b[i] * (1 - c[i])
        W[i, BI[f"hand.{sfx}"]] = a[i] * b[i] * c[i]
    tail = part == 4
    W[tail, BI["tail"]] = 1
    # Legs: tarsus to the leg, toes to the foot (below the ankle).
    legs = part == 5
    _, ankle = leg_points(sp, L, 1)
    foot = z < (ankle.z / L + sp["legw"] * 0.5)
    for i in np.nonzero(legs)[0]:
        W[i, BI[("foot." if foot[i] else "leg.") + side[i]]] = 1
    order = np.argsort(-W, axis=1)
    b0, b1 = order[:, 0], order[:, 1]
    w0 = W[np.arange(n), b0]
    w1 = W[np.arange(n), b1]
    tot = np.maximum(w0 + w1, 1e-6)
    return b0, b1, w0 / tot


# ---------------------------------------------------------------------------------------
# Paint (unit coordinates)


def paint(name, sp, L, pos, nrm, part, fs, fc, upf):
    x, y, z = pos[:, 0] / L, pos[:, 1] / L, pos[:, 2] / L
    n = len(pos)
    fine = np.array([noise.noise(Vector((px * 60, py * 60, pz * 60))) for px, py, pz in pos / L])
    upn = np.clip((nrm[:, 2] + 0.1) * 2.2, 0, 1)
    col = np.zeros((n, 3))
    gloss = np.zeros(n)
    hy = sp["headPos"][0] + sp["neck"]
    head = y > hy - 0.12
    face = head & (y > hy + 0.05)
    throat = head & (z < sp["headPos"][1] - 0.03) & (nrm[:, 2] < 0.3)
    span = sp["wing"][-1][0]
    wrist = (sp["upperarm"] + sp["forearm"]) / span
    upper = upf > 0.5
    c3 = lambda r, g, b: np.array([r, g, b])
    if name == "swallow":
        BLUE, RED, CREAM = c3(0.035, 0.05, 0.12), c3(0.55, 0.12, 0.05), c3(0.9, 0.82, 0.68)
        body = CREAM * (1 - upn[:, None]) + BLUE * upn[:, None]
        body = np.where((throat | (face & (z > sp["headPos"][1] + 0.02)))[:, None], RED, body)
        band = (y > 0.18) & (y < 0.3) & (nrm[:, 2] < 0.2) & ~throat
        body = np.where(band[:, None], BLUE * 1.15, body)
        wingc = np.where(upper[:, None], BLUE * (1.0 + 0.35 * (1 - np.clip(fc * 1.6, 0, 1)))[:, None] * 1.0, np.where((fc < 0.45)[:, None], CREAM * 0.85, c3(0.2, 0.2, 0.22)))
        tailc = BLUE * 0.9 + 0.01
        spot = (np.abs(fs) > 0.25) & (np.abs(fs) < 0.95) & (fc > 0.55) & (fc < 0.8)
        tailc = np.where(spot[:, None], c3(0.88, 0.87, 0.82), tailc)
        legc, billc = c3(0.12, 0.08, 0.07), c3(0.03, 0.03, 0.03)
        gl_body, gl_wing = upn, upper * 1.0
    elif name == "sparrow":
        BROWN, CHEST, BUFF, DARK = c3(0.42, 0.25, 0.12), c3(0.55, 0.32, 0.15), c3(0.78, 0.7, 0.58), c3(0.12, 0.09, 0.07)
        streak = np.clip(np.sin(x * 70 + np.sin(y * 30) * 2) * 3 - 1.2, 0, 1)
        back = BROWN * (1 - 0.6 * streak[:, None]) + DARK * 0.0
        body = BUFF * (1 - upn[:, None]) + back * upn[:, None]
        crown = head & (z > sp["headPos"][1] + 0.05)
        cheek = head & ~crown & (np.abs(x) > 0.08) & (nrm[:, 2] > -0.3)
        body = np.where(cheek[:, None], c3(0.9, 0.88, 0.82), body)
        spot = cheek & (np.hypot(y - hy, z - sp["headPos"][1]) < 0.06)
        body = np.where(spot[:, None], DARK, body)
        body = np.where(crown[:, None], CHEST, body)
        bib = throat & (y > hy - 0.02)
        body = np.where(bib[:, None], DARK, body)
        flight = fc > 0.42
        wup = np.where(flight[:, None], DARK * 1.4 + BROWN * 0.5 * (1 - fc[:, None]), CHEST * (0.9 + 0.2 * fine[:, None]))
        bar = (fc > 0.3) & (fc < 0.38) & (fs < wrist)
        wup = np.where(bar[:, None], c3(0.92, 0.9, 0.84), wup)
        wingc = np.where(upper[:, None], wup, BUFF * 0.8)
        tailc = BROWN * 0.7
        legc, billc = c3(0.55, 0.42, 0.36), c3(0.12, 0.1, 0.1)
        gl_body, gl_wing = upn * 0, upper * 0.0
    elif name == "kingfisher":
        BLUE, TEAL, ORANGE, WHITE = c3(0.05, 0.3, 0.6), c3(0.1, 0.55, 0.62), c3(0.85, 0.38, 0.08), c3(0.95, 0.92, 0.85)
        body = ORANGE * (1 - upn[:, None]) + BLUE * upn[:, None]
        stripe = (y < -0.05) & (np.abs(x) < 0.06) & (nrm[:, 2] > 0.4)
        body = np.where(stripe[:, None], TEAL * 1.4, body)
        cheekO = head & (np.abs(x) > 0.1) & (z > sp["headPos"][1] - 0.06) & (z < sp["headPos"][1] + 0.02) & (y > hy - 0.04)
        body = np.where(cheekO[:, None], ORANGE, body)
        neckW = head & (np.abs(x) > 0.1) & (y < hy - 0.02) & (z > sp["headPos"][1] - 0.08)
        body = np.where(neckW[:, None], WHITE, body)
        body = np.where((throat & (y > hy))[:, None], WHITE, body)
        wingc = np.where(upper[:, None], BLUE * (1.0 + 0.3 * fine[:, None]) * np.where((fc < 0.35)[:, None], 1.3, 0.8), ORANGE * 0.6 + 0.1)
        tailc = BLUE * 0.9
        legc, billc = c3(0.85, 0.3, 0.15), c3(0.04, 0.04, 0.05)
        gl_body, gl_wing = upn, upper * 1.0
    else:  # hawk (a common buzzard)
        BROWN, PALE, DARK = c3(0.32, 0.2, 0.11), c3(0.85, 0.78, 0.66), c3(0.1, 0.07, 0.05)
        barred = 0.5 + 0.5 * np.sin(y * 40 + fine * 2)
        body = BROWN * upn[:, None] + (PALE * barred[:, None] + BROWN * (1 - barred[:, None])) * (1 - upn[:, None])
        band = (y > -0.05) & (y < 0.1) & (nrm[:, 2] < 0)
        body = np.where(band[:, None], BROWN, body)
        under = np.where((fc < 0.4)[:, None], BROWN * 0.9, PALE * 0.9)
        under = np.where(((fs > wrist) & (fs < wrist + 0.08) & (fc < 0.3))[:, None], DARK, under)
        under = np.where((fc > 0.9)[:, None] | (fs > 0.92)[:, None], DARK, under)
        wingc = np.where(upper[:, None], BROWN * (0.9 + 0.25 * fine[:, None]), under)
        bars = 0.5 + 0.5 * np.sin(fc * 40)
        tailc = BROWN * (0.7 + 0.5 * bars[:, None])
        legc, billc = c3(0.85, 0.7, 0.2), c3(0.15, 0.15, 0.17)
        gl_body, gl_wing = upn * 0, upper * 0.0
    tailc = np.broadcast_to(tailc, (n, 3))
    body = body * (0.94 + 0.1 * fine[:, None])
    col[part == 0] = body[part == 0]
    col[part == 1] = billc
    col[part == 2] = (0.012, 0.01, 0.01)
    col[part == 3] = wingc[part == 3]
    col[part == 4] = tailc[part == 4]
    col[part == 5] = legc
    gloss[part == 0] = gl_body[part == 0]
    gloss[part == 3] = gl_wing[part == 3]
    gloss[part == 4] = 1.0 if name in ("swallow", "kingfisher") else 0.0
    return np.clip(col, 0, 1), np.clip(gloss, 0, 1)


# ---------------------------------------------------------------------------------------
# Animation. A pose is a dict of bone -> (rx, ry, rz) degrees in the bone's local axes, with
# the left side mirrored from the right (rx, -ry, -rz) unless given explicitly:
#   wings (Y along the bone, Z up): rx + raises the tip, rz + sweeps it forward, ry + lifts
#     the leading edge (rotation order: twist, sweep, then raise)
#   body / neck / head (Y forward): rx + noses up, rz + turns left, ry + rolls
#   tail (Y backward): rx + lifts it; "tail_s" spreads the fan (scale across)
#   legs (Y down, Z forward): rx + swings the foot forward; "leg_s" scales the leg
#   "body_loc": the body's offset (units of L), for bobbing and crouching

ZERO = (0.0, 0.0, 0.0)


def pose(**kw):
    p = {b: ZERO for b in ("body", "neck", "head", "tail", "upperarm", "forearm", "hand", "leg", "foot")}
    p.update(body_loc=ZERO, arm_loc=ZERO, tail_s=1.0, leg_s=1.0)
    p.update(kw)
    return p


def mix(a, b, t):
    out = {}
    for k in set(a) | set(b):
        va, vb = a.get(k, ZERO if k not in ("tail_s", "leg_s") else 1.0), b.get(k, ZERO if k not in ("tail_s", "leg_s") else 1.0)
        if isinstance(va, tuple):
            out[k] = tuple(x + (y - x) * t for x, y in zip(va, vb))
        else:
            out[k] = va + (vb - va) * t
    return out


def over(p, **kw):
    q = dict(p)
    q.update(kw)
    return q


def species_poses(sp):
    pp = sp["perchPitch"]
    # Legs tucked back under the belly in flight (and smaller: hidden in the feathers).
    tucked = dict(leg=(-95, 0, 0), foot=(-60, 0, 0), leg_s=0.55)
    glide = pose(upperarm=(4, 2, 0), forearm=(0, 0, -6), hand=(-2, -3, -10), tail=(2, 0, 0), tail_s=0.85, **tucked)
    # Folded: the wing turns on edge (leading edge up), sweeps back along the flank and
    # the hand folds a little forward over the arm, the tips crossing over the rump.
    fold = sp.get("fold", dict(upperarm=(-34, 0, -86), forearm=(0, 0, 10), hand=(-4, 0, -8), arm_loc=(0, -0.03, 0.07)))
    perch = pose(body=(pp, 0, 0), neck=(-pp * 0.4, 0, 0), head=(-pp * 0.5, 0, 0), tail=(-pp * 0.7, 0, 0), tail_s=0.8,
                 leg=(-pp * 0.9 + 6, 0, 2), foot=(-pp * 0.1 - 4, 0, 0), body_loc=(0, 0, 0), **fold)
    return glide, fold, perch, tucked


def flap_pose(sp, glide, phase, amp=1.0):
    """The wingbeat at `phase` (0 top .. 1): a long powered downstroke with the wing fully
    spread and the hand lagging then flicking through, a quick upstroke with the hand swept
    back and the wrist flexed."""
    # Downstroke takes ~58% of the cycle.
    w = phase + 0.08 * math.sin(2 * math.pi * phase) / (2 * math.pi) * 2
    c = math.cos(2 * math.pi * w)
    s = math.sin(2 * math.pi * w)
    up = max(0.0, -s)  # upstroke half (sin < 0 after the bottom)
    arm = 8 + 50 * c * amp
    hand_lag = 22 * s * amp
    sweep = -38 * up * amp
    p = over(glide,
             upperarm=(arm, 4 * s * amp, 6 * s * amp),
             forearm=(-6 * up * amp, -6 * up * amp, sweep * 0.6),
             hand=(hand_lag - 8 * up * amp, 10 * up * amp - 4 * s * amp, sweep * 1.1),
             body_loc=(0, 0, -0.035 * c * amp),
             body=(2 * s * amp, 0, 0),
             head=(-2 * s * amp, 0, 0),
             tail=(4 - 4 * s * amp, 0, 0))
    return p


def clip_defs(name, sp):
    """Clips as key poses: name -> (fps, loop, [(time s, pose)])."""
    glide, fold, perch, tucked = species_poses(sp)
    pp = sp["perchPitch"]
    clips = {}
    # Flap: one beat, sampled every 1/16 cycle (the runtime plays it at the wingbeat rate).
    clips["flap"] = (16, True, [(i / 16, flap_pose(sp, glide, i / 16)) for i in range(17)])
    # Glide: the wings trim and the head looks about.
    clips["glide"] = (20, True, [
        (0.0, glide),
        (1.0, over(glide, upperarm=(7, 3, 2), hand=(-5, -2, -14), tail=(-2, 0, 0), head=(0, 0, 10), tail_s=0.95)),
        (2.0, over(glide, upperarm=(2, 0, -2), hand=(0, -4, -8), tail=(4, 0, 0), head=(-4, 0, -6))),
        (3.0, glide),
    ])
    # Stoop: wings tucked back, a dart.
    tuck = over(glide, upperarm=(-4, 10, -42), forearm=(0, 0, -30), hand=(0, -6, -28), tail_s=0.55, tail=(0, 0, 0))
    clips["tuck"] = (10, True, [(0.0, tuck), (0.2, tuck)])
    if name == "hawk":
        return clips
    # Landing (touchdown at 0.9 s): flare up with the wings forward and tail fanned, two
    # braking beats, feet reaching forward; touch down with the wings high, then fold.
    flare = over(glide, body=(38, 0, 0), neck=(-25, 0, 0), head=(-15, 0, 0), upperarm=(30, 18, 26), forearm=(0, 0, 10), hand=(10, 8, 8),
                 tail=(-28, 0, 0), tail_s=1.6, leg=(30, 0, 4), foot=(10, 0, 0), leg_s=1.0)
    brake_up = over(flare, upperarm=(68, 20, 30), hand=(30, 12, 0), body=(48, 0, 0))
    brake_dn = over(flare, upperarm=(-30, 12, 34), hand=(-18, 4, 12), body=(44, 0, 0))
    touch = over(perch, upperarm=(70, 30, 10), forearm=(0, 0, 0), hand=(14, 10, -6), tail=(-30, 0, 0), tail_s=1.4,
                 body=(pp + 14, 0, 0), body_loc=(0, -0.02, -0.06), head=(-pp * 0.5 - 10, 0, 0))
    settle = over(perch, body_loc=(0, 0, -0.025), head=(-pp * 0.5 + 8, 0, 0), tail=(-pp * 0.7 + 14, 0, 0))
    clips["land"] = (30, False, [
        (0.0, glide),
        (0.3, flare),
        (0.45, brake_up),
        (0.6, brake_dn),
        (0.75, brake_up),
        (0.9, touch),
        (1.05, over(touch, upperarm=(40, 60, -40), hand=(6, 0, -10))),
        (1.25, over(perch, body_loc=(0, 0, -0.02), upperarm=(-20, 0, -80))),
        (1.45, settle),
        (1.7, perch),
    ])
    # Taking off: crouch, spring, the wings snap up and beat hard.
    crouch = over(perch, body=(pp - 6, 0, 0), body_loc=(0, 0, -0.05), head=(-pp * 0.5 + 6, 0, 0), leg=(-pp * 0.9 + 18, 0, 2))
    clips["takeoff"] = (30, False, [
        (0.0, perch),
        (0.12, crouch),
        (0.24, over(flap_pose(sp, glide, 0.0), body=(20, 0, 0), leg=(-40, 0, 0), foot=(-50, 0, 0), leg_s=0.9, tail_s=1.4)),
        (0.36, over(flap_pose(sp, glide, 0.5), body=(10, 0, 0), leg=(-80, 0, 0), leg_s=0.7)),
        (0.48, flap_pose(sp, glide, 0.0)),
    ])
    # Perched: breathing, looking about, a tail flick and a wing shuffle (loops at 9 s).
    breathe = lambda p, k: over(p, body_loc=(0, 0, 0.006 * k))
    look_l = over(perch, head=(-pp * 0.5 + 4, 4, 55), neck=(-pp * 0.4, 0, 15))
    look_r = over(perch, head=(-pp * 0.5 - 6, -6, -60), neck=(-pp * 0.4, 0, -15))
    look_up = over(perch, head=(-pp * 0.5 + 22, 10, 20))
    flick = over(perch, tail=(-pp * 0.7 + 22, 0, 0), tail_s=1.3)
    shuffle = over(perch, upperarm=(-18, 0, -78), arm_loc=(0, -0.03, 0.09), body_loc=(0, 0, 0.012))
    clips["perch"] = (20, True, [
        (0.0, perch), (0.8, breathe(perch, 1)), (1.3, look_l), (2.3, look_l), (2.6, breathe(perch, 0)),
        (3.0, flick), (3.15, perch), (3.9, look_r), (4.8, breathe(look_r, 1)), (5.2, perch),
        (5.9, look_up), (6.6, look_up), (7.0, perch), (7.6, shuffle), (7.9, perch), (9.0, perch),
    ])
    # Preening: the head turns back into the shoulder and nibbles, then a fluff and a shake.
    reach = over(perch, neck=(-10, 0, 70), head=(-40, 30, 80), upperarm=(-14, 0, -74))
    nib = over(reach, head=(-55, 30, 88))
    fluff = over(perch, body_loc=(0, 0, 0.02), tail_s=1.25)
    clips["preen"] = (20, False, [
        (0.0, perch), (0.5, reach), (0.7, nib), (0.85, reach), (1.0, nib), (1.15, reach), (1.4, nib), (1.6, reach),
        (2.0, perch), (2.2, over(fluff, head=(-pp * 0.5, 0, 25))), (2.3, over(fluff, head=(-pp * 0.5, 0, -25))),
        (2.4, over(fluff, head=(-pp * 0.5, 0, 20))), (2.5, over(fluff, head=(-pp * 0.5, 0, -15))), (2.8, perch),
    ])
    if name in ("sparrow",):
        # On the ground: a two-footed hop and a peck.
        clips["hop"] = (30, False, [
            (0.0, perch), (0.07, crouch), (0.16, over(perch, leg=(-pp * 0.9 - 20, 0, 0), foot=(-30, 0, 0), upperarm=(14, 0, -58), arm_loc=(0, -0.02, 0.04), tail=(-pp * 0.7 + 18, 0, 0))),
            (0.3, over(perch, leg=(-pp * 0.9 + 14, 0, 2))), (0.38, crouch), (0.5, perch),
        ])
        peck = over(perch, body=(-18, 0, 0), neck=(-35, 0, 0), head=(-30, 0, 0), tail=(10, 0, 0), leg=(-pp * 0.9 + 26, 0, 2))
        clips["peck"] = (30, False, [
            (0.0, perch), (0.15, over(peck, head=(-10, 0, 0))), (0.22, peck), (0.3, over(peck, head=(-12, 0, 0))), (0.37, peck), (0.55, perch),
        ])
    return clips


def apply_pose(rig, p):
    for pb in rig.pose.bones:
        base, _, sfx = pb.name.partition(".")
        val = p.get(pb.name, p.get(base, ZERO))
        if sfx == "L" and pb.name not in p:
            val = (val[0], -val[1], -val[2])
        pb.rotation_euler = Euler([math.radians(v) for v in val], pb.rotation_mode)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    L = rig["L"]
    bl = p.get("body_loc", ZERO)
    # Pose locations are in the bone's local frame (body: x right, y forward, z up... in
    # bone space Y is along the bone and Z is up): local (x, y, z) = world (x, y, z) here.
    rig.pose.bones["body"].location = (bl[0] * L, bl[1] * L, bl[2] * L)
    rig.pose.bones["tail"].scale = (p.get("tail_s", 1.0), 1, 1)
    # Upper arms (local Y outward along the bone, Z up): lifted over the back when folded.
    al = p.get("arm_loc", ZERO)
    for s in "RL":
        rig.pose.bones[f"upperarm.{s}"].location = (al[0] * L, al[1] * L, al[2] * L)
    for s in "RL":
        k = p.get("leg_s", 1.0)
        rig.pose.bones[f"leg.{s}"].scale = (k, k, k)


def author(rig, clips):
    """Creates one Blender action per clip, keyed at its key poses."""
    actions = {}
    scene = bpy.context.scene
    for cname, (fps, loop, keys) in clips.items():
        rig.animation_data_create()
        act = bpy.data.actions.new(cname)
        act.use_fake_user = True
        rig.animation_data.action = act
        for t, p in keys:
            f = round(t * fps)
            apply_pose(rig, p)
            for pb in rig.pose.bones:
                pb.keyframe_insert("rotation_euler", frame=f)
                pb.keyframe_insert("location", frame=f)
                pb.keyframe_insert("scale", frame=f)
        actions[cname] = (act, fps, loop, round(keys[-1][0] * fps))
    return actions


def bake(rig, actions):
    """Skinning matrices (game axes) per clip frame and bone."""
    C = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    inv_bind = {b.name: b.matrix_local.inverted() for b in rig.data.bones}
    mats, meta, start = [], {}, 0
    scene = bpy.context.scene
    for cname, (act, fps, loop, frames) in actions.items():
        rig.animation_data.action = act
        for f in range(frames + 1):
            scene.frame_set(f)
            for b in BONES:
                m = C @ rig.pose.bones[b].matrix @ inv_bind[b] @ C
                mats.append([m[r][c] for r in range(3) for c in range(4)])
        meta[cname] = {"start": start, "frames": frames, "fps": fps, "loop": loop}
        start += frames + 1
    return np.array(mats, np.float32), meta


# ---------------------------------------------------------------------------------------


def collect(obs):
    dg = bpy.context.evaluated_depsgraph_get()
    pos, nrm, idx, part, fs, fc, up = [], [], [], [], [], [], []
    base = 0
    for ob in obs:
        me = ob.evaluated_get(dg).to_mesh()
        me.calc_loop_triangles()
        n = len(me.vertices)
        co = np.zeros(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        nr = np.zeros(n * 3, np.float32)
        me.vertex_normals.foreach_get("vector", nr)
        tris = np.zeros(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tris)
        pos.append(co.reshape(-1, 3))
        nrm.append(nr.reshape(-1, 3))
        idx.append(tris.reshape(-1, 3) + base)
        for key, lst, dt in (("part", part, np.int32), ("fs", fs, np.float32), ("fc", fc, np.float32), ("up", up, np.float32)):
            a = np.zeros(n, dt)
            me.attributes[key].data.foreach_get("value", a)
            lst.append(a)
        base += n
        ob.evaluated_get(dg).to_mesh_clear()
    return [np.concatenate(a) for a in (pos, nrm, idx, part, fs, fc, up)]


def build(name):
    reset()
    sp = SPECIES[name]
    L = sp["L"]
    scene = bpy.context.scene
    skin = build_body(sp, L)
    obs = [skin, build_bill(sp, L, skin), build_eyes(sp, L, skin), *build_wing(sp, L, 1), *build_wing(sp, L, -1), *build_tail(sp, L), build_legs(sp, L)]
    pos, nrm, idx, part, fs, fc, up = collect(obs)
    # Upper-surface flag for parts without one: by the normal.
    up = np.where((part == 3) | (part == 4), up, (nrm[:, 2] > 0).astype(np.float32))
    b0, b1, w0 = weights(sp, L, pos, part, fs)
    col, gloss = paint(name, sp, L, pos, nrm, part, fs, fc, up)
    rig = build_armature(sp, L)
    rig["L"] = L
    clips = clip_defs(name, sp)
    actions = author(rig, clips)
    mats, meta = bake(rig, actions)
    # Perch height: the soles (lowest toe vertex) in the perching pose, below the body origin.
    perch_h = 0.0
    if "perch" in meta:
        f = meta["perch"]["start"]
        gp = np.stack([pos[:, 0], pos[:, 2], pos[:, 1]], 1)
        toes = (part == 5) & (pos[:, 2] < leg_points(sp, L, 1)[1].z + sp["legw"] * L)
        ys = []
        for i in np.nonzero(toes)[0]:
            M0 = mats[f * len(BONES) + b0[i]].reshape(3, 4)
            M1 = mats[f * len(BONES) + b1[i]].reshape(3, 4)
            v = np.append(gp[i], 1.0)
            ys.append(w0[i] * (M0 @ v)[1] + (1 - w0[i]) * (M1 @ v)[1])
        perch_h = -float(min(ys)) - sp["legw"] * L * 0.4
    export(name, sp, pos, nrm, idx, part, fs, fc, up, col, gloss, b0, b1, w0, mats, meta, perch_h)
    if PREVIEW or BLEND:
        for ob in obs:
            bpy.data.objects.remove(ob)
        prev = make_preview_mesh(name, pos, idx, col, b0, b1, w0, rig)
        if BLEND:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(BLEND.replace(".blend", f"-{name}.blend")))
        if PREVIEW:
            render_previews(name, sp, rig, prev, actions)


def export(name, sp, pos, nrm, idx, part, fs, fc, up, col, gloss, b0, b1, w0, mats, meta, perch_h):
    ao_body = np.clip(0.62 + 0.38 * (nrm[:, 2] * 0.5 + 0.5), 0, 1)
    ao = np.where(part == 3, 0.85 + 0.15 * np.clip(fs * 3, 0, 1), ao_body)
    gp = np.stack([pos[:, 0], pos[:, 2], pos[:, 1]], 1)
    gn = np.stack([nrm[:, 0], nrm[:, 2], nrm[:, 1]], 1)
    idx = idx[:, [0, 2, 1]]
    v = np.zeros(len(gp), dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4), ("j", "u1", 4)])
    v["p"][:, :3] = gp
    v["p"][:, 3] = part
    v["n"][:, :3] = np.clip(np.round(gn * 127), -127, 127)
    v["n"][:, 3] = np.where(up > 0.5, 127, 0)
    v["t"][:, 0] = fs
    v["t"][:, 1] = fc
    v["e"] = np.clip(np.round(np.concatenate([col, ao[:, None]], 1) * 255), 0, 255)
    v["j"][:, 0] = b0
    v["j"][:, 1] = b1
    v["j"][:, 2] = np.clip(np.round(w0 * 255), 0, 255)
    v["j"][:, 3] = np.clip(np.round(gloss * 255), 0, 255)
    m16 = mats.astype("<f2")
    with open(os.path.join(OUT, f"{name}.bin"), "wb") as f:
        f.write(v.tobytes())
        f.write(idx.reshape(-1).astype("<u4").tobytes())
        f.write(m16.tobytes())
    manifest = {
        "vertexBytes": len(v) * 24,
        "indexCount": int(idx.size),
        "matrixBytes": int(m16.nbytes),
        "bones": BONES,
        "clips": meta,
        "perchHeight": round(perch_h, 5),
        "length": sp["L"],
        "credit": "Modeled, rigged and animated in Blender (tools/blender/model_birds.py)",
    }
    json.dump(manifest, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(f"[{name}] {len(v)} verts, {idx.shape[0]} tris, {len(mats) // len(BONES)} frames, perch {perch_h:.4f}", flush=True)


# ---------------------------------------------------------------------------------------
# Previews: the skinned mesh on the real armature, rendered at sampled frames of every clip.


def make_preview_mesh(name, pos, idx, col, b0, b1, w0, rig):
    me = bpy.data.meshes.new(f"{name}-mesh")
    me.from_pydata([tuple(p) for p in pos], [], [tuple(t) for t in idx])
    attr = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.concatenate([col ** 2.2, np.ones((len(col), 1))], 1).astype(np.float32).ravel())
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    groups = [ob.vertex_groups.new(name=b) for b in BONES]
    for i in range(len(pos)):
        groups[b0[i]].add([i], float(w0[i]), "REPLACE")
        if w0[i] < 0.999:
            groups[b1[i]].add([i], float(1 - w0[i]), "ADD")
    mod = ob.modifiers.new("rig", "ARMATURE")
    mod.object = rig
    ob.parent = rig
    mat = bpy.data.materials.new("plumage")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    ca = nt.nodes.new("ShaderNodeVertexColor")
    ca.layer_name = "Col"
    nt.links.new(ca.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    me.materials.append(mat)
    return ob


def render_previews(name, sp, rig, ob, actions):
    from_dir = os.path.abspath(PREVIEW)
    os.makedirs(from_dir, exist_ok=True)
    scene = bpy.context.scene
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(40), 0, math.radians(35))
    scene.collection.objects.link(so)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.62, 0.75, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.1
    scene.world = world
    cam = bpy.data.cameras.new("cam")
    cam.lens = 60
    co = bpy.data.objects.new("cam", cam)
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 360
    scene.render.resolution_y = 270
    L = sp["L"]
    span = sp["wing"][-1][0] + sp["shoulder"][0]
    dist = max(span, 1.0) * L * 4.2
    views = {"side": (Vector((1.0, 0.25, 0.25)), Vector((0, -0.05, 0))), "front": (Vector((0.35, 1.0, 0.35)), Vector((0, 0, 0))),
             "top": (Vector((0.25, -0.5, 1.0)), Vector((0, -0.05, 0)))}
    for view, (d, tgt) in views.items():
        loc = d.normalized() * dist
        co.location = loc
        co.rotation_euler = (tgt * L - loc).to_track_quat("-Z", "Y").to_euler()
        for cname, (act, fps, loop, frames) in actions.items():
            if (view != "side" and cname not in ("flap", "perch", "land")) or (CLIPS_ONLY and cname not in CLIPS_ONLY):
                continue
            rig.animation_data.action = act
            n = 8 if frames > 8 else frames + 1
            for k in range(n):
                f = round(k * frames / max(n - 1, 1)) if not loop else round(k * frames / n)
                scene.frame_set(f)
                scene.render.filepath = os.path.join(from_dir, f"{name}-{cname}-{view}-{k:02d}.png")
                bpy.ops.render.render(write_still=True)


def main():
    for name in ONLY:
        build(name)


main()
