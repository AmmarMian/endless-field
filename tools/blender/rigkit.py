"""
Shared rigging kit for code-built, skinned and baked creatures (crabs, gulls): the same
pipeline and output format as model_birds.py / model_animals.py (on the animals branch).
Imported by the model_*.py scripts; reads their command line (<out dir> [--preview <dir>]
[--clips a,b] [--blend <file>]).
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector, Euler, Quaternion, noise

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


PREVIEW = opt("--preview")
BLEND = opt("--blend")
CLIPS_ONLY = opt("--clips", "").split(",") if opt("--clips") else None
os.makedirs(OUT, exist_ok=True)
V = Vector


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def catmull(points, n, closed=False):
    """Resamples control points (rows of k values) to n along a Catmull-Rom spline."""
    P = np.array(points, float)
    m = len(P)
    out = []
    for i in range(n):
        if closed:
            u = i / n * m
            k = int(u) % m
            t = u - int(u)
            p0, p1, p2, p3 = P[(k - 1) % m], P[k], P[(k + 1) % m], P[(k + 2) % m]
        else:
            u = i / (n - 1) * (m - 1)
            k = min(int(u), m - 2)
            t = u - k
            p0, p1, p2, p3 = P[max(k - 1, 0)], P[k], P[k + 1], P[min(k + 2, m - 1)]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


def set_attrs(ob, part, attrs=None):
    attrs = attrs or {}
    me = ob.data
    n = len(me.vertices)
    a = me.attributes.new("part", "INT", "POINT")
    a.data.foreach_set("value", np.full(n, part, np.int32))
    for k in ("fs", "fc", "up"):
        at = me.attributes.new(k, "FLOAT", "POINT")
        at.data.foreach_set("value", np.asarray(attrs.get(k, np.zeros(n)), np.float32))


FORCED = []


def mesh_obj(name, verts, faces, part, attrs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    set_attrs(ob, part, attrs)
    return ob


def bm_obj(bm, name, part):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    set_attrs(ob, part)
    return ob


# ---------------------------------------------------------------------------------------
# Shapes


class Skin:
    """Metaball body: ellipsoids and capsules, then one smooth mesh."""

    def __init__(self, res, scale=1.0):
        # Blender polygonizes metaballs no finer than 5 mm: small animals are built `scale`
        # times larger and shrunk back before remeshing.
        self.k = scale
        self.mb = bpy.data.metaballs.new("skin")
        self.mb.resolution = self.mb.render_resolution = res * scale
        self.mb.threshold = 0.6
        self.res = res

    def ball(self, co, r, size=(1, 1, 1), stiff=2.0):
        e = self.mb.elements.new(type="ELLIPSOID")
        e.co = V(co) * self.k
        e.radius = r * self.k
        e.stiffness = stiff
        e.size_x, e.size_y, e.size_z = size
        return e

    def capsule(self, a, b, r, stiff=2.0):
        """A limb segment from a to b (round ends)."""
        a, b = V(a) * self.k, V(b) * self.k
        r = r * self.k
        d = b - a
        e = self.mb.elements.new(type="CAPSULE")
        e.co = (a + b) / 2
        e.radius = r
        e.stiffness = stiff
        e.size_x = d.length / 2
        e.rotation = V((1, 0, 0)).rotation_difference(d.normalized())
        return e

    def limb(self, pts, radii, stiff=2.0):
        for i in range(len(pts) - 1):
            self.capsule(pts[i], pts[i + 1], radii[i], stiff)

    def build(self, voxel, ratio, smooth=8):
        ob = bpy.data.objects.new("skin", self.mb)
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        bpy.ops.object.convert(target="MESH")
        ob = bpy.context.view_layer.objects.active
        if self.k != 1.0:
            ob.scale = (1 / self.k,) * 3
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        for mod, kw in (("REMESH", dict(mode="VOXEL", voxel_size=voxel)), ("SMOOTH", dict(iterations=smooth, factor=0.6)), ("DECIMATE", dict(ratio=ratio))):
            m = ob.modifiers.new(mod, mod)
            for k, v in kw.items():
                setattr(m, k, v)
            bpy.ops.object.modifier_apply(modifier=m.name)
        set_attrs(ob, 0)
        return ob


def surface(skin, target):
    ok, co, n, _ = skin.closest_point_on_mesh(V(target))
    return co, n


def eyes(skin, centre, r, part=2, sink=0.4, segs=(12, 8)):
    """A pair of eyes snapped onto the skin near (±x, y, z)."""
    bm = bmesh.new()
    for s in (-1, 1):
        co, n = surface(skin, V((s * centre[0], centre[1], centre[2])))
        res = bmesh.ops.create_uvsphere(bm, u_segments=segs[0], v_segments=segs[1], radius=r)
        bmesh.ops.translate(bm, verts=res["verts"], vec=co - n * r * sink)
    return bm_obj(bm, "eyes", part)


def sphere(centre, r, part, scale=(1, 1, 1), segs=(10, 7)):
    bm = bmesh.new()
    res = bmesh.ops.create_uvsphere(bm, u_segments=segs[0], v_segments=segs[1], radius=r)
    bmesh.ops.scale(bm, vec=V(scale), verts=res["verts"])
    bmesh.ops.translate(bm, verts=res["verts"], vec=V(centre))
    return bm_obj(bm, "sphere", part)


def tube(path, radii, part, segs=8, name="tube", cap=True):
    """A tube along `path` (points), radius per point; capped ends."""
    pts = [V(p) for p in path]
    verts, faces = [], []
    for i, p in enumerate(pts):
        d = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        o = d.orthogonal().normalized()
        o2 = d.cross(o)
        for k in range(segs):
            a = 2 * math.pi * k / segs
            verts.append(tuple(p + (o * math.cos(a) + o2 * math.sin(a)) * radii[i]))
    for i in range(len(pts) - 1):
        for k in range(segs):
            a, b = i * segs + k, i * segs + (k + 1) % segs
            faces.append((a, b, b + segs, a + segs))
    if cap:
        faces.append(tuple(range(segs))[::-1])
        last = (len(pts) - 1) * segs
        faces.append(tuple(range(last, last + segs)))
    return mesh_obj(name, verts, faces, part)


def leaf(base, direction, normal, length, width, cup, part, rows=9, cols=7, tip_round=0.6, thick=0.0, name="leaf", profile=None):
    """A cupped, tapered blade (ears): base point, direction of the tip, the open side's normal.
    Two-sided (closed thin solid when `thick`); fs = along, fc = across."""
    d = V(direction).normalized()
    nn = V(normal).normalized()
    side = d.cross(nn).normalized()
    nn = side.cross(d).normalized()
    verts, faces, fs, fc, up = [], [], [], [], []
    layers = (1, -1) if thick else (1,)
    for li, sgn in enumerate(layers):
        for i in range(rows + 1):
            u = i / rows
            w = width * (profile(u) if profile else (math.sin(math.pi * min(1.0, 0.15 + u * 0.95)) ** tip_round)) * 0.5
            for j in range(cols):
                v = j / (cols - 1) * 2 - 1
                depth = cup * width * (1 - v * v)
                p = V(base) + d * (u * length) + side * (v * w) - nn * depth + nn * (thick * 0.5 * sgn)
                verts.append(tuple(p))
                fs.append(u)
                fc.append(v * 0.5 + 0.5)
                up.append(1.0 if sgn > 0 else 0.0)
        o = li * (rows + 1) * cols
        for i in range(rows):
            for j in range(cols - 1):
                a = o + i * cols + j
                q = (a, a + 1, a + cols + 1, a + cols)
                faces.append(q if sgn > 0 else q[::-1])
    if thick:
        n1 = (rows + 1) * cols
        for i in range(rows):
            for j in (0, cols - 1):
                a, b = i * cols + j, (i + 1) * cols + j
                q = (a, b, n1 + b, n1 + a)
                faces.append(q if j == 0 else q[::-1])
    return mesh_obj(name, verts, faces, part, dict(fs=fs, fc=fc, up=up))


def fan_wing(root, outline, part, rings=10, side=1, hind=False, lift=0.0, thick=0.0004, name="wing"):
    """A flat wing: rings from the root out to `outline` (closed list of (x, y) offsets in the
    wing plane, right side). fs = how far out (0 root .. 1 margin), fc = around (0..1, +2 hind)."""
    pts = catmull(outline, 40, closed=False)
    verts, faces, fs, fc, up = [], [], [], [], []
    n = len(pts)
    for layer, sgn in enumerate((1, -1)):
        o = len(verts)
        verts.append((root[0], root[1], root[2] + sgn * thick * 0.5))
        fs.append(0.0)
        fc.append(0.5 + (2 if hind else 0))
        up.append(1.0 if sgn > 0 else 0.0)
        for r in range(1, rings + 1):
            t = r / rings
            for k in range(n):
                x, y = pts[k]
                z = root[2] + lift * t * t + sgn * thick * 0.5 * (1 - t * 0.6)
                verts.append((root[0] + side * x * t, root[1] + y * t, z))
                fs.append(t)
                fc.append(k / (n - 1) + (2 if hind else 0))
                up.append(1.0 if sgn > 0 else 0.0)
        def vi(r, k):
            return o if r == 0 else o + 1 + (r - 1) * n + k
        for k in range(n - 1):
            f = (vi(0, 0), vi(1, k), vi(1, k + 1))
            faces.append(f if (sgn > 0) == (side > 0) else f[::-1])
        for r in range(1, rings):
            for k in range(n - 1):
                f = (vi(r, k), vi(r + 1, k), vi(r + 1, k + 1), vi(r, k + 1))
                faces.append(f if (sgn > 0) == (side > 0) else f[::-1])
    return mesh_obj(name, verts, faces, part, dict(fs=fs, fc=fc, up=up))


# ---------------------------------------------------------------------------------------
# Rig


class Rig:
    """Bones: name -> (head, tail, parent, radius). Sagittal bones keep local X = +x."""

    def __init__(self):
        self.bones = {}
        self.order = []

    def bone(self, name, head, tail, parent=None, radius=0.02, mirror=True):
        """Adds a bone; a name ending in .R also adds its .L mirror (x negated)."""
        self.bones[name] = (V(head), V(tail), parent, radius)
        self.order.append(name)
        if mirror and name.endswith(".R"):
            m = lambda p: V((-p[0], p[1], p[2]))
            lp = parent[:-2] + ".L" if parent and parent.endswith(".R") else parent
            self.bones[name[:-2] + ".L"] = (m(head), m(tail), lp, radius)
            self.order.append(name[:-2] + ".L")

    def build(self, wing_like=()):
        arm = bpy.data.armatures.new("rig")
        ob = bpy.data.objects.new("rig", arm)
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode="EDIT")
        for name in self.order:
            h, t, parent, _ = self.bones[name]
            eb = arm.edit_bones.new(name)
            eb.head, eb.tail = h, t
            y = (t - h).normalized()
            if name.split(".")[0] in wing_like:
                # Wings (pointing out sideways): Z up, like the birds' wings.
                eb.align_roll(V((0, 0, 1)))
            else:
                z = V((1, 0, 0)).cross(y)
                if z.length < 1e-4:
                    z = V((0, 0, 1))
                eb.align_roll(z.normalized())
        for name in self.order:
            parent = self.bones[name][2]
            if parent:
                arm.edit_bones[name].parent = arm.edit_bones[parent]
                arm.edit_bones[name].use_connect = False
        bpy.ops.object.mode_set(mode="OBJECT")
        for pb in ob.pose.bones:
            pb.rotation_mode = "YZX" if pb.name.split(".")[0] in wing_like else "XYZ"
        self.ob = ob
        return ob


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    q = a + np.outer(t, ab)
    return np.linalg.norm(p - q, axis=1)


def auto_weights(rig, pos, part, allowed, side_tol):
    """Two bones per vertex from distance to each bone (over its radius). `allowed`: part ->
    list of bone names (None: all). Bones of one side never pull the other side."""
    names = rig.order
    n = len(pos)
    W = np.zeros((n, len(names)))
    for bi, name in enumerate(names):
        h, t, _, r = rig.bones[name]
        d = seg_dist(pos, np.array(h), np.array(t)) / r
        w = 1.0 / (d ** 4 + 1e-3)
        if name.endswith(".R"):
            w = w * (pos[:, 0] > -side_tol)
        if name.endswith(".L"):
            w = w * (pos[:, 0] < side_tol)
        W[:, bi] = w
    for p, lst in allowed.items():
        if lst is None:
            continue
        mask = part == p
        keep = np.zeros(len(names), bool)
        for b in lst:
            for bi, name in enumerate(names):
                if name == b or name.split(".")[0] == b:
                    keep[bi] = True
        W[np.ix_(mask, ~keep)] = 0
    # Parts made for one bone (wings, ears) follow it alone; a side's bone by the side.
    for i, f in enumerate(FORCED):
        if not f:
            continue
        want = f if f in names else f + (".R" if pos[i, 0] >= 0 else ".L")
        W[i] = 0
        W[i, names.index(want)] = 1
    order = np.argsort(-W, axis=1)
    b0, b1 = order[:, 0], order[:, 1]
    w0 = W[np.arange(n), b0]
    w1 = W[np.arange(n), b1]
    return b0, b1, w0 / np.maximum(w0 + w1, 1e-9)


def apply_pose(rig_ob, p, L):
    """Pose: bone (or base name for both sides) -> (rx, ry, rz) degrees; left mirrors right
    (rx, -ry, -rz) unless given. "loc" moves the root (m); "scale": {bone: s or (x, y, z)}."""
    for pb in rig_ob.pose.bones:
        base, _, sfx = pb.name.partition(".")
        if pb.name in p:
            val = p[pb.name]
        else:
            val = p.get(base, (0.0, 0.0, 0.0))
            if sfx == "L":
                val = (val[0], -val[1], -val[2])
        pb.rotation_euler = Euler([math.radians(v) for v in val], pb.rotation_mode)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    root = rig_ob.pose.bones[0]
    loc = p.get("loc", (0, 0, 0))
    # The root's local frame: X right, Y along the bone (forward), Z = X x Y (up).
    root.location = (loc[0], loc[1], loc[2])
    for name, s in p.get("scale", {}).items():
        for pb in rig_ob.pose.bones:
            if pb.name == name or pb.name.split(".")[0] == name:
                pb.scale = (s, s, s) if isinstance(s, (int, float)) else s


def pose(**kw):
    return dict(kw)


def over(p, **kw):
    q = dict(p)
    q.update(kw)
    return q


def author(rig_ob, clips, L):
    actions = {}
    for cname, (fps, loop, keys) in clips.items():
        rig_ob.animation_data_create()
        act = bpy.data.actions.new(cname)
        act.use_fake_user = True
        rig_ob.animation_data.action = act
        for t, p in keys:
            f = round(t * fps)
            apply_pose(rig_ob, p, L)
            for pb in rig_ob.pose.bones:
                for path in ("rotation_euler", "location", "scale"):
                    pb.keyframe_insert(path, frame=f)
        actions[cname] = (act, fps, loop, round(keys[-1][0] * fps))
    return actions


def keyed_cycle(keys, samples=24, period=1.0):
    """A smooth loop through key poses at phases (0..1): every channel follows a periodic
    Catmull-Rom spline through its keys (no holds, no corners). Returns cycle keys."""
    keys = sorted(keys, key=lambda k: k[0])
    chans = set()
    for _, p in keys:
        chans |= set(p)
    def val(p, ch):
        v = p.get(ch)
        if v is None:
            return (1.0, 1.0, 1.0) if ch == "scale" else (0.0, 0.0, 0.0)
        return v
    n = len(keys)
    out = []
    for i in range(samples + 1):
        ph = (i / samples) % 1.0 if i < samples else 0.0
        # Find the segment.
        k = max(j for j in range(n) if keys[j][0] <= ph) if ph >= keys[0][0] else n - 1
        k1 = (k + 1) % n
        t0 = keys[k][0]
        t1 = keys[k1][0] + (1.0 if k1 <= k else 0.0)
        phx = ph + (1.0 if ph < t0 else 0.0)
        t = (phx - t0) / max(t1 - t0, 1e-6)
        pose = {}
        for ch in chans:
            if ch == "scale":
                continue
            P = [np.array(val(keys[(k + d) % n][1], ch), float) for d in (-1, 0, 1, 2)]
            v = 0.5 * ((2 * P[1]) + (-P[0] + P[2]) * t + (2 * P[0] - 5 * P[1] + 4 * P[2] - P[3]) * t * t + (-P[0] + 3 * P[1] - 3 * P[2] + P[3]) * t ** 3)
            pose[ch] = tuple(float(x) for x in v)
        out.append((i / samples * period, pose))
    return out


def scale_keys(keys, k, loc_k=None):
    """The same key poses, smaller (a slower gait from a faster one)."""
    lk = k if loc_k is None else loc_k
    out = []
    for ph, p in keys:
        q = {}
        for ch, v in p.items():
            q[ch] = tuple(x * (lk if ch == "loc" else k) for x in v)
        out.append((ph, q))
    return out


def quad_bound(a=1.0, flight=0.2, extra=None):
    """Key poses for a quadruped's bounding gallop (legs: thigh / shin / hcannon behind,
    upper / fore / fcannon in front; positive turns a leg's foot forward)."""
    e = extra or {}
    def k(**kw):
        d = {n: tuple(x * a for x in v) for n, v in kw.items() if n != "loc"}
        d["loc"] = kw.get("loc", (0, 0, 0))
        d.update(e)
        return d
    return [
        # Gathered: the hind feet planted far forward under the belly, the back arched.
        (0.0, k(body=(-10, 0, 0), chest=(-6, 0, 0), neck=(10, 0, 0), head=(6, 0, 0), thigh=(34, 0, 0), shin=(-24, 0, 0), hcannon=(14, 0, 0),
                upper=(-42, 0, 0), fore=(-14, 0, 0), fcannon=(36, 0, 0), loc=(0, 0, flight * 0.1))),
        # Push off: hind legs drive back, the fronts fold and swing forward.
        (0.2, k(body=(8, 0, 0), chest=(4, 0, 0), neck=(-2, 0, 0), head=(-2, 0, 0), thigh=(-12, 0, 0), shin=(14, 0, 0), hcannon=(-24, 0, 0),
                upper=(18, 0, 0), fore=(30, 0, 0), fcannon=(-95, 0, 0), loc=(0, 0, flight * 0.45))),
        # Stretched out in the air.
        (0.42, k(body=(2, 0, 0), chest=(6, 0, 0), neck=(-8, 0, 0), head=(-4, 0, 0), thigh=(-46, 0, 0), shin=(32, 0, 0), hcannon=(-40, 0, 0),
                 upper=(52, 0, 0), fore=(12, 0, 0), fcannon=(-14, 0, 0), loc=(0, 0, flight))),
        # Front feet land; the hinds fold and swing forward.
        (0.62, k(body=(-6, 0, 0), chest=(-4, 0, 0), neck=(4, 0, 0), head=(2, 0, 0), thigh=(4, 0, 0), shin=(-55, 0, 0), hcannon=(62, 0, 0),
                 upper=(16, 0, 0), fore=(0, 0, 0), fcannon=(0, 0, 0), loc=(0, 0, flight * 0.3))),
        # The fronts sweep back under the chest as the hinds reach forward.
        (0.8, k(body=(-13, 0, 0), chest=(-8, 0, 0), neck=(10, 0, 0), head=(6, 0, 0), thigh=(30, 0, 0), shin=(-38, 0, 0), hcannon=(34, 0, 0),
                upper=(-22, 0, 0), fore=(-6, 0, 0), fcannon=(14, 0, 0), loc=(0, 0, flight * 0.12))),
    ]


def alive(fps, keys, breathe=0.006, chest="chest"):
    """A loop made organic: a smooth periodic spline through its key poses (nothing freezes
    between keys) with breathing on top (the body rising a little, the chest swelling)."""
    period = keys[-1][0]
    phased = [(t / period, p) for t, p in keys[:-1]]
    samples = max(8, round(period * fps))
    out = keyed_cycle(phased, samples, period)
    n = max(1, round(period / 3.2))  # a breath every ~3 s, a whole number per loop
    for t, p in out:
        b = math.sin(2 * math.pi * n * t / period)
        loc = p.get("loc", (0.0, 0.0, 0.0))
        p["loc"] = (loc[0], loc[1], loc[2] + breathe * b)
        c = p.get(chest, (0.0, 0.0, 0.0))
        p[chest] = (c[0] + 1.2 * b, c[1], c[2])
    return (fps, True, out)


def cycle(fn, period, samples):
    """Keys for a loop: `fn(phase 0..1)` sampled `samples` times (and once more at the end)."""
    return [(i / samples * period, fn(i / samples)) for i in range(samples + 1)]


def bake(rig_ob, actions, names):
    C = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    inv_bind = {b.name: b.matrix_local.inverted() for b in rig_ob.data.bones}
    mats, meta, start = [], {}, 0
    scene = bpy.context.scene
    for cname, (act, fps, loop, frames) in actions.items():
        rig_ob.animation_data.action = act
        for f in range(frames + 1):
            scene.frame_set(f)
            for b in names:
                m = C @ rig_ob.pose.bones[b].matrix @ inv_bind[b] @ C
                mats.append([m[r][c] for r in range(3) for c in range(4)])
        meta[cname] = {"start": start, "frames": frames, "fps": fps, "loop": loop}
        start += frames + 1
    return np.array(mats, np.float32), meta


def collect(obs):
    dg = bpy.context.evaluated_depsgraph_get()
    pos, nrm, idx, part, fs, fc, up = [], [], [], [], [], [], []
    global FORCED
    FORCED = []
    base = 0
    for ob in obs:
        me = ob.evaluated_get(dg).to_mesh()
        FORCED += [ob.get("bone", "")] * len(me.vertices)
        me.calc_loop_triangles()
        n = len(me.vertices)
        co = np.zeros(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        nr = np.zeros(n * 3, np.float32)
        me.vertex_normals.foreach_get("vector", nr)
        tris = np.zeros(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tris)
        mw = np.array(ob.matrix_world)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        pos.append(co)
        nrm.append(nr.reshape(-1, 3))
        idx.append(tris.reshape(-1, 3) + base)
        for key, lst, dt in (("part", part, np.int32), ("fs", fs, np.float32), ("fc", fc, np.float32), ("up", up, np.float32)):
            a = np.zeros(n, dt)
            me.attributes[key].data.foreach_get("value", a)
            lst.append(a)
        base += n
        ob.evaluated_get(dg).to_mesh_clear()
    return [np.concatenate(a) for a in (pos, nrm, idx, part, fs, fc, up)]


def fine_noise(pos, freq):
    return np.array([noise.noise(V((x * freq, y * freq, z * freq))) for x, y, z in pos])


def export(name, pos, nrm, idx, part, fs, fc, up, col, ao, gloss, b0, b1, w0, mats, meta, extra):
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
    manifest = {"vertexBytes": len(v) * 24, "indexCount": int(idx.size), "matrixBytes": int(m16.nbytes), "clips": meta,
                "perchHeight": 0, "credit": f"Modeled, rigged and animated in Blender (tools/blender/model_{name}.py)", **extra}
    json.dump(manifest, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(f"[{name}] {len(v)} verts, {idx.shape[0]} tris, {len(mats) // len(extra['bones'])} frames", flush=True)


# ---------------------------------------------------------------------------------------
# Preview: the skinned mesh on the armature, frames of every clip.


def preview(name, pos, idx, col, b0, b1, w0, rig, actions, size, look=(0, 0, 0), views=("side",)):
    me = bpy.data.meshes.new(f"{name}-mesh")
    me.from_pydata([tuple(p) for p in pos], [], [tuple(t) for t in idx])
    attr = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.concatenate([col ** 2.2, np.ones((len(col), 1))], 1).astype(np.float32).ravel())
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    groups = [ob.vertex_groups.new(name=b) for b in rig.order]
    for i in range(len(pos)):
        groups[b0[i]].add([i], float(w0[i]), "REPLACE")
        if w0[i] < 0.999:
            groups[b1[i]].add([i], float(1 - w0[i]), "ADD")
    mod = ob.modifiers.new("rig", "ARMATURE")
    mod.object = rig.ob
    mat = bpy.data.materials.new("m")
    nt = mat.node_tree if mat.node_tree else None
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    ca = nt.nodes.new("ShaderNodeVertexColor")
    ca.layer_name = "Col"
    nt.links.new(ca.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.7
    me.materials.append(mat)
    scene = bpy.context.scene
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(40), 0, math.radians(35))
    scene.collection.objects.link(so)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.62, 0.75, 1)
    scene.world = world
    # A ground plane at z = 0 for contact.
    bpy.ops.mesh.primitive_plane_add(size=size * 8, location=(0, 0, 0))
    cam = bpy.data.cameras.new("cam")
    cam.lens = 50
    co = bpy.data.objects.new("cam", cam)
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 360
    scene.render.resolution_y = 270
    dirs = {"side": V((1.0, 0.15, 0.3)), "front": V((0.3, 1.0, 0.3)), "top": V((0.2, -0.4, 1.0)), "back": V((-0.6, -1.0, 0.4))}
    os.makedirs(PREVIEW, exist_ok=True)
    for view in views:
        loc = dirs[view].normalized() * size * 3.2 + V(look)
        co.location = loc
        co.rotation_euler = (V(look) - loc).to_track_quat("-Z", "Y").to_euler()
        for cname, (act, fps, loop, frames) in actions.items():
            if CLIPS_ONLY and cname not in CLIPS_ONLY:
                continue
            rig.ob.animation_data.action = act
            n = min(8, frames + 1)
            for k in range(n):
                f = round(k * frames / n) if loop else round(k * frames / max(n - 1, 1))
                scene.frame_set(f)
                scene.render.filepath = os.path.join(os.path.abspath(PREVIEW), f"{name}-{cname}-{view}-{k:02d}.png")
                bpy.ops.render.render(write_still=True)


def finish(name, obs, rig, allowed, side_tol, paint, clips, extra, size, look=(0, 0, 0), views=("side",), wing_like=()):
    """Collect, weight, paint, author, bake, export (and preview)."""
    pos, nrm, idx, part, fs, fc, up = collect(obs)
    b0, b1, w0 = auto_weights(rig, pos, part, allowed, side_tol)
    col, ao, gloss = paint(pos, nrm, part, fs, fc, up)
    rig_ob = rig.build(wing_like)
    actions = author(rig_ob, clips, 1.0)
    mats, meta = bake(rig_ob, actions, rig.order)
    # Head tracking (the game turns the head and neck toward what the animal watches): the
    # bones, and the pivot at the base of the neck (game axes: x, z up, y).
    if "neck" in rig.bones and "head" in rig.bones:
        hp = rig.bones["neck"][0]
        full = [i for i, b in enumerate(rig.order) if b.split(".")[0] in ("head", "ear", "throat")]
        extra = {**extra, "look": {"full": full, "neck": rig.order.index("neck"), "pivot": [hp[0], hp[2], hp[1]]}}
    export(name, pos, nrm, idx, part, fs, fc, up, col, ao, gloss, b0, b1, w0, mats, meta, {"bones": rig.order, **extra})
    if BLEND:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(BLEND.replace(".blend", f"-{name}.blend")))
    if PREVIEW:
        for ob in obs:
            bpy.data.objects.remove(ob)
        preview(name, pos, idx, col, b0, b1, w0, rig, actions, size, look, views)


def default_ao(nrm, pos, floor_z):
    return np.clip(0.6 + 0.4 * (nrm[:, 2] * 0.5 + 0.5), 0, 1)

