"""
Models, rigs and animates the field's animals in Blender and bakes their animation for the GPU
(the same pipeline and output format as the birds, tools/blender/model_birds.py):

  blender -b --factory-startup --python tools/blender/model_animals.py -- <out dir> \
      [--species butterfly,hare,...] [--preview <dir>] [--clips a,b] [--blend <file.blend>]

Each species is a function below that builds:
  skin     one smooth body: metaballs (ellipsoids and capsules for limbs) remeshed, smoothed
           and reduced, so legs, neck and tail grow out of the body without seams
  parts    eyes, noses and hooves, ears (lofted, cupped), antlers, wings (fans of radial rings
           with a wing coordinate for the shader's patterns)
  rig      an armature; bones lying along the body keep local X pointing right (+x), so a
           positive X rotation always turns the bone's tip "nose up / foot forward"
  weights  two bones per vertex, from the distance to each bone (scaled by its thickness),
           limited per part and side
  clips    key poses (Blender actions, Bezier-eased), or poses sampled from a gait function
           for cycles; baked per frame to skinning matrices

Output per species: <name>.bin (24-byte vertices, uint32 indices, float16 3x4 matrices per
frame and bone) and <name>.json (bones, clips {start, frames, fps, loop}, standHeight: body
origin above the ground in the rest pose, plus species extras such as gait strides).

Vertex layout: p float16x4 (xyz, part: 0 fur/skin, 1 nose/hoof/claw (dark, glossy), 2 eye,
3 wing, 4 antler/horn, 5 inner ear / pale bare skin); n snorm8x4 (normal, 1 on upper/outer
surfaces); t float16x2 (wings: span 0..1 and around 0..1, +2 on hind wings; else 0);
e unorm8x4 (sRGB albedo, AO); j uint8x4 (bone 0, bone 1, weight of bone 0 * 255, gloss * 255).
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
                "perchHeight": 0, "credit": "Modeled, rigged and animated in Blender (tools/blender/model_animals.py)", **extra}
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
    export(name, pos, nrm, idx, part, fs, fc, up, col, ao, gloss, b0, b1, w0, mats, meta, {"bones": rig.order, **extra})
    if BLEND:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(BLEND.replace(".blend", f"-{name}.blend")))
    if PREVIEW:
        for ob in obs:
            bpy.data.objects.remove(ob)
        preview(name, pos, idx, col, b0, b1, w0, rig, actions, size, look, views)


def default_ao(nrm, pos, floor_z):
    return np.clip(0.6 + 0.4 * (nrm[:, 2] * 0.5 + 0.5), 0, 1)


# =======================================================================================
# Butterfly: thorax, abdomen, head, clubbed antennae, four wings on their own bones. The
# wing patterns are drawn by the shader (palette by species) from the wing coordinates.


def butterfly():
    reset()
    sk = Skin(0.0007, scale=10)
    sk.ball((0, 0.001, 0), 0.0032, (0.9, 1.6, 0.95))  # thorax
    sk.capsule((0, -0.002, 0), (0, -0.016, -0.0015), 0.0021)  # abdomen
    sk.ball((0, 0.0065, 0.0004), 0.0026, (1.0, 0.9, 0.9))  # head
    skin = sk.build(0.0005, 0.35, smooth=4)
    obs = [skin, eyes(skin, (0.0016, 0.0075, 0.001), 0.0011, sink=0.3, segs=(8, 6))]
    for s in (-1, 1):
        path = [V((s * 0.0006, 0.0078, 0.0012)), V((s * 0.003, 0.012, 0.004)), V((s * 0.0055, 0.016, 0.0065)), V((s * 0.0072, 0.0185, 0.0078))]
        obs.append(tube(path, [0.00022, 0.0002, 0.00018, 0.00045], 1, segs=5, name="antenna"))
    # Forewing (right side, in the wing plane: x out, y forward), hindwing rounder below it.
    fore = [(0.0, 0.004), (0.01, 0.009), (0.02, 0.011), (0.028, 0.009), (0.029, 0.004), (0.025, -0.003), (0.017, -0.007), (0.006, -0.006), (0.0, -0.001)]
    hind = [(0.0, -0.001), (0.01, -0.002), (0.018, -0.006), (0.02, -0.012), (0.016, -0.019), (0.009, -0.021), (0.003, -0.016), (0.0, -0.006)]
    for s in (-1, 1):
        obs.append(fan_wing((s * 0.0012, 0.002, 0.0012), fore, 3, side=s, name="fore", lift=0.0008))
        obs[-1]["bone"] = "fore"
        obs.append(fan_wing((s * 0.0012, -0.001, 0.0008), hind, 3, side=s, hind=True, name="hind", lift=0.0004))
        obs[-1]["bone"] = "hind"
    rig = Rig()
    rig.bone("body", (0, -0.003, 0), (0, 0.004, 0), None, 0.004)
    rig.bone("abdomen", (0, -0.003, 0), (0, -0.016, -0.0015), "body", 0.003)
    rig.bone("head", (0, 0.005, 0.0004), (0, 0.009, 0.0004), "body", 0.003)
    rig.bone("fore.R", (0.0012, 0.002, 0.0012), (0.028, 0.006, 0.0012), "body", 0.02)
    rig.bone("hind.R", (0.0012, -0.001, 0.0008), (0.018, -0.012, 0.0008), "body", 0.02)
    allowed = {0: ["body", "abdomen", "head"], 1: ["head"], 2: ["head"], 3: None}

    def wing_bone(pos, part):
        return None

    # Wings belong entirely to their own bone (no blending into the body).
    def paint(pos, nrm, part, fs, fc, up):
        n = len(pos)
        col = np.zeros((n, 3))
        col[:] = (0.08, 0.06, 0.05)
        hairy = fine_noise(pos, 3000)
        col[part == 0] = np.array((0.12, 0.09, 0.07)) * (0.9 + 0.2 * hairy[part == 0, None])
        col[part == 1] = (0.05, 0.05, 0.05)
        col[part == 2] = (0.02, 0.02, 0.02)
        # Wings: neutral here (the shader paints them by species); a darker root.
        col[part == 3] = np.array((0.8, 0.8, 0.8))[None, :] * (0.8 + 0.2 * np.clip(fs[part == 3, None] * 3, 0, 1))
        ao = np.where(part == 3, 1.0, default_ao(nrm, pos, 0))
        gloss = np.zeros(n)
        return col, ao, gloss

    def wings(f, h, body=(0, 0, 0), ab=(0, 0, 0), head=(0, 0, 0)):
        return pose(**{"fore": f, "hind": h, "body": body, "abdomen": ab, "head": head})

    def flap(ph):
        # Up (wings nearly meeting over the back) to down below level; the hindwing a touch late.
        a = math.cos(2 * math.pi * ph)
        b = math.cos(2 * math.pi * (ph - 0.06))
        return wings((15 + 62 * a, -6 * math.sin(2 * math.pi * ph), 4 * a), (12 + 58 * b, -4 * math.sin(2 * math.pi * ph), -6 + 4 * b),
                     body=(6 * math.sin(2 * math.pi * ph), 0, 0), ab=(-8 - 6 * a, 0, 0))

    clips = {
        "flap": (16, True, cycle(flap, 1.0, 16)),
        # Gliding on held wings, a slight shiver.
        "glide": (10, True, [(0, wings((8, 0, 2), (6, 0, -4), ab=(-6, 0, 0))), (0.5, wings((12, 0, 2), (9, 0, -4), ab=(-9, 0, 0))), (1.0, wings((8, 0, 2), (6, 0, -4), ab=(-6, 0, 0)))]),
        # On a flower, basking: wings open flat, slowly closing up over the back and opening again.
        "bask": (10, True, [(0, wings((2, 0, 0), (0, 0, -3))), (1.6, wings((4, 0, 0), (2, 0, -3), head=(0, 0, 6))), (2.2, wings((80, 0, 0), (78, 0, -3))),
                            (2.8, wings((82, 0, 0), (80, 0, -3), head=(0, 0, -6))), (3.4, wings((6, 0, 0), (4, 0, -3))), (5.0, wings((2, 0, 0), (0, 0, -3)))]),
    }
    finish("butterfly", obs, rig, allowed, 0.0005, paint, clips, {"span": 0.06}, 0.03, views=("side", "top"), wing_like=("fore", "hind"))


# =======================================================================================
# Brown hare: long ears with black tips, big hind legs folded under it, amber eyes.

def hare():
    reset()
    sk = Skin(0.006)
    # Compact and round-backed: big haunches high behind, the body sloping down to a deep
    # chest, a short thick neck and a long-muzzled head. (Metaball radii read ~25% larger
    # than the surface they make.)
    sk.ball((0, -0.1, 0.17), 0.16, (0.95, 1.15, 1.05))  # haunches
    sk.ball((0, 0.02, 0.18), 0.15, (0.85, 1.3, 0.95))  # back and belly
    sk.ball((0, 0.12, 0.18), 0.12, (0.85, 0.95, 1.05))  # chest
    sk.ball((0, 0.18, 0.245), 0.085, (0.85, 0.9, 1.0))  # neck
    sk.ball((0, 0.235, 0.29), 0.09, (0.85, 1.25, 0.9))  # head
    sk.ball((0, 0.295, 0.275), 0.055, (0.78, 1.1, 0.8))  # muzzle
    sk.ball((0, -0.225, 0.2), 0.045, (0.9, 0.9, 1.0))  # tail
    for s_ in (-1, 1):
        # Hind leg folded under the haunch, the long foot flat on the ground.
        sk.limb([(s_ * 0.07, -0.1, 0.15), (s_ * 0.085, 0.0, 0.095), (s_ * 0.08, -0.15, 0.045), (s_ * 0.075, 0.0, 0.02)], [0.075, 0.045, 0.026], 2.0)
        sk.capsule((s_ * 0.075, 0.0, 0.02), (s_ * 0.072, 0.04, 0.018), 0.024)
        # Front leg: slim, a little forward of the chest.
        sk.limb([(s_ * 0.05, 0.12, 0.14), (s_ * 0.048, 0.125, 0.075), (s_ * 0.045, 0.15, 0.02)], [0.04, 0.024], 2.0)
        sk.capsule((s_ * 0.045, 0.15, 0.018), (s_ * 0.043, 0.185, 0.016), 0.02)
    skin = sk.build(0.005, 0.22)
    obs = [skin, eyes(skin, (0.05, 0.255, 0.315), 0.015, sink=0.35)]
    nose_co, _ = surface(skin, (0, 0.36, 0.28))
    obs.append(sphere(nose_co + V((0, -0.004, 0)), 0.009, 1, (1.2, 0.8, 0.8)))
    for s_ in (-1, 1):
        obs.append(leaf(V((s_ * 0.024, 0.215, 0.345)), V((s_ * 0.25, -0.35, 1.0)), V((s_ * 0.6, 0.9, 0.0)), 0.15, 0.05, 0.35, 0, thick=0.004, name="ear", tip_round=0.7))
        obs[-1]["bone"] = "ear"
    rig = Rig()
    rig.bone("body", (0, -0.08, 0.17), (0, 0.06, 0.18), None, 0.15)
    rig.bone("chest", (0, 0.06, 0.18), (0, 0.16, 0.21), "body", 0.12)
    rig.bone("neck", (0, 0.16, 0.22), (0, 0.22, 0.28), "chest", 0.08)
    rig.bone("head", (0, 0.22, 0.29), (0, 0.34, 0.28), "neck", 0.09)
    rig.bone("ear.R", (0.024, 0.215, 0.345), (0.062, 0.16, 0.485), "head", 0.035)
    rig.bone("tail", (0, -0.21, 0.19), (0, -0.26, 0.21), "body", 0.05)
    rig.bone("thigh.R", (0.07, -0.1, 0.15), (0.085, 0.0, 0.095), "body", 0.075)
    rig.bone("shin.R", (0.085, 0.0, 0.095), (0.08, -0.15, 0.045), "thigh.R", 0.045)
    rig.bone("foot.R", (0.08, -0.15, 0.045), (0.072, 0.04, 0.018), "shin.R", 0.03)
    rig.bone("arm.R", (0.05, 0.12, 0.15), (0.048, 0.125, 0.075), "chest", 0.04)
    rig.bone("fore.R", (0.048, 0.125, 0.075), (0.045, 0.15, 0.02), "arm.R", 0.025)
    rig.bone("paw.R", (0.045, 0.15, 0.02), (0.043, 0.185, 0.016), "fore.R", 0.02)
    allowed = {0: None, 1: ["head"], 2: ["head"]}

    def paint(pos, nrm, part, fs, fc, up):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        n = len(pos)
        fine = fine_noise(pos, 60)
        agouti = np.array((0.5, 0.36, 0.22))
        dark = np.array((0.24, 0.17, 0.11))
        belly = np.array((0.88, 0.84, 0.76))
        upn = np.clip((nrm[:, 2] + 0.2) * 1.6, 0, 1)
        col = belly * (1 - upn[:, None]) + agouti * upn[:, None]
        # Grizzled back: dark-tipped guard hairs.
        col = col * (1 - 0.35 * np.clip(fine[:, None] * 2, 0, 1) * upn[:, None]) + dark * 0.0
        col = np.where(((y > 0.27) & (z < 0.27))[:, None], belly * 0.95, col)  # pale chin
        tailtop = (y < -0.22) & (z > 0.2)
        col = np.where(tailtop[:, None], np.array((0.06, 0.05, 0.05)), col)
        col = np.where(((y < -0.22) & (z <= 0.2))[:, None], belly, col)
        # Eye ring.
        er = np.hypot(np.abs(x) - 0.05, np.hypot(y - 0.255, z - 0.315))
        col = np.where(((er < 0.026) & (part == 0))[:, None], np.array((0.82, 0.72, 0.55)), col)
        # Ears: agouti with black tips, pale inside.
        ear = (part == 0) & (fs > 0.001)
        tip = ear & (fs > 0.82)
        inner = ear & (up < 0.5) == False
        col = np.where(tip[:, None], np.array((0.05, 0.04, 0.04)), col)
        col[part == 1] = (0.25, 0.15, 0.13)
        col[part == 2] = (0.55, 0.36, 0.08)
        ao = default_ao(nrm, pos, 0)
        gloss = np.where((part == 1) | (part == 2), 1.0, 0.0)
        return np.clip(col, 0, 1), ao, gloss

    stand = pose()

    # Gaits, as functions of phase. Hares lope: both hind feet land ahead of the front ones.
    def lope(ph, k=1.0):
        s = math.sin(2 * math.pi * ph)
        c = math.cos(2 * math.pi * ph)
        hind = (-35 * c * k, 0, 0)
        return pose(
            body=(-10 * s * k, 0, 0), chest=(8 * s * k, 0, 0), neck=(-6 * s * k, 0, 0), head=(4 * s * k, 0, 0),
            ear=(45 * k + 8 * s * k, 0, 6), tail=(20 * k, 0, 0),
            thigh=(-30 * c * k + 10 * k, 0, 0), shin=(20 * max(0, s) * k, 0, 0), foot=(25 * c * k - 20 * max(0, -s) * k, 0, 0),
            arm=(35 * math.cos(2 * math.pi * (ph - 0.3)) * k, 0, 0), fore=(-30 * max(0, math.sin(2 * math.pi * (ph - 0.3))) * k, 0, 0),
            paw=(30 * max(0, math.sin(2 * math.pi * (ph - 0.3))) * k, 0, 0),
            loc=(0, 0, 0.05 * max(0, s) * k),
        )

    alert = pose(body=(28, 0, 0), chest=(14, 0, 0), neck=(6, 0, 0), head=(-22, 0, 0), ear=(-12, 0, -2), thigh=(-26, 0, 0), foot=(-2, 0, 0),
                 arm=(-30, 0, 0), fore=(-12, 0, 0), paw=(-40, 0, 0), loc=(0, -0.02, 0.06))
    graze = pose(body=(-6, 0, 0), chest=(-12, 0, 0), neck=(-35, 0, 0), head=(-20, 0, 0), ear=(16, 0, 10), arm=(14, 0, 0), fore=(4, 0, 0), paw=(-18, 0, 0))
    nibble = over(graze, head=(-26, 0, 0))
    clips = {
        "idle": (12, True, [
            (0, stand), (0.6, over(stand, ear=(6, 0, 4))), (1.4, over(stand, head=(4, 0, 22), ear=(10, 8, 12))), (2.6, over(stand, head=(4, 0, 22))),
            (3.0, over(stand, ear=(30, 0, 4))), (3.6, stand), (4.4, over(stand, head=(0, 0, -18), ear=(0, -6, -4))), (5.4, over(stand, head=(0, 0, -18))), (6.0, stand),
        ]),
        "graze": (12, True, [(0, graze), (0.25, nibble), (0.5, graze), (0.75, nibble), (1.0, graze), (1.6, over(graze, ear=(10, 0, 18))), (2.2, graze), (2.45, nibble), (2.7, graze), (3.2, graze)]),
        "alert": (12, True, [(0, alert), (1.0, over(alert, head=(-22, 0, 18), ear=(-16, 0, 4))), (2.0, over(alert, head=(-22, 0, -16))), (3.0, alert)]),
        "hop": (16, True, cycle(lambda ph: lope(ph, 0.55), 1.0, 16)),
        "run": (24, True, cycle(lambda ph: lope(ph, 1.0), 1.0, 16)),
    }
    # Strides: ground covered per cycle (m, life size), to match the clip to the speed.
    finish("hare", obs, rig, allowed, 0.02, paint, clips, {"stride": {"hop": 0.45, "run": 1.4}, "standHeight": 0.0}, 0.35, look=(0, 0.05, 0.2), views=("side", "front"))



# =======================================================================================
# Quadrupeds (roe deer, red fox): one builder. A spec gives the torso, neck, head and the
# leg joints (right side; hind: hip, stifle, hock, toe; front: shoulder, elbow, carpus, toe);
# gaits are sampled from leg cycles with each foot's phase.


def leg_cycle(p, duty, amp, lift, front):
    """One leg at phase p: stance sweeps the foot back flat on the ground; swing lifts and
    folds it forward. Returns (upper, middle, lower) rotations (degrees)."""
    p %= 1.0
    if p < duty:
        u = p / duty
        swing, flex = amp * (1 - 2 * u), 0.0
    else:
        u = (p - duty) / (1 - duty)
        e = u * u * (3 - 2 * u)
        swing, flex = amp * (-1 + 2 * e), math.sin(math.pi * u)
    if front:
        return (swing * 0.9, flex * lift * 0.5, -flex * lift * 1.5)
    return (swing, -flex * lift * 0.9, flex * lift * 1.6)


def gait(q, ph, offsets, duty, amp, lift, bob, rock, head_k=1.0, extra=None):
    """A pose for the gait at phase ph. offsets: phase of (hind R, hind L, front R, front L)."""
    names = (("thigh", "shin", "hcannon", "R", False), ("thigh", "shin", "hcannon", "L", False), ("upper", "fore", "fcannon", "R", True), ("upper", "fore", "fcannon", "L", True))
    p = {}
    for (a, b, c, side, front), off in zip(names, offsets):
        r = leg_cycle(ph + off, duty, amp, lift, front)
        p[f"{a}.{side}"] = (r[0], 0, 0)
        p[f"{b}.{side}"] = (r[1], 0, 0)
        p[f"{c}.{side}"] = (r[2], 0, 0)
    w = 2 * math.pi * ph
    p["loc"] = (0, 0, bob * (0.5 - 0.5 * math.cos(2 * w)) if rock == 0 else bob * max(0, math.sin(w)))
    p["body"] = (rock * math.sin(w), 0, 0)
    p["chest"] = (-rock * 0.5 * math.sin(w), 0, 0)
    p["neck"] = (-rock * 0.6 * math.sin(w) * head_k, 0, 0)
    p["head"] = (rock * 0.4 * math.sin(w) * head_k, 0, 0)
    if extra:
        for k, v in extra(ph).items():
            p[k] = v
    return p


def quadruped(name, q):
    reset()
    sk = Skin(q["res"])
    for co, r, size in q["balls"]:
        sk.ball(co, r, size)
    for s_ in (-1, 1):
        m = lambda pt: (s_ * pt[0], pt[1], pt[2])
        sk.limb([m(pt) for pt in q["hind"]], q["hind_r"])
        sk.limb([m(pt) for pt in q["front"]], q["front_r"])
    for path, radii in q.get("chains", []):
        sk.limb(path, radii)
    skin = sk.build(q["voxel"], q["ratio"])
    obs = [skin, eyes(skin, q["eye"], q["eye_r"], sink=0.35)]
    nose, _ = surface(skin, q["nose"])
    obs.append(sphere(nose + V((0, -q["nose_r"] * 0.3, 0)), q["nose_r"], 1, (1.2, 0.8, 0.85)))
    for s_ in (-1, 1):
        e = q["ear"]
        obs.append(leaf(V((s_ * e["base"][0], e["base"][1], e["base"][2])), V((s_ * e["dir"][0], e["dir"][1], e["dir"][2])), V((s_ * 0.7, 1.0, 0.1)), e["len"], e["width"], e["cup"], 0, thick=e["len"] * 0.04, name="ear", tip_round=e.get("round", 0.8)))
        obs[-1]["bone"] = "ear"
        # Hooves / paws.
        toe = q["hind"][-1], q["front"][-1]
        for t in toe:
            obs.append(sphere((s_ * t[0], t[1] + q["hoof"][1], t[2] + q["hoof"][0]), q["hoof"][0], 1, q["hoof"][2]))
            obs[-1]["bone"] = "hcannon" if t is toe[0] else "fcannon"
    for path, radii in q.get("antlers", []):
        for s_ in (-1, 1):
            obs.append(tube([(s_ * x, y, z) for x, y, z in path], radii, 4, segs=6, name="antler"))
            obs[-1]["bone"] = "head"
    rig = Rig()
    b = q["bones"]
    rig.bone("body", b["body"][0], b["body"][1], None, b["body"][2])
    rig.bone("chest", b["chest"][0], b["chest"][1], "body", b["chest"][2])
    rig.bone("neck", b["neck"][0], b["neck"][1], "chest", b["neck"][2])
    rig.bone("head", b["head"][0], b["head"][1], "neck", b["head"][2])
    e = q["ear"]
    rig.bone("ear.R", e["base"], tuple(V(e["base"]) + V(e["dir"]).normalized() * e["len"]), "head", e["width"] * 0.6)
    prev = "body"
    for i, (h, t, r) in enumerate(b["tail"]):
        rig.bone(f"tail{i}", h, t, prev, r)
        prev = f"tail{i}"
    H, F = q["hind"], q["front"]
    rig.bone("thigh.R", H[0], H[1], "body", q["hind_r"][0])
    rig.bone("shin.R", H[1], H[2], "thigh.R", q["hind_r"][1])
    rig.bone("hcannon.R", H[2], H[3], "shin.R", q["hind_r"][2] * 1.2)
    rig.bone("upper.R", F[0], F[1], "chest", q["front_r"][0])
    rig.bone("fore.R", F[1], F[2], "upper.R", q["front_r"][1])
    rig.bone("fcannon.R", F[2], F[3], "fore.R", q["front_r"][2] * 1.2)
    allowed = {0: None, 1: None, 2: ["head"], 4: ["head"]}
    clips = q["clips"]()
    finish(name, obs, rig, allowed, q["side_tol"], q["paint"], clips, {"stride": q["stride"]}, q["view"], look=q["look"], views=("side", "front"))


def roe_deer():
    def paint(pos, nrm, part, fs, fc, up):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        fine = fine_noise(pos, 25)
        coat = np.array((0.56, 0.3, 0.14))
        belly = np.array((0.78, 0.62, 0.45))
        upn = np.clip((nrm[:, 2] + 0.4) * 1.4, 0, 1)
        col = belly * (1 - upn[:, None]) + coat * upn[:, None]
        col *= (0.95 + 0.1 * fine)[:, None]
        rump = (y < -0.36) & (z > 0.45) & (np.abs(x) < 0.16)
        col = np.where(rump[:, None], np.array((0.95, 0.93, 0.88)), col)
        legs = z < 0.3
        col = np.where(legs[:, None], col * 0.8 + np.array((0.03, 0.02, 0.0)), col)
        muzzle = (y > 0.63) & (z < 1.0)
        col = np.where(muzzle[:, None], np.array((0.2, 0.17, 0.15)), col)
        chin = (y > 0.6) & (z < 0.94)
        col = np.where(chin[:, None], np.array((0.93, 0.9, 0.86)), col)
        ear = fs > 0.001
        col = np.where((ear & (up < 0.5))[:, None], np.array((0.82, 0.74, 0.66)), col)
        col = np.where((ear & (fs > 0.9))[:, None], np.array((0.1, 0.08, 0.07)), col)
        col[part == 1] = (0.06, 0.05, 0.05)
        col[part == 2] = (0.05, 0.04, 0.03)
        col[part == 4] = (0.36, 0.3, 0.22)
        return np.clip(col, 0, 1), default_ao(nrm, pos, 0), np.where((part == 1) | (part == 2), 1.0, 0.0)

    def clips():
        stand = {}
        graze = {"neck": (-62, 0, 0), "head": (-12, 0, 0), "chest": (-6, 0, 0), "upper": (8, 0, 0), "ear": (14, 0, 0)}
        chew = over(graze, head=(-16, 0, 4))
        alert = {"neck": (14, 0, 0), "head": (-8, 0, 0), "ear": (-12, 0, 0), "tail0": (30, 0, 0)}
        return {
            "idle": (12, True, [(0, stand), (1.2, over(stand, head=(0, 0, 25), neck=(4, 0, 10))), (2.4, over(stand, head=(0, 0, 25), neck=(4, 0, 10), ear=(12, 0, 6))),
                                (3.0, over(stand, tail0=(30, 0, 0))), (3.3, stand), (4.5, over(stand, head=(0, 0, -22), neck=(4, 0, -12))), (5.6, over(stand, head=(0, 0, -22), neck=(4, 0, -12))), (6.4, stand), (8.0, stand)]),
            "graze": (12, True, [(0, graze), (0.4, chew), (0.8, graze), (1.2, chew), (1.6, graze), (2.4, over(graze, ear=(-10, 0, 8))), (3.0, graze), (3.4, chew), (4.0, graze)]),
            "alert": (12, True, [(0, alert), (1.5, over(alert, head=(-8, 0, 18), ear=(-16, 0, 6))), (3.0, over(alert, head=(-8, 0, -14))), (4.0, alert)]),
            # Walk: hind right, front right, hind left, front left (a lateral sequence).
            "walk": (24, True, cycle(lambda ph: gait(None, ph, (0.0, 0.5, 0.25, 0.75), 0.66, 22, 30, 0.015, 0, extra=lambda ph: {"neck": (-6 + 4 * math.sin(4 * math.pi * ph), 0, 0)}), 1.0, 16)),
            # Bounding gallop: hinds push together, the fronts reach far ahead.
            "run": (30, True, cycle(lambda ph: gait(None, ph, (0.0, 0.06, 0.48, 0.56), 0.32, 38, 45, 0.16, 10, extra=lambda ph: {"ear": (40, 0, 0), "tail0": (35, 0, 0)}), 1.0, 16)),
        }

    q = dict(
        res=0.012, voxel=0.01, ratio=0.22, side_tol=0.03, view=0.75, look=(0, 0.1, 0.55),
        balls=[((0, -0.3, 0.68), 0.25, (0.82, 1.1, 1.0)), ((0, -0.02, 0.63), 0.26, (0.78, 1.55, 0.95)), ((0, 0.25, 0.64), 0.23, (0.78, 1.0, 1.1)),
               ((0, 0.6, 1.0), 0.1, (0.82, 1.2, 0.95)), ((0, 0.69, 0.965), 0.06, (0.72, 1.25, 0.8)), ((0, -0.43, 0.72), 0.06, (0.8, 0.7, 1.0))],
        chains=[([(0, 0.34, 0.74), (0, 0.46, 0.88), (0, 0.55, 0.98)], [0.1, 0.075])],
        hind=[(0.1, -0.32, 0.64), (0.11, -0.2, 0.42), (0.1, -0.38, 0.24), (0.09, -0.32, 0.02)], hind_r=[0.11, 0.05, 0.03],
        front=[(0.1, 0.26, 0.58), (0.1, 0.22, 0.4), (0.09, 0.26, 0.2), (0.09, 0.27, 0.02)], front_r=[0.08, 0.045, 0.028],
        eye=(0.06, 0.61, 1.02), eye_r=0.018, nose=(0, 0.76, 0.965), nose_r=0.022, hoof=(0.022, 0.012, (0.9, 1.4, 0.8)),
        ear=dict(base=(0.045, 0.555, 1.06), dir=(0.6, -0.4, 0.75), len=0.13, width=0.075, cup=0.35, round=0.9),
        antlers=[([(0.03, 0.55, 1.07), (0.04, 0.54, 1.15), (0.045, 0.52, 1.24)], [0.012, 0.009, 0.005]), ([(0.04, 0.545, 1.14), (0.05, 0.6, 1.18)], [0.007, 0.004]),
                 ([(0.045, 0.53, 1.2), (0.05, 0.48, 1.23)], [0.006, 0.003])],
        bones=dict(body=((0, -0.32, 0.66), (0, 0.08, 0.66), 0.26), chest=((0, 0.08, 0.66), (0, 0.3, 0.68), 0.24), neck=((0, 0.32, 0.74), (0, 0.55, 0.98), 0.1),
                   head=((0, 0.55, 0.99), (0, 0.76, 0.96), 0.1), tail=[((0, -0.42, 0.72), (0, -0.48, 0.7), 0.06)]),
        stride={"walk": 0.9, "run": 2.6},
        paint=paint, clips=clips,
    )
    quadruped("deer", q)


def red_fox():
    def paint(pos, nrm, part, fs, fc, up):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        fine = fine_noise(pos, 40)
        red = np.array((0.78, 0.35, 0.1))
        white = np.array((0.94, 0.92, 0.88))
        black = np.array((0.08, 0.06, 0.05))
        upn = np.clip((nrm[:, 2] + 0.5) * 1.4, 0, 1)
        col = white * (1 - upn[:, None]) + red * upn[:, None]
        col *= (0.94 + 0.12 * fine)[:, None]
        throat = (y > 0.15) & (z < 0.42) & (nrm[:, 2] < 0.2) & (z > 0.2)
        col = np.where(throat[:, None], white, col)
        cheek = (y > 0.36) & (z < 0.47) & (nrm[:, 2] < 0.5)
        col = np.where(cheek[:, None], white, col)
        socks = (z < 0.15) & (part == 0)
        col = np.where(socks[:, None], black, col)
        tip = y < -0.62
        col = np.where(tip[:, None], white, col)
        ear = fs > 0.001
        col = np.where((ear & (up > 0.5))[:, None], red * 0.8, col)
        col = np.where((ear & (up > 0.5) & (fs > 0.55))[:, None], black, col)
        col = np.where((ear & (up < 0.5))[:, None], white * 0.9, col)
        col[part == 1] = (0.05, 0.04, 0.04)
        col[part == 2] = (0.55, 0.35, 0.05)
        return np.clip(col, 0, 1), default_ao(nrm, pos, 0), np.where((part == 1) | (part == 2), 1.0, 0.0)

    def clips():
        stand = {"tail0": (-10, 0, 0), "tail1": (-6, 0, 0)}
        listen = over(stand, head=(-10, 18, 0), neck=(10, 0, 0), ear=(-14, 0, 0))
        crouch = over(stand, loc=(0, -0.03, -0.08), body=(-6, 0, 0), neck=(-12, 0, 0), head=(-30, 0, 0), thigh=(20, 0, 0), shin=(-40, 0, 0), hcannon=(30, 0, 0),
                      upper=(10, 0, 0), fore=(-10, 0, 0), fcannon=(-6, 0, 0), tail0=(-4, 0, 0))
        leap = over(stand, loc=(0, 0.1, 0.42), body=(-24, 0, 0), neck=(-22, 0, 0), head=(-30, 0, 0), thigh=(-40, 0, 0), shin=(10, 0, 0), hcannon=(-20, 0, 0),
                    upper=(-30, 0, 0), fore=(40, 0, 0), fcannon=(-60, 0, 0), tail0=(20, 0, 0), tail1=(10, 0, 0))
        dive = over(stand, loc=(0, 0.25, 0.25), body=(-55, 0, 0), neck=(-20, 0, 0), head=(-20, 0, 0), thigh=(-30, 0, 0), upper=(40, 0, 0), fore=(20, 0, 0), fcannon=(0, 0, 0), tail0=(30, 0, 0), tail1=(20, 0, 0))
        land = over(stand, loc=(0, 0.3, -0.04), body=(-30, 0, 0), neck=(-25, 0, 0), head=(-35, 0, 0), upper=(30, 0, 0), fore=(-10, 0, 0), thigh=(-20, 0, 0), tail0=(12, 0, 0))
        return {
            "idle": (12, True, [(0, stand), (1.5, over(stand, head=(0, 0, 20), neck=(0, 0, 10))), (2.5, listen), (3.8, listen), (4.6, over(stand, head=(0, 0, -20))), (6.0, stand)]),
            "sniff": (12, True, [(0, over(stand, neck=(-40, 0, 0), head=(-20, 0, 0))), (0.6, over(stand, neck=(-44, 0, 8), head=(-26, 0, 6))), (1.2, over(stand, neck=(-40, 0, -6), head=(-20, 0, 0))), (2.0, over(stand, neck=(-40, 0, 0), head=(-20, 0, 0)))]),
            # Trot: diagonal pairs (hind right with front left).
            "trot": (24, True, cycle(lambda ph: gait(None, ph, (0.0, 0.5, 0.5, 0.0), 0.5, 26, 40, 0.025, 0, extra=lambda ph: {"tail0": (-8, 0, 0), "tail1": (-4 + 4 * math.sin(2 * math.pi * ph), 0, 0)}), 1.0, 16)),
            "run": (30, True, cycle(lambda ph: gait(None, ph, (0.0, 0.08, 0.5, 0.58), 0.35, 40, 50, 0.08, 9, extra=lambda ph: {"tail0": (8, 0, 0), "ear": (30, 0, 0)}), 1.0, 16)),
            # Mousing: freeze, listen, crouch, spring high and dive nose first into the grass.
            "pounce": (24, False, [(0, stand), (0.3, listen), (0.9, over(listen, head=(-28, 22, 0))), (1.2, crouch), (1.45, crouch), (1.7, leap), (1.95, dive), (2.15, land), (2.6, over(land, head=(-45, 0, 0))), (3.2, stand)]),
        }

    q = dict(
        res=0.007, voxel=0.006, ratio=0.22, side_tol=0.02, view=0.5, look=(0, -0.05, 0.3),
        balls=[((0, -0.17, 0.34), 0.165, (0.8, 1.05, 1.0)), ((0, 0.0, 0.33), 0.165, (0.78, 1.3, 0.95)), ((0, 0.16, 0.34), 0.155, (0.82, 1.0, 1.1)),
               ((0, 0.36, 0.5), 0.095, (0.95, 1.0, 0.85)), ((0, 0.44, 0.475), 0.045, (0.75, 1.3, 0.7)), ((0, 0.5, 0.465), 0.028, (0.7, 1.2, 0.7))],
        chains=[([(0, 0.22, 0.38), (0, 0.3, 0.46), (0, 0.35, 0.5)], [0.08, 0.07]),
                ([(0, -0.3, 0.36), (0, -0.46, 0.31), (0, -0.6, 0.26), (0, -0.7, 0.24)], [0.05, 0.085, 0.075])],
        hind=[(0.06, -0.2, 0.3), (0.065, -0.11, 0.19), (0.06, -0.24, 0.1), (0.055, -0.2, 0.012)], hind_r=[0.075, 0.04, 0.026],
        front=[(0.06, 0.16, 0.28), (0.06, 0.13, 0.19), (0.055, 0.17, 0.08), (0.055, 0.19, 0.012)], front_r=[0.06, 0.034, 0.024],
        eye=(0.04, 0.41, 0.52), eye_r=0.011, nose=(0, 0.53, 0.465), nose_r=0.011, hoof=(0.016, 0.01, (1.0, 1.4, 0.7)),
        ear=dict(base=(0.035, 0.35, 0.565), dir=(0.3, -0.05, 1.0), len=0.07, width=0.048, cup=0.3, round=0.5),
        bones=dict(body=((0, -0.2, 0.33), (0, 0.06, 0.33), 0.16), chest=((0, 0.06, 0.33), (0, 0.2, 0.36), 0.15), neck=((0, 0.22, 0.38), (0, 0.33, 0.49), 0.08),
                   head=((0, 0.33, 0.5), (0, 0.52, 0.465), 0.07), tail=[((0, -0.28, 0.36), (0, -0.48, 0.3), 0.08), ((0, -0.48, 0.3), (0, -0.72, 0.24), 0.08)]),
        stride={"trot": 0.55, "run": 1.5},
        paint=paint, clips=clips,
    )
    quadruped("fox", q)


# =======================================================================================
# Common frog: squat body, bulging eyes, long hind legs folded at its sides, webbed feet; a
# throat that pulses as it breathes and swells as it calls.

def frog():
    reset()
    sk = Skin(0.0012, scale=10)
    # Squat: the belly almost on the ground, a broad flat head.
    sk.ball((0, -0.006, 0.016), 0.034, (0.95, 1.3, 0.62))
    sk.ball((0, 0.024, 0.017), 0.028, (1.15, 0.85, 0.55))
    for s_ in (-1, 1):
        m = lambda p: (s_ * p[0], p[1], p[2])
        sk.limb([m((0.014, -0.022, 0.013)), m((0.03, 0.0, 0.009)), m((0.022, -0.028, 0.005)), m((0.033, -0.006, 0.002))], [0.011, 0.0065, 0.0045])
        sk.limb([m((0.012, 0.016, 0.009)), m((0.015, 0.025, 0.005)), m((0.016, 0.031, 0.0015))], [0.0045, 0.0035])
        sk.ball(m((0.0105, 0.027, 0.026)), 0.0085, (1, 1, 0.9))  # eye bumps
    skin = sk.build(0.001, 0.3, smooth=6)
    obs = [skin, eyes(skin, (0.012, 0.028, 0.03), 0.0042, sink=0.5)]
    for s_ in (-1, 1):
        # Webbed hind feet: flat fans.
        obs.append(leaf(V((s_ * 0.032, -0.008, 0.002)), V((s_ * 0.5, 1.0, 0)), V((0, 0, 1)), 0.017, 0.015, 0.05, 0, thick=0.001, name="web", tip_round=0.3))
        obs[-1]["bone"] = "foot"
    rig = Rig()
    rig.bone("body", (0, -0.02, 0.016), (0, 0.01, 0.016), None, 0.03)
    rig.bone("head", (0, 0.01, 0.017), (0, 0.042, 0.017), "body", 0.025)
    rig.bone("throat", (0, 0.026, 0.006), (0, 0.037, 0.005), "head", 0.009)
    rig.bone("arm.R", (0.012, 0.016, 0.009), (0.015, 0.025, 0.005), "body", 0.0045)
    rig.bone("hand.R", (0.015, 0.025, 0.005), (0.016, 0.034, 0.0015), "arm.R", 0.0035)
    rig.bone("thigh.R", (0.014, -0.022, 0.013), (0.03, 0.0, 0.009), "body", 0.011)
    rig.bone("shin.R", (0.03, 0.0, 0.009), (0.022, -0.028, 0.005), "thigh.R", 0.0065)
    rig.bone("foot.R", (0.022, -0.028, 0.005), (0.038, 0.008, 0.002), "shin.R", 0.006)
    allowed = {0: None, 2: ["head"]}

    def paint(pos, nrm, part, fs, fc, up):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        blot = fine_noise(pos, 260)
        olive = np.array((0.42, 0.38, 0.18))
        brown = np.array((0.36, 0.25, 0.14))
        belly = np.array((0.85, 0.8, 0.6))
        upn = np.clip((nrm[:, 2] + 0.3) * 1.5, 0, 1)
        base = olive * (0.5 + 0.5 * np.clip(blot[:, None] + 0.5, 0, 1)) + brown * 0.3
        spots = np.clip((blot - 0.25) * 6, 0, 1)
        base = base * (1 - 0.7 * spots[:, None])
        col = belly * (1 - upn[:, None]) + base * upn[:, None]
        # The dark mask behind the eye.
        mask = (np.abs(x) > 0.017) & (y > 0.01) & (y < 0.022) & (z > 0.012) & (z < 0.022)
        col = np.where(mask[:, None], np.array((0.24, 0.15, 0.08)), col)
        col[part == 2] = (0.22, 0.16, 0.05)
        gloss = np.full(len(pos), 0.55)
        gloss[part == 2] = 1.0
        return np.clip(col, 0, 1), default_ao(nrm, pos, 0), gloss

    sit = {}
    def breath(k):
        return {"scale": {"throat": 1.0 + 0.18 * k}, "body": (1.5 * k, 0, 0)}
    def call(k):
        return {"scale": {"throat": (1.0 + 0.9 * k, 1.0 + 0.6 * k, 1.0 + 1.1 * k)}, "head": (6 * k, 0, 0)}
    # Legs kicked straight out behind: the thigh swings back, knee and ankle unfold.
    leap = {"body": (-12, 0, 0), "head": (6, 0, 0), "thigh": (0, 0, -55), "shin": (0, 0, 130), "foot": (0, 0, -70), "arm": (-30, 0, 0), "hand": (30, 0, 0), "loc": (0, 0.0, 0.015)}
    clips = {
        "sit": (12, True, [(0, breath(0)), (0.35, breath(1)), (0.7, breath(0)), (1.05, breath(1)), (1.4, breath(0)), (2.4, breath(0)), (2.75, breath(1)), (3.1, breath(0))]),
        "croak": (24, True, [(0, call(0)), (0.18, call(1)), (0.35, call(0.2)), (0.5, call(1)), (0.68, call(0)), (1.6, call(0))]),
        "hop": (30, False, [(0, sit), (0.08, {"body": (10, 0, 0), "thigh": (8, 0, 0)}), (0.2, leap), (0.42, over(leap, body=(5, 0, 0), arm=(-20, 0, 0))), (0.6, sit)]),
        "swim": (24, True, [(0, over(leap, body=(0, 0, 0))), (0.25, {"thigh": (10, 0, -10), "shin": (-10, 0, 0), "arm": (-30, 0, 0)}), (0.6, over(leap, body=(0, 0, 0)))]),
    }
    finish("frog", obs, rig, allowed, 0.003, paint, clips, {"stride": {}}, 0.05, look=(0, 0.0, 0.02), views=("side", "front"))


# =======================================================================================
# Dragonfly: big eyes, a long slender abdomen, two pairs of glassy wings beating out of phase.

def dragonfly():
    reset()
    sk = Skin(0.0007, scale=10)
    sk.ball((0, 0.018, 0.0), 0.0062, (0.85, 1.25, 1.0))  # thorax
    sk.limb([(0, 0.012, 0.0), (0, -0.005, -0.0005), (0, -0.025, -0.001), (0, -0.046, 0.0)], [0.0024, 0.0019, 0.0016])
    sk.ball((0, 0.027, 0.0006), 0.0045, (1.2, 0.8, 0.9))  # head
    skin = sk.build(0.0005, 0.3, smooth=4)
    obs = [skin, eyes(skin, (0.0028, 0.028, 0.0016), 0.0034, sink=0.15, segs=(12, 8))]
    fore = [(0.0, 0.002), (0.012, 0.0035), (0.03, 0.0035), (0.042, 0.0012), (0.04, -0.003), (0.024, -0.0045), (0.008, -0.004), (0.0, -0.002)]
    hind = [(0.0, 0.002), (0.012, 0.003), (0.03, 0.002), (0.039, -0.001), (0.035, -0.006), (0.02, -0.008), (0.006, -0.009), (0.0, -0.004)]
    for s_ in (-1, 1):
        obs.append(fan_wing((s_ * 0.0015, 0.021, 0.0028), fore, 3, side=s_, name="fw", rings=8, thick=0.0002))
        obs[-1]["bone"] = "fw"
        obs.append(fan_wing((s_ * 0.0015, 0.0155, 0.0028), hind, 3, side=s_, hind=True, name="hw", rings=8, thick=0.0002))
        obs[-1]["bone"] = "hw"
    rig = Rig()
    rig.bone("body", (0, 0.012, 0), (0, 0.024, 0), None, 0.006)
    rig.bone("abdomen", (0, 0.012, 0), (0, -0.016, -0.001), "body", 0.003)
    rig.bone("tip", (0, -0.016, -0.001), (0, -0.046, 0), "abdomen", 0.0025)
    rig.bone("head", (0, 0.024, 0.0006), (0, 0.031, 0.0006), "body", 0.004)
    rig.bone("fw.R", (0.0015, 0.021, 0.0028), (0.042, 0.022, 0.0028), "body", 0.02)
    rig.bone("hw.R", (0.0015, 0.0155, 0.0028), (0.039, 0.014, 0.0028), "body", 0.02)
    allowed = {0: ["body", "abdomen", "tip", "head"], 2: ["head"]}

    def paint(pos, nrm, part, fs, fc, up):
        y = pos[:, 1]
        col = np.zeros((len(pos), 3))
        # Body: a pale base the shader tints by species; dark rings along the abdomen.
        ring = (np.sin(y * 900) > 0.6) & (y < 0.01)
        col[:] = (0.75, 0.75, 0.75)
        col[ring] = (0.15, 0.15, 0.15)
        col[pos[:, 1] > 0.012] = (0.45, 0.42, 0.3)
        col[part == 2] = (0.2, 0.35, 0.45)
        col[part == 3] = (0.8, 0.85, 0.9)
        gloss = np.where(part == 3, 0.0, 0.8)
        return col, np.ones(len(pos)), gloss

    def buzz(ph, a=34):
        w = 2 * math.pi * ph
        return {"fw": (a * math.cos(w), -12 * math.sin(w), 0), "hw": (a * math.cos(w + math.pi * 0.6), -12 * math.sin(w + math.pi * 0.6), 0), "abdomen": (1.5 * math.sin(w), 0, 0)}

    clips = {
        "buzz": (16, True, cycle(buzz, 1.0, 16)),
        "perch": (6, True, [(0, {"fw": (-6, 0, 4), "hw": (-8, 0, -4), "abdomen": (4, 0, 0)}), (1.5, {"fw": (-5, 0, 4), "hw": (-7, 0, -4), "abdomen": (5, 0, 0), "head": (0, 0, 15)}), (3.0, {"fw": (-6, 0, 4), "hw": (-8, 0, -4), "abdomen": (4, 0, 0)})]),
    }
    finish("dragonfly", obs, rig, allowed, 0.0004, paint, clips, {"stride": {}}, 0.04, views=("side", "top"), wing_like=("fw", "hw"))


# =======================================================================================
# Mallard: a boat-shaped body riding the water (origin at the waterline), a curled drake's
# tail, a flat bill. Painted as a drake; the shader turns about half of them into ducks.

def mallard():
    reset()
    sk = Skin(0.008)
    sk.ball((0, -0.03, 0.0), 0.24, (0.72, 1.45, 0.62))
    sk.ball((0, 0.11, 0.03), 0.18, (0.78, 0.95, 0.85))
    sk.ball((0, -0.24, 0.06), 0.09, (0.55, 1.2, 0.5))
    sk.limb([(0, 0.15, 0.08), (0, 0.2, 0.15), (0, 0.23, 0.2)], [0.07, 0.06])
    sk.ball((0, 0.25, 0.22), 0.085, (0.8, 1.15, 0.85))
    sk.limb([(0, -0.29, 0.1), (0, -0.28, 0.14)], [0.02])
    skin = sk.build(0.007, 0.2)
    obs = [skin, eyes(skin, (0.04, 0.27, 0.235), 0.01, sink=0.4)]
    bill = []
    for i in range(7):
        u = i / 6
        bill.append((0, 0.3 + u * 0.075, 0.205 - u * 0.012))
    obs.append(tube(bill, [0.02, 0.022, 0.023, 0.023, 0.022, 0.019, 0.012], 1, segs=10, name="bill"))
    obs[-1].scale = (1.0, 1.0, 0.45)
    obs[-1].location = (0, 0, 0.205 * 0.55)
    obs[-1]["bone"] = "head"
    rig = Rig()
    rig.bone("body", (0, -0.15, 0.03), (0, 0.1, 0.03), None, 0.2)
    rig.bone("neck", (0, 0.14, 0.08), (0, 0.22, 0.18), "body", 0.07)
    rig.bone("head", (0, 0.22, 0.2), (0, 0.38, 0.19), "neck", 0.08)
    rig.bone("tail", (0, -0.2, 0.05), (0, -0.3, 0.1), "body", 0.07)
    allowed = {0: None, 2: ["head"]}

    def paint(pos, nrm, part, fs, fc, up):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        fine = fine_noise(pos, 80)
        grey = np.array((0.7, 0.7, 0.68)) * (0.92 + 0.08 * fine[:, None])
        col = grey.copy()
        green = np.array((0.04, 0.32, 0.12))
        head = (z > 0.13) & (y > 0.16)
        col = np.where(head[:, None], green, col)
        collar = (z > 0.1) & (z < 0.13) & (y > 0.14)
        col = np.where(collar[:, None], np.array((0.95, 0.95, 0.92)), col)
        breast = (y > 0.04) & (z <= 0.1) & (z > -0.08) & ~collar & (y > 0.08)
        col = np.where(breast[:, None], np.array((0.45, 0.22, 0.12)), col)
        back = (nrm[:, 2] > 0.6) & (y < 0.08) & (y > -0.18)
        col = np.where(back[:, None], np.array((0.45, 0.42, 0.38)), col)
        spec = (np.abs(x) > 0.12) & (y > -0.12) & (y < -0.04) & (z > 0.04)
        col = np.where(spec[:, None], np.array((0.15, 0.2, 0.75)), col)
        tail = y < -0.22
        col = np.where(tail[:, None], np.array((0.05, 0.05, 0.05)), col)
        col = np.where((tail & (z < 0.04))[:, None], np.array((0.9, 0.9, 0.88)), col)
        col[part == 1] = (0.9, 0.78, 0.2)
        col[part == 2] = (0.05, 0.04, 0.03)
        gloss = np.where(head, 0.8, 0.0)
        gloss[part == 2] = 1.0
        return np.clip(col, 0, 1), default_ao(nrm, pos, 0), gloss

    float_ = {}
    def bob(k):
        return {"neck": (6 * k, 0, 0), "head": (-6 * k, 0, 0), "body": (1.5 * k, 0, 0)}
    tip = {"body": (-75, 0, 0), "neck": (-20, 0, 0), "head": (-10, 0, 0), "tail": (30, 0, 0), "loc": (0, 0.05, 0.02)}
    clips = {
        "swim": (12, True, [(0, bob(0)), (0.3, bob(1)), (0.6, bob(0)), (0.9, bob(1)), (1.2, bob(0))]),
        "idle": (12, True, [(0, float_), (1.0, {"head": (0, 0, 30), "neck": (0, 0, 10)}), (2.0, {"head": (0, 0, 30), "neck": (0, 0, 10)}), (2.6, float_),
                            (3.6, {"head": (0, 0, -25)}), (4.4, {"head": (0, 0, -25)}), (5.0, float_), (5.4, {"tail": (0, 0, 20)}), (5.6, {"tail": (0, 0, -20)}), (5.8, {"tail": (0, 0, 15)}), (6.0, float_)]),
        "dabble": (12, False, [(0, float_), (0.4, tip), (0.8, over(tip, tail=(36, 0, 12))), (1.2, over(tip, tail=(32, 0, -12))), (1.8, over(tip, body=(-80, 4, 0))), (2.3, float_), (2.6, bob(1)), (2.9, float_)]),
        "preen": (12, False, [(0, float_), (0.5, {"neck": (-10, 0, 70), "head": (-40, 20, 80)}), (0.8, {"neck": (-12, 0, 72), "head": (-50, 20, 85)}), (1.1, {"neck": (-10, 0, 70), "head": (-40, 20, 80)}),
                              (1.5, {"neck": (-12, 0, 72), "head": (-50, 20, 85)}), (2.0, float_)]),
    }
    finish("mallard", obs, rig, allowed, 0.02, paint, clips, {"stride": {}}, 0.45, look=(0, 0.0, 0.08), views=("side", "front"))


SPECIES = {"butterfly": butterfly, "hare": hare, "deer": roe_deer, "fox": red_fox, "frog": frog, "dragonfly": dragonfly, "mallard": mallard}


def main():
    names = opt("--species", ",".join(SPECIES)).split(",")
    for n in names:
        SPECIES[n]()


main()
