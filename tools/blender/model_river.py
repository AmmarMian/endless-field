"""
Models river-side flora and stones in Blender; each family exports as a glTF whose top-level
nodes are its variants (the layout tools/blender/build_plants.py expects).

  blender -b --factory-startup --python tools/blender/model_river.py -- <assets-src dir>

Families: lilies (pads + blooms, painted atlas), irises (sword-leaf clumps with yellow flag
flowers, painted atlas), riverrocks (water-worn boulders, Poly Haven lichen_rock).
"""
import bpy, bmesh, math, os, random, sys
import numpy as np
from mathutils import Matrix, Vector, noise

OUT = sys.argv[sys.argv.index("--") + 1]
TEX = os.path.abspath("assets-src/textures")
rnd = random.Random(33)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def save_image(name, img, path):
    n = img.shape[0]
    im = bpy.data.images.new(name, n, n, alpha=True)
    im.pixels.foreach_set(img[::-1].astype(np.float32).ravel())
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    return path


def material(name, image_path):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    t = m.node_tree.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(image_path)
    m.node_tree.links.new(t.outputs["Color"], m.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def finish(bm, name, mat, offset):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = offset
    bpy.context.scene.collection.objects.link(ob)


def export(family):
    d = os.path.join(OUT, family)
    os.makedirs(d, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(d, f"{family}.gltf"), export_format="GLTF_SEPARATE", export_image_format="JPEG")
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.context.scene.objects if o.type == "MESH")
    print(f"[river] {family}: {len(bpy.context.scene.objects)} variants, {tris} tris", flush=True)


def quad_grid(bm, uv, fn, nu, nv):
    """Grid surface: fn(u, v) -> (position, uv)."""
    rows = [[None] * (nu + 1) for _ in range(nv + 1)]
    for j in range(nv + 1):
        for i in range(nu + 1):
            p, t = fn(i / nu, j / nv)
            rows[j][i] = (bm.verts.new(p), t)
    for j in range(nv):
        for i in range(nu):
            q = (rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i])
            f = bm.faces.new([a for a, _ in q])
            for loop, (_, t) in zip(f.loops, q):
                loop[uv].uv = t


# ---------------------------------------------------------------- lilies
def paint_lily_atlas(path):
    """Left half: pad (radial veins, glossy green); right top: petal (white to pink); right
    bottom: golden stamens."""
    n = 512
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.ones((n, n, 4))
    # Pad, in polar coordinates around (0.25, 0.5).
    px, py = (xx - 0.25) / 0.25, (yy - 0.5) / 0.5
    r = np.sqrt(px * px + py * py)
    a = np.arctan2(py, px)
    veins = 0.5 + 0.5 * np.cos(a * 22) ** 40
    pad = np.stack([0.12 + 0.05 * r, 0.30 + 0.12 * (1 - r), 0.07 + 0.03 * r], -1)
    pad = pad * (0.85 + 0.25 * veins[..., None]) * (1 - 0.25 * (r > 0.92)[..., None])
    # Petals.
    pv = yy * 2
    petal = np.stack([0.97 - 0.05 * pv, 0.88 - 0.35 * pv ** 2, 0.9 - 0.25 * pv ** 2], -1)
    stamen = np.stack([0.95 + 0 * xx, 0.68 + 0.1 * np.sin(xx * 90) ** 2, 0.12 + 0 * xx], -1)
    right = np.where((yy < 0.5)[..., None], petal, stamen)
    img[..., :3] = np.where((xx < 0.5)[..., None], pad, right)
    return save_image("lily_atlas", img, path)


def lily_pad(bm, uv, center, radius, rot):
    """Disc with a notch, edges curling up a little; UVs in the left half of the atlas."""
    seg, rings = 18, 3
    notch = 0.35
    verts = []
    c = bm.verts.new(center + Vector((0, 0, 0.01)))
    for j in range(1, rings + 1):
        row = []
        rr = radius * j / rings
        for i in range(seg + 1):
            a = notch / 2 + (2 * math.pi - notch) * i / seg + rot
            lift = 0.035 * radius * (j / rings) ** 3 + 0.01
            p = center + Vector((math.cos(a) * rr, math.sin(a) * rr, lift))
            row.append((bm.verts.new(p), (0.25 + 0.24 * math.cos(a - rot) * j / rings, 0.5 + 0.48 * math.sin(a - rot) * j / rings)))
        verts.append(row)
    for i in range(seg):
        f = bm.faces.new((c, verts[0][i][0], verts[0][i + 1][0]))
        for loop, t in zip(f.loops, ((0.25, 0.5), verts[0][i][1], verts[0][i + 1][1])):
            loop[uv].uv = t
    for j in range(rings - 1):
        for i in range(seg):
            q = (verts[j][i], verts[j + 1][i], verts[j + 1][i + 1], verts[j][i + 1])
            f = bm.faces.new([v for v, _ in q])
            for loop, (_, t) in zip(f.loops, q):
                loop[uv].uv = t


def lily_flower(bm, uv, center, size):
    """Three rings of pointed petals around a golden center."""
    for ring, (count, tilt, length) in enumerate(((8, 0.35, 1.0), (8, 0.75, 0.9), (6, 1.15, 0.72))):
        for k in range(count):
            az = (k + 0.5 * ring) / count * 2 * math.pi
            d = Vector((math.cos(az), math.sin(az), 0))
            side = Vector((-d.y, d.x, 0))
            L = size * length

            def fn(u, v, d=d, side=side, L=L):
                elev = (math.pi / 2 - tilt) * (0.35 + 0.65 * v) + 0.0
                w = math.sin(math.pi * min(v, 0.999) ** 0.8) * 0.32 * L
                along = d * math.cos(elev) * v * L + Vector((0, 0, math.sin(elev) * v * L * 0.75))
                cup = Vector((0, 0, ((u - 0.5) * 2) ** 2 * w * 0.3))
                p = center + along + side * (u - 0.5) * 2 * w + cup + Vector((0, 0, 0.02))
                return p, (0.55 + 0.4 * u, 0.55 + 0.42 * v)

            quad_grid(bm, uv, fn, 2, 3)
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=size * 0.18, radius2=size * 0.1, depth=size * 0.12)
    bmesh.ops.translate(bm, verts=res["verts"], vec=center + Vector((0, 0, size * 0.1)))
    for f in {f for v in res["verts"] for f in v.link_faces}:
        for loop in f.loops:
            loop[uv].uv = (0.75, 0.25)


reset()
mat = material("lilies", paint_lily_atlas(os.path.join(TEX, "lily_atlas.png")))
for v in range(4):
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UV")
    pads = [1, 4, 3, 5][v]
    for i in range(pads):
        a = rnd.uniform(0, 2 * math.pi)
        rr = 0 if i == 0 else rnd.uniform(0.35, 0.8)
        lily_pad(bm, uv, Vector((math.cos(a) * rr, math.sin(a) * rr, 0)), rnd.uniform(0.18, 0.32), rnd.uniform(0, 2 * math.pi))
    for i in range([1, 0, 1, 2][v]):
        lily_flower(bm, uv, Vector((rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2), 0.02)), rnd.uniform(0.11, 0.15))
    finish(bm, f"lily_{v}", mat, (v * 3, 0, 0))
export("lilies")

# ---------------------------------------------------------------- irises
def paint_iris_atlas(path):
    """Left half: sword leaf (green, paler midrib); right top: yellow falls with brown veins;
    right bottom: stem green."""
    n = 512
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.ones((n, n, 4))
    mid = np.exp(-((xx - 0.25) / 0.02) ** 2)
    leaf = np.stack([0.12 + 0.08 * mid, 0.3 + 0.1 * mid + 0.06 * yy, 0.08 + 0.03 * mid], -1)
    vein = (np.cos((xx - 0.75) * 120) ** 30) * (yy < 0.35)
    petal = np.stack([0.98 + 0 * xx, 0.82 - 0.1 * yy, 0.1 + 0 * xx], -1) * (1 - 0.45 * vein[..., None])
    stem = np.stack([0.2 + 0 * xx, 0.38 + 0 * xx, 0.12 + 0 * xx], -1)
    right = np.where((yy < 0.5)[..., None], petal, stem)
    img[..., :3] = np.where((xx < 0.5)[..., None], leaf, right)
    return save_image("iris_atlas", img, path)


def sword_leaf(bm, uv, base, az, height, width, lean):
    d = Vector((math.cos(az), math.sin(az), 0))
    side = Vector((-d.y, d.x, 0))

    def fn(u, v):
        bend = lean * v * v * height
        w = width * (1 - v ** 1.6) * (1 if v < 0.95 else 0.2)
        p = base + Vector((0, 0, v * height)) + d * bend + side * (u - 0.5) * 2 * w
        return p, (0.15 + 0.2 * u, 0.02 + 0.96 * v)

    quad_grid(bm, uv, fn, 1, 5)


def iris_flower(bm, uv, top, size):
    for k in range(3):
        for falls in (True, False):
            az = k * 2 * math.pi / 3 + (0 if falls else math.pi / 3)
            d = Vector((math.cos(az), math.sin(az), 0))
            side = Vector((-d.y, d.x, 0))

            def fn(u, v, d=d, side=side, falls=falls):
                if falls:
                    p = top + d * (v * size) + Vector((0, 0, size * 0.2 * math.sin(math.pi * v) - size * 0.55 * v * v))
                else:
                    p = top + d * (v * size * 0.3) + Vector((0, 0, v * size * 0.8))
                w = math.sin(math.pi * min(v, 0.999) ** 0.7) * size * (0.32 if falls else 0.22)
                return p + side * (u - 0.5) * 2 * w, (0.55 + 0.4 * u, 0.53 + 0.45 * v)

            quad_grid(bm, uv, fn, 2, 3)


reset()
mat = material("irises", paint_iris_atlas(os.path.join(TEX, "iris_atlas.png")))
for v in range(3):
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UV")
    leaves = [9, 13, 7][v]
    for i in range(leaves):
        az = rnd.uniform(0, 2 * math.pi)
        base = Vector((rnd.uniform(-0.12, 0.12), rnd.uniform(-0.12, 0.12), 0))
        sword_leaf(bm, uv, base, az, rnd.uniform(0.6, 1.05), rnd.uniform(0.022, 0.035), rnd.uniform(0.05, 0.35))
    for i in range([2, 3, 1][v]):
        h = rnd.uniform(0.75, 1.0)
        base = Vector((rnd.uniform(-0.08, 0.08), rnd.uniform(-0.08, 0.08), 0))
        res = bmesh.ops.create_cone(bm, cap_ends=False, segments=5, radius1=0.008, radius2=0.006, depth=h)
        bmesh.ops.translate(bm, verts=res["verts"], vec=base + Vector((0, 0, h / 2)))
        for f in {f for vv in res["verts"] for f in vv.link_faces}:
            for loop in f.loops:
                loop[uv].uv = (0.75, 0.25)
        iris_flower(bm, uv, base + Vector((0, 0, h)), rnd.uniform(0.1, 0.14))
    finish(bm, f"iris_{v}", mat, (v * 2, 0, 0))
export("irises")

# ---------------------------------------------------------------- river rocks
reset()
mat = material("riverrocks", os.path.join(TEX, "lichen_rock", "diff.jpg"))
for v, (size, flat) in enumerate(((0.45, 0.55), (0.9, 0.5), (1.6, 0.45), (0.3, 0.6))):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=size)
    off = Vector((rnd.uniform(0, 50), rnd.uniform(0, 50), rnd.uniform(0, 50)))
    for vert in bm.verts:
        p = vert.co.normalized()
        # Water-worn: low-frequency shape only, no sharp detail.
        d = 1 + 0.22 * noise.noise(p * 1.1 + off) + 0.04 * noise.noise(p * 3.0 + off)
        vert.co = Vector((p.x * size * d * 1.25, p.y * size * d, p.z * size * d * flat))
    uv = bm.loops.layers.uv.new("UV")
    for f in bm.faces:
        n = f.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for loop in f.loops:
            c = loop.vert.co
            loop[uv].uv = tuple(x * 0.4 for x in ((c.y, c.z), (c.x, c.z), (c.x, c.y))[ax])
    finish(bm, f"riverrock_{v}", mat, (v * 5, 0, 0))
export("riverrocks")
