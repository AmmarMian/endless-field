"""
Models forest-floor props in Blender and exports each family as a glTF whose top-level
nodes are its variants (the layout tools/blender/build_plants.py expects).

  blender -b --factory-startup --python tools/blender/model_props.py -- <assets-src dir>

Families: mushrooms (procedurally painted atlas: fly agaric + porcini), fallen logs and
stumps (Poly Haven moss_wood), mossy boulders (Poly Haven mossy_rock).
"""
import bpy, bmesh, math, os, random, sys
import numpy as np
from mathutils import Matrix, Vector, noise

OUT = sys.argv[sys.argv.index("--") + 1]
TEX = os.path.abspath("assets-src/textures")
rnd = random.Random(21)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


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
    return ob


def export(family):
    d = os.path.join(OUT, family)
    os.makedirs(d, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(d, f"{family}.gltf"), export_format="GLTF_SEPARATE", export_image_format="JPEG")
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.context.scene.objects if o.type == "MESH")
    print(f"[props] {family}: {len(bpy.context.scene.objects)} variants, {tris} tris", flush=True)


# ---------------------------------------------------------------- mushrooms
def paint_mushroom_atlas(path):
    """256x256: top-left fly agaric cap, bottom-left porcini cap, right half stems."""
    n = 256
    img = np.zeros((n, n, 4), np.float32)
    yy, xx = np.mgrid[0:n, 0:n] / n
    rng = np.random.default_rng(4)
    # Caps: u in [0, 0.5); v (image rows) top half = agaric, bottom half = porcini.
    agaric = np.stack([0.75 - yy * 0.25, 0.08 + yy * 0.05, 0.05 + yy * 0.03], -1)
    spots = np.zeros((n, n))
    for _ in range(40):
        cx, cy, r = rng.uniform(0, 0.5), rng.uniform(0, 0.5), rng.uniform(0.012, 0.03)
        spots = np.maximum(spots, ((xx - cx) ** 2 + (yy - cy) ** 2 < r * r).astype(float))
    agaric = agaric * (1 - spots[..., None]) + np.array([0.95, 0.92, 0.85]) * spots[..., None]
    porcini = np.stack([0.42 - (yy - 0.5) * 0.3, 0.26 - (yy - 0.5) * 0.15, 0.12 + 0.0 * yy], -1)
    caps = np.where((yy < 0.5)[..., None], agaric, porcini)
    stem = np.stack([0.9 - yy * 0.1, 0.86 - yy * 0.12, 0.74 - yy * 0.1], -1)
    img[..., :3] = np.where((xx < 0.5)[..., None], caps, stem)
    img[..., :3] *= (0.92 + 0.08 * rng.random((n, n)))[..., None]
    img[..., 3] = 1
    im = bpy.data.images.new("mushroom_atlas", n, n, alpha=True)
    im.pixels.foreach_set(img[::-1].ravel())
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    return path


def mushroom(bm, uv, base, height, cap_r, kind):
    """Tapered stem + domed cap; kind 0 = agaric (flatter, red), 1 = porcini (fat, brown)."""
    stem_r = cap_r * (0.22 if kind == 0 else 0.42)
    seg = 10
    lean = Vector((rnd.uniform(-0.06, 0.06), rnd.uniform(-0.06, 0.06), 0)) * height
    rings = []
    for j in range(5):
        t = j / 4
        c = base + Vector((0, 0, height * t)) + lean * t * t
        r = stem_r * (1.15 - 0.3 * t + (0.25 if kind == 1 and t < 0.5 else 0))
        rings.append([(bm.verts.new(c + Vector((math.cos(a) * r, math.sin(a) * r, 0))), (0.5 + 0.5 * i / seg, 1 - t)) for i, a in enumerate(np.linspace(0, 2 * math.pi, seg + 1))])
    for j in range(4):
        for i in range(seg):
            q = (rings[j][i], rings[j][i + 1], rings[j + 1][i + 1], rings[j + 1][i])
            f = bm.faces.new([v for v, _ in q])
            for loop, (_, u) in zip(f.loops, q):
                loop[uv].uv = u
    top = base + Vector((0, 0, height)) + lean
    v0 = 0.5 if kind == 0 else 0.0  # atlas half for this cap (UV space, v up)
    dome = 0.45 if kind == 0 else 0.75
    caprings = []
    for j in range(6):
        t = j / 5  # 0 at the rim, 1 at the top
        a = t * math.pi / 2
        r = cap_r * math.cos(a)
        z = cap_r * dome * math.sin(a)
        caprings.append([(bm.verts.new(top + Vector((math.cos(b) * r, math.sin(b) * r, z - cap_r * 0.12))), (0.5 * (0.5 + 0.5 * math.cos(b) * (1 - t)), v0 + 0.5 * (0.5 + 0.5 * math.sin(b) * (1 - t)))) for b in np.linspace(0, 2 * math.pi, seg + 1)])
    for j in range(5):
        for i in range(seg):
            q = (caprings[j][i], caprings[j][i + 1], caprings[j + 1][i + 1], caprings[j + 1][i])
            f = bm.faces.new([v for v, _ in q])
            for loop, (_, u) in zip(f.loops, q):
                loop[uv].uv = u
    # Underside (gills): a cone back to the stem top.
    under = bm.verts.new(top - Vector((0, 0, cap_r * 0.1)))
    for i in range(seg):
        f = bm.faces.new((caprings[0][i + 1][0], caprings[0][i][0], under))
        for loop, u in zip(f.loops, ((0.75, 0.9), (0.75, 0.9), (0.75, 0.95))):
            loop[uv].uv = u


reset()
atlas = paint_mushroom_atlas(os.path.join(TEX, "mushroom_atlas.png"))
mat = material("mushrooms", atlas)
for v in range(4):
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UV")
    kind = v % 2
    count = [1, 3, 5, 2][v]
    for m in range(count):
        p = Vector((rnd.uniform(-0.12, 0.12), rnd.uniform(-0.12, 0.12), 0)) * (count > 1)
        mushroom(bm, uv, p, rnd.uniform(0.08, 0.17) * (1.3 if kind == 0 else 1.0), rnd.uniform(0.05, 0.09), kind)
    finish(bm, f"mushroom_{v}", mat, (v * 1.5, 0, 0))
export("mushrooms")

# ---------------------------------------------------------------- logs and stumps
reset()
mat = material("logs", os.path.join(TEX, "moss_wood", "diff.jpg"))


def log(bm, uv, length, radius, bend, stubs):
    seg, rings = 12, 14
    verts = []
    for j in range(rings + 1):
        t = j / rings
        x = (t - 0.5) * length
        zc = radius + bend * math.sin(t * math.pi) - 0.08
        r = radius * (1 - 0.18 * t) * (1 + 0.05 * noise.noise(Vector((t * 6, 0, 0))))
        row = []
        for i in range(seg + 1):
            a = 2 * math.pi * i / seg
            bumps = 1 + 0.06 * noise.noise(Vector((t * 8, a, 1)))
            row.append((bm.verts.new((x, math.cos(a) * r * bumps, zc + math.sin(a) * r * bumps)), (i / seg * 2, t * length / 1.5)))
        verts.append(row)
    for j in range(rings):
        for i in range(seg):
            q = (verts[j][i], verts[j + 1][i], verts[j + 1][i + 1], verts[j][i + 1])
            f = bm.faces.new([v for v, _ in q])
            for loop, (_, u) in zip(f.loops, q):
                loop[uv].uv = u
    for end in (0, rings):
        c = bm.verts.new(sum((v.co for v, _ in verts[end][:-1]), Vector()) / seg)
        for i in range(seg):
            tri = (verts[end][i][0], verts[end][i + 1][0], c) if end else (verts[end][i + 1][0], verts[end][i][0], c)
            f = bm.faces.new(tri)
            for loop in f.loops:
                loop[uv].uv = (0.5 + 0.2 * (loop.vert.co.y / radius), 0.5 + 0.2 * (loop.vert.co.z - radius) / radius)
    for s in range(stubs):
        t = rnd.uniform(0.2, 0.8)
        x = (t - 0.5) * length
        a = rnd.uniform(0.3, 2.8)
        root = Vector((x, math.cos(a) * radius * 0.9, radius + math.sin(a) * radius * 0.9))
        dirv = Vector((rnd.uniform(-0.3, 0.3), math.cos(a), math.sin(a))).normalized()
        res = bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=radius * 0.25, radius2=radius * 0.12, depth=radius * 1.2)
        rot = dirv.to_track_quat("Z", "Y").to_matrix().to_4x4()
        bmesh.ops.transform(bm, matrix=Matrix.Translation(root + dirv * radius * 0.5) @ rot, verts=res["verts"])


def stump(bm, uv, radius, height):
    seg = 14
    rows = []
    for j in range(5):
        t = j / 4
        flare = 1 + 0.6 * (1 - t) ** 3
        r = radius * flare
        top = height * t
        row = []
        for i in range(seg + 1):
            a = 2 * math.pi * i / seg
            root = 1 + (0.25 * max(0, math.sin(a * 5)) * (1 - t) ** 2)
            jag = (rnd.uniform(-0.06, 0.06) if j == 4 else 0)
            row.append((bm.verts.new((math.cos(a) * r * root, math.sin(a) * r * root, top + jag - 0.1)), (i / seg * 2, t * height)))
        rows.append(row)
    for j in range(4):
        for i in range(seg):
            q = (rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i])
            f = bm.faces.new([v for v, _ in q])
            for loop, (_, u) in zip(f.loops, q):
                loop[uv].uv = u
    c = bm.verts.new((0, 0, height - 0.12))
    for i in range(seg):
        f = bm.faces.new((rows[4][i][0], rows[4][i + 1][0], c))
        for loop in f.loops:
            loop[uv].uv = (0.5 + 0.25 * loop.vert.co.x / radius, 0.5 + 0.25 * loop.vert.co.y / radius)


for v, (L, R, bend, stubs) in enumerate(((5.5, 0.38, 0.1, 3), (3.8, 0.28, -0.05, 2), (7.0, 0.45, 0.18, 4))):
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UV")
    log(bm, uv, L, R, bend, stubs)
    finish(bm, f"log_{v}", mat, (v * 9, 0, 0))
bm = bmesh.new()
uv = bm.loops.layers.uv.new("UV")
stump(bm, uv, 0.42, 0.75)
finish(bm, "stump_0", mat, (30, 0, 0))
export("logs")

# ---------------------------------------------------------------- mossy boulders
reset()
mat = material("rocks", os.path.join(TEX, "mossy_rock", "diff.jpg"))
for v, (size, flat) in enumerate(((0.7, 0.75), (1.4, 0.6), (2.3, 0.55))):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=size)
    off = Vector((rnd.uniform(0, 50), rnd.uniform(0, 50), rnd.uniform(0, 50)))
    for vert in bm.verts:
        p = vert.co.normalized()
        d = 1 + 0.28 * noise.noise(p * 1.6 + off) + 0.1 * noise.noise(p * 4.5 + off)
        vert.co = Vector((p.x * size * d * 1.15, p.y * size * d, max(p.z * size * d * flat, -size * 0.15)))
    uv = bm.loops.layers.uv.new("UV")
    for f in bm.faces:
        n = f.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for loop in f.loops:
            c = loop.vert.co
            loop[uv].uv = ((c.y, c.z), (c.x, c.z), (c.x, c.y))[ax]
            loop[uv].uv = (loop[uv].uv[0] * 0.35, loop[uv].uv[1] * 0.35)
    finish(bm, f"rock_{v}", mat, (v * 8, 0, 0))
export("rocks")
