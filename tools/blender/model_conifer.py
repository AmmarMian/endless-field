"""
Models a spruce in Blender (procedural, game-ready) and exports it as glTF for the tree
pipeline (tools/blender/build_tree.py), which then refits cards, builds LODs and impostors.

  blender -b --factory-startup --python tools/blender/model_conifer.py -- <out_dir> [seed] [height]

Uses Poly Haven's fir_tree_01 twig + bark textures (CC0): the trunk and branches are tapered
tubes; the foliage is fir-twig cards laid along every branch of ~40 whorls.
Materials are named *_trunk, *_branches, *_leaves as the pipeline expects.
"""
import bpy, bmesh, math, os, random, sys
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
SEED = int(argv[1]) if len(argv) > 1 else 1
H = float(argv[2]) if len(argv) > 2 else 20.0
NAME = os.path.basename(os.path.normpath(OUT))
TEX = os.path.abspath("assets-src/textures/fir_twig")
rnd = random.Random(SEED)
os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)

# Twig sprites in the fir twig atlas (u0, v0, u1, v1 in image space, v down; stem at the bottom).
TWIGS = [(0.29, 0.40, 0.64, 0.79), (0.63, 0.44, 0.97, 0.84), (0.19, 0.04, 0.42, 0.33), (0.65, 0.00, 0.96, 0.40)]


def material(name, diff, nor=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(diff)
    nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if nor:
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = bpy.data.images.load(nor)
        tn.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def tube(bm, uv, points, radii, sides, v_scale):
    """Tapered tube along `points`; UVs wrap around (u) and run along the length (v)."""
    rings = []
    for i, (p, r) in enumerate(zip(points, radii)):
        d = (points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]).normalized()
        side = d.cross(Vector((0, 0, 1)))
        if side.length < 1e-3:
            side = Vector((1, 0, 0))
        side.normalize()
        up = side.cross(d).normalized()
        rings.append([bm.verts.new(p + (side * math.cos(a) + up * math.sin(a)) * r) for a in [2 * math.pi * k / sides for k in range(sides + 1)]])
    length = 0.0
    vs = [0.0]
    for i in range(1, len(points)):
        length += (points[i] - points[i - 1]).length
        vs.append(length)
    for i in range(len(rings) - 1):
        for k in range(sides):
            f = bm.faces.new((rings[i][k], rings[i][k + 1], rings[i + 1][k + 1], rings[i + 1][k]))
            for loop, (uu, vv) in zip(f.loops, ((k, vs[i]), (k + 1, vs[i]), (k + 1, vs[i + 1]), (k, vs[i + 1]))):
                loop[uv].uv = (uu / sides * 2.0, vv * v_scale)


def card(bm, uv, base, along, across, length, width, sprite):
    """A slightly cupped card (2x2 quads) with its stem at `base`, pointing `along`."""
    u0, v0, u1, v1 = sprite
    normal = along.cross(across).normalized()
    grid = []
    for j in range(3):
        row = []
        for i in range(3):
            s = i / 2 - 0.5
            t = j / 2
            cup = (s * s) * 0.25 * width
            p = base + along * (t * length) + across * (s * width) + normal * cup
            row.append((bm.verts.new(p), (u0 + (u1 - u0) * (i / 2), 1 - (v1 - (v1 - v0) * t))))
        grid.append(row)
    for j in range(2):
        for i in range(2):
            quad = (grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i])
            f = bm.faces.new([q[0] for q in quad])
            for loop, q in zip(f.loops, quad):
                loop[uv].uv = q[1]


trunk_bm, branch_bm, leaf_bm = bmesh.new(), bmesh.new(), bmesh.new()
uvs = [b.loops.layers.uv.new("UV") for b in (trunk_bm, branch_bm, leaf_bm)]

# Trunk: tapered, with a gentle sway so it is not ruler-straight.
lean = Vector((rnd.uniform(-0.15, 0.15), rnd.uniform(-0.15, 0.15), 0))
trunk_pts = [Vector((0, 0, z)) + lean * (z / H) ** 2 for z in [H * i / 24 for i in range(25)]]
trunk_r = [0.34 * (1 - i / 24) ** 1.15 + 0.015 for i in range(25)]
tube(trunk_bm, uvs[0], trunk_pts, trunk_r, 12, 0.5)

# Whorls of branches with fir-twig cards.
z = 2.8
whorl = 0
golden = math.pi * (3 - math.sqrt(5))
while z < H - 0.6:
    k = (z - 2.8) / (H - 2.8)
    n = rnd.randint(5, 7)
    L = 4.6 * (1 - k) ** 0.85 + 0.35
    center = Vector((0, 0, z)) + lean * (z / H) ** 2
    for b in range(n):
        az = whorl * golden + b * (2 * math.pi / n) + rnd.uniform(-0.25, 0.25)
        dirh = Vector((math.cos(az), math.sin(az), 0))
        elev = math.radians(-18 + 40 * k ** 1.5 + rnd.uniform(-6, 6))
        pts = []
        for i in range(5):
            t = i / 4
            droop = -0.22 * L * t * t * (1 - k)
            pts.append(center + dirh * (L * t * math.cos(elev)) + Vector((0, 0, L * t * math.sin(elev) + droop)))
        radii = [0.06 * (1 - k * 0.6) * (1 - t / 4 * 0.8) for t in range(5)]
        tube(branch_bm, uvs[1], pts, radii, 5, 1.5)
        cards = max(3, round(L / 0.36))
        for c in range(cards):
            t = (c + 0.3) / cards
            seg = min(int(t * 4), 3)
            f = t * 4 - seg
            p = pts[seg].lerp(pts[seg + 1], f)
            along = (pts[seg + 1] - pts[seg]).normalized()
            roll = rnd.uniform(-0.6, 0.6) + (0.5 if c % 2 else -0.5)
            across = Matrix.Rotation(roll, 3, along) @ along.cross(Vector((0, 0, 1))).normalized()
            size = 1.05 + 0.6 * L / 4.6
            card(leaf_bm, uvs[2], p - along * 0.1, along, across, size, size * 0.85, rnd.choice(TWIGS))
    z += 0.42 + rnd.uniform(-0.05, 0.05)
    whorl += 1

# Leader: a few upright cards crown the tip.
tip = Vector((0, 0, H - 0.8)) + lean
for i in range(4):
    a = i * math.pi / 2 + 0.4
    across = Vector((math.cos(a), math.sin(a), 0))
    card(leaf_bm, uvs[2], tip - Vector((0, 0, 0.2)), Vector((0, 0, 1)), across, 1.4, 0.9, TWIGS[2])

mats = {
    "trunk": material(f"{NAME}_trunk", os.path.join(TEX, "bark_diff.png"), os.path.join(TEX, "bark_nor_gl.png")),
    "branches": material(f"{NAME}_branches", os.path.join(TEX, "bark_diff.png"), os.path.join(TEX, "bark_nor_gl.png")),
    "leaves": material(f"{NAME}_leaves", os.path.join(TEX, "twig_diff.png"), os.path.join(TEX, "twig_nor_gl.png")),
}
objs = []
for key, bm in (("trunk", trunk_bm), ("branches", branch_bm), ("leaves", leaf_bm)):
    me = bpy.data.meshes.new(key)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mats[key])
    for p in me.polygons:
        p.use_smooth = key != "leaves"
    ob = bpy.data.objects.new(key, me)
    bpy.context.scene.collection.objects.link(ob)
    objs.append(ob)
bpy.ops.object.select_all(action="DESELECT")
for o in objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]
bpy.ops.object.join()
tri = sum(len(p.vertices) - 2 for p in bpy.context.view_layer.objects.active.data.polygons)
bpy.ops.export_scene.gltf(
    filepath=os.path.join(OUT, f"{NAME}.gltf"),
    export_format="GLTF_SEPARATE",
    export_image_format="JPEG",
    export_texcoords=True,
    export_normals=True,
    export_materials="EXPORT",
    use_selection=True,
)
print(f"[conifer] {NAME}: {tri} tris, height {H}")
