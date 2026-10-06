"""
Models the small Inari shrine at the end of the lantern path, in Blender.

  blender -b --factory-startup --python tools/blender/model_shrine.py -- <out dir> [--preview <png>]

Approach from -X (the path) toward +X:
  torii      vermilion myojin gate: pillars with black sleeves (kamaki), tie beam (nuki),
             plaque strut (gakuzuka), double top lintel (shimaki + kasagi) swept up at the ends
  kitsune    two seated stone foxes (metaballs: haunch, chest, neck, head, snout, ears, legs,
             a brush tail curling up the side) on pedestals, red bibs (yodarekake); one holds
             the jewel, the other the key, as at Inari shrines
  hokora     stone steps, raised floor, posts, plank walls, latticed doors, gabled roof with
             crossed chigi and katsuogi on the ridge, shimenawa rope with zigzag shide, two
             paper chochin lanterns at the eaves
  offering   a small sanbo stand with a lone sake cup and an incense burner with three sticks

Materials are a part id per vertex (p.w):
  0 vermilion wood, 1 black lacquer, 2 granite, 3 straw rope, 4 paper (glows at night),
  5 fox stone, 6 red cloth, 7 cedar / weathered wood, 8 sake, 9 incense ember, 10 bronze,
  11 bark roof (hiwadabuki)
Output: shrine.bin + shrine.json (single mesh; anchor points for smoke and lights). Y-up,
the shrine faces -X (toward the path).
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None
# --torii-only: just the gate, centered at the origin (the lantern path's two gates).
TORII_ONLY = "--torii-only" in argv
os.makedirs(OUT, exist_ok=True)
OBJS = []
# The whole shrine is scaled up at export (a torii of ~5.8 m, foxes ~1.2 m on their pedestals).
SCALE = 1.5


def link(bm, name, part, bevel=0.0, segs=2, angle=40):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob["part"] = part
    if bevel > 0:
        m = ob.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = segs
        m.limit_method = "ANGLE"
        m.angle_limit = math.radians(angle)
    OBJS.append(ob)
    return ob


def box(name, size, at, part, bevel=0.01, rot=None):
    bm = bmesh.new()
    m = Matrix.Translation(at)
    if rot is not None:
        m = m @ rot
    bmesh.ops.create_cube(bm, size=1.0, matrix=m @ Matrix.Diagonal((size[0], size[1], size[2], 1)))
    return link(bm, name, part, bevel)


def cyl(name, r0, r1, a, b, part, segs=16, bevel=0.0):
    """Cylinder / frustum from point a to point b."""
    a, b = Vector(a), Vector(b)
    d = b - a
    bm = bmesh.new()
    rot = Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r0, radius2=r1, depth=d.length,
                          matrix=Matrix.Translation((a + b) / 2) @ rot)
    return link(bm, name, part, bevel)


def lathe(name, profile, segs, at, part):
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        rings.append([bm.verts.new((at[0] + r * math.cos(k / segs * 2 * math.pi), at[1] + r * math.sin(k / segs * 2 * math.pi), at[2] + z)) for k in range(segs)])
    for i in range(len(rings) - 1):
        for k in range(segs):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % segs], rings[i + 1][(k + 1) % segs], rings[i + 1][k]))
    if profile[0][0] > 1e-4:
        bm.faces.new(list(reversed(rings[0])))
    if profile[-1][0] > 1e-4:
        bm.faces.new(rings[-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    return link(bm, name, part)


def swept_beam(name, length, height, depth, lift, at, part, n=24):
    """A lintel whose ends sweep upward (kasagi): a box bent along a curve."""
    bm = bmesh.new()
    rings = []
    for i in range(n + 1):
        u = i / n * 2 - 1
        x = u * length / 2
        z = at[2] + lift * abs(u) ** 2.6
        # Ends are cut on a slant, flaring slightly.
        h = height * (1 + 0.25 * abs(u) ** 4)
        ring = [bm.verts.new((at[0] + dx, at[1] + x, z + dz)) for dx, dz in ((-depth / 2, 0), (depth / 2, 0), (depth / 2, h), (-depth / 2, h))]
        rings.append(ring)
    for i in range(n):
        for k in range(4):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % 4], rings[i + 1][(k + 1) % 4], rings[i + 1][k]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    return link(bm, name, part, 0.008)


def torii(cx):
    span, H = 3.2, 3.7
    for side in (-1, 1):
        y = side * span / 2
        cyl(f"pillar{side}", 0.19, 0.17, (cx, y, 0), (cx, y * 0.985, H), 0, 24)
        cyl(f"kamaki{side}", 0.24, 0.22, (cx, y, -0.05), (cx, y, 0.45), 1, 24, 0.01)
        cyl(f"daiishi{side}", 0.32, 0.3, (cx, y, -0.2), (cx, y, 0.0), 2, 12, 0.01)
    box("nuki", (0.16, span + 1.2, 0.26), (cx, 0, H - 0.75), 0)
    box("gakuzuka", (0.12, 0.22, 0.55), (cx, 0, H - 0.33), 0)
    swept_beam("shimaki", span + 1.6, 0.22, 0.32, 0.12, (cx, 0, H - 0.06), 0)
    swept_beam("kasagi", span + 2.2, 0.2, 0.42, 0.32, (cx, 0, H + 0.16), 1)


def metaball_mesh(name, elements, part, resolution=0.012):
    """Metaballs -> mesh, then voxel remesh + smoothing (a clean, sculpted stone surface)."""
    mb = bpy.data.metaballs.new(name)
    mb.resolution = resolution
    mb.render_resolution = resolution
    mb.threshold = 0.6
    ob = bpy.data.objects.new(name, mb)
    bpy.context.scene.collection.objects.link(ob)
    for kind, co, radius, size, rot in elements:
        el = mb.elements.new(type=kind)
        el.co = co
        el.radius = radius
        el.stiffness = 2.6
        if size is not None:
            el.size_x, el.size_y, el.size_z = size
        if rot is not None:
            el.rotation = rot
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    mo = bpy.data.objects.new(name + "_mesh", me)
    bpy.context.scene.collection.objects.link(mo)
    mo["part"] = part
    rm = mo.modifiers.new("remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = 0.006
    sm = mo.modifiers.new("smooth", "LAPLACIANSMOOTH")
    sm.iterations = 6
    sm.lambda_factor = 0.6
    dec = mo.modifiers.new("dec", "DECIMATE")
    dec.ratio = 0.12
    for p in me.polygons:
        p.use_smooth = True
    OBJS.append(mo)
    return mo


def capsule(a, b, radius):
    """Metaball capsule spanning points a -> b (local x is the capsule axis)."""
    from mathutils import Vector as V
    a, b = V(a), V(b)
    d = b - a
    q = V((1, 0, 0)).rotation_difference(d.normalized())
    return ("CAPSULE", (a + b) / 2, radius, (d.length / 2, 1.0, 1.0), q)


def kitsune(at, facing, holds):
    """A seated fox, ~0.75 m, on a pedestal. `facing` = +1 looks toward +Y side, mirrored."""
    px, py = at
    box(f"pedestal{facing}", (0.62, 0.5, 0.75), (px, py, 0.375), 2, 0.025)
    box(f"plinth{facing}", (0.72, 0.6, 0.09), (px, py, 0.795), 2, 0.012)
    box(f"cap{facing}", (0.56, 0.44, 0.05), (px, py, 0.865), 2, 0.01)
    z0 = 0.89
    # Foxes face down the approach (-X), heads turned slightly toward the path's center.
    def P(x, y, z):
        # local: x forward(-X world), y left/right, z up
        return Vector((px - x, py + y * facing, z0 + z))
    from mathutils import Quaternion
    # Inari fox, seated: slender upright body, long neck, narrow pointed muzzle, tall ears,
    # straight forelegs, hind legs folded along the flanks, a flame-shaped tail raised behind.
    # (Influence radii: the visible surface sits at ~0.6x.)
    def ell(x, y, z, r, sx, sy, sz, rot=None):
        return ("ELLIPSOID", P(x, y, z), r, (sx, sy, sz), rot)
    els = [
        ell(-0.06, 0, 0.15, 0.27, 1.0, 0.85, 0.72),                         # haunch
        capsule(P(-0.03, 0, 0.18), P(0.06, 0, 0.42), 0.15),                  # torso
        ell(0.09, 0, 0.37, 0.15, 0.85, 0.95, 1.15),                          # chest ruff
        capsule(P(0.07, 0, 0.44), P(0.115, 0, 0.64), 0.095),                 # neck
        ell(0.13, 0, 0.69, 0.125, 1.05, 0.85, 0.8),                          # skull
        capsule(P(0.17, 0, 0.675), P(0.29, 0, 0.652), 0.06),                 # muzzle
        ("BALL", P(0.305, 0, 0.648), 0.04, None, None),                       # nose
        ell(0.105, 0.05, 0.795, 0.07, 0.42, 0.62, 1.55, Quaternion((1, 0, 0), math.radians(-12))),   # ears
        ell(0.105, -0.05, 0.795, 0.07, 0.42, 0.62, 1.55, Quaternion((1, 0, 0), math.radians(12))),
    ]
    for side in (-1, 1):
        els.append(capsule(P(0.125, 0.055 * side, 0.04), P(0.105, 0.055 * side, 0.36), 0.055))  # forelegs
        els.append(ell(0.15, 0.055 * side, 0.025, 0.07, 1.5, 0.85, 0.55))                         # forepaws
        els.append(ell(-0.03, 0.125 * side, 0.08, 0.12, 1.25, 0.45, 0.7))                        # folded hind leg
    # Flame tail: rises behind the back, swelling, then tapering to a point that curls forward.
    for i in range(18):
        t = i / 17
        x = -0.23 - 0.05 * math.sin(math.pi * t) + 0.1 * t ** 3
        y = -0.02 * math.sin(math.pi * t)
        z = 0.1 + 0.72 * t
        r = 0.07 + 0.13 * math.sin(math.pi * min(1.0, t * 1.15) ** 0.85) * (1 - t) ** 0.35
        els.append(("BALL", P(x, y, z), max(r, 0.03), None, None))
    fox = metaball_mesh(f"kitsune{facing}", els, 5)
    # Red bib around the neck: a short skirt, longer at the front.
    bm = bmesh.new()
    n = 24
    top, bot = [], []
    for k in range(n):
        a = k / n * 2 * math.pi
        front = max(0.0, math.cos(a))
        c = P(0.095, 0, 0.52)
        r0, r1 = 0.07, 0.115
        top.append(bm.verts.new((c.x - r0 * math.cos(a), c.y + r0 * math.sin(a) * facing, c.z + 0.02)))
        bot.append(bm.verts.new((c.x - r1 * math.cos(a), c.y + r1 * math.sin(a) * facing, c.z - 0.06 - 0.09 * front ** 3)))
    for k in range(n):
        bm.faces.new((top[k], top[(k + 1) % n], bot[(k + 1) % n], bot[k]))
    link(bm, f"bib{facing}", 6)
    # Jewel or key in the mouth.
    if holds == "jewel":
        lathe(f"tama{facing}", [(0.0, -0.035), (0.03, -0.02), (0.034, 0.0), (0.024, 0.025), (0.0, 0.045)], 12, tuple(P(0.275, 0, 0.6)), 5)
    else:
        cyl(f"key{facing}", 0.012, 0.012, tuple(P(0.27, -0.07, 0.63)), tuple(P(0.27, 0.07, 0.63)), 5, 8)
        cyl(f"keybit{facing}", 0.02, 0.02, tuple(P(0.27, 0.07, 0.63)), tuple(P(0.27, 0.07, 0.59)), 5, 8)


def hokora(cx):
    """The little shrine house; its front faces -X."""
    W, D = 2.1, 1.7
    # Stone foundation with two steps.
    box("found", (D + 0.6, W + 0.6, 0.5), (cx, 0, 0.25), 2, 0.02)
    box("step1", (0.34, 1.1, 0.17), (cx - (D + 0.6) / 2 - 0.34, 0, 0.085), 2, 0.012)
    box("step2", (0.34, 1.1, 0.34), (cx - (D + 0.6) / 2 - 0.0, 0, 0.17), 2, 0.012)
    floor = 0.5
    box("deck", (D + 0.2, W + 0.2, 0.06), (cx, 0, floor + 0.03), 7, 0.006)
    base = floor + 0.06
    wallH = 1.45
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"post{sx}{sy}", (0.08, 0.08, wallH), (cx + sx * D / 2, sy * W / 2, base + wallH / 2), 0, 0.004)
    # Plank walls (back and sides) and a latticed double door in front.
    for k in range(9):
        y = -W / 2 + (k + 0.5) * W / 9
        box(f"backplank{k}", (0.03, W / 9 - 0.006, wallH), (cx + D / 2 - 0.02, y, base + wallH / 2), 7, 0.002)
    for sy in (-1, 1):
        for k in range(7):
            x = cx - D / 2 + (k + 0.5) * D / 7
            box(f"side{sy}{k}", (D / 7 - 0.006, 0.03, wallH), (x, sy * (W / 2 - 0.02), base + wallH / 2), 7, 0.002)
    door_x = cx - D / 2 + 0.02
    box("doorframe", (0.05, W - 0.08, 0.06), (door_x, 0, base + wallH - 0.05), 0)
    for leaf in (-1, 1):
        y0 = leaf * (W - 0.1) / 4
        box(f"door{leaf}", (0.025, (W - 0.12) / 2, wallH - 0.12), (door_x + 0.02, y0, base + (wallH - 0.12) / 2), 4, 0.0)
        for i in range(5):
            box(f"latv{leaf}{i}", (0.03, 0.018, wallH - 0.12), (door_x, y0 + ((i + 0.5) / 5 - 0.5) * (W - 0.12) / 2, base + (wallH - 0.12) / 2), 1, 0.0)
        for j in range(7):
            box(f"lath{leaf}{j}", (0.03, (W - 0.12) / 2, 0.018), (door_x, y0, base + (j + 0.5) / 7 * (wallH - 0.12)), 1, 0.0)
    # Gabled roof: two slabs meeting at a ridge along X, deep eaves.
    top = base + wallH
    ridgeH = 0.78
    over_x, over_y = 0.6, 0.5
    half = W / 2 + over_y
    slope = math.atan2(ridgeH, half)
    L = math.hypot(half, ridgeH)
    for sy in (-1, 1):
        rot = Matrix.Rotation(-sy * slope, 4, "X")
        at = Vector((cx, sy * half / 2, top + ridgeH / 2 + 0.05))
        box(f"roof{sy}", (D + 2 * over_x, L + 0.05, 0.07), at, 11, 0.01, rot)
        box(f"roofedge{sy}", (D + 2 * over_x + 0.02, 0.06, 0.1), (cx, sy * (half + 0.0), top + 0.02), 1, 0.004)
    box("ridge", (D + 2 * over_x + 0.1, 0.16, 0.12), (cx, 0, top + ridgeH + 0.1), 1, 0.01)
    for i in range(4):
        x = cx - (D / 2 + 0.1) + i * (D + 0.2) / 3
        cyl(f"katsuogi{i}", 0.055, 0.055, (x, -0.22, top + ridgeH + 0.2), (x, 0.22, top + ridgeH + 0.2), 7, 12)
    for sx in (-1, 1):
        x = cx + sx * (D / 2 + over_x - 0.05)
        for sy in (-1, 1):
            cyl(f"chigi{sx}{sy}", 0.025, 0.025, (x, 0, top + ridgeH + 0.05), (x, sy * 0.32, top + ridgeH + 0.5), 7, 6)
    # Shimenawa across the front eave: a twisted straw rope sagging between the posts.
    bm = bmesh.new()
    n, segs = 48, 10
    rings = []
    for i in range(n + 1):
        u = i / n
        y = (u - 0.5) * (W + 0.2)
        z = top - 0.12 - 0.12 * math.sin(u * math.pi)
        thick = 0.05 + 0.03 * math.sin(u * math.pi)
        c = Vector((cx - D / 2 - 0.12, y, z))
        ring = []
        for k in range(segs):
            a = k / segs * 2 * math.pi + u * 18
            rr = thick * (1 + 0.18 * math.cos(3 * a))
            ring.append(bm.verts.new((c.x + rr * math.cos(a), c.y, c.z + rr * math.sin(a))))
        rings.append(ring)
    for i in range(n):
        for k in range(segs):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % segs], rings[i + 1][(k + 1) % segs], rings[i + 1][k]))
    link(bm, "shimenawa", 3)
    # Shide: zigzag paper streamers hanging from the rope.
    for j, u in enumerate((0.2, 0.4, 0.6, 0.8)):
        y = (u - 0.5) * (W + 0.2)
        z = top - 0.12 - 0.12 * math.sin(u * math.pi) - 0.06
        bm = bmesh.new()
        pts = []
        for s in range(5):
            dz = -s * 0.07
            dy = 0.035 * (1 if s % 2 else -1)
            pts.append((Vector((cx - D / 2 - 0.13, y + dy - 0.03, z + dz)), Vector((cx - D / 2 - 0.13, y + dy + 0.03, z + dz))))
        vs = [(bm.verts.new(a), bm.verts.new(b)) for a, b in pts]
        for s in range(len(vs) - 1):
            bm.faces.new((vs[s][0], vs[s][1], vs[s + 1][1], vs[s + 1][0]))
        link(bm, f"shide{j}", 4)
    # Two chochin paper lanterns hanging under the front eave.
    for sy in (-1, 1):
        c = (cx - D / 2 - 0.3, sy * (W / 2 + 0.05), top - 0.38)
        prof = [(0.06, -0.17), (0.1, -0.15), (0.13, -0.08), (0.14, 0.0), (0.13, 0.08), (0.1, 0.15), (0.06, 0.17)]
        lathe(f"chochin{sy}", prof, 20, c, 4)
        cyl(f"chochincap{sy}", 0.065, 0.065, (c[0], c[1], c[2] + 0.17), (c[0], c[1], c[2] + 0.2), 1, 16)
        cyl(f"chochinbase{sy}", 0.065, 0.065, (c[0], c[1], c[2] - 0.2), (c[0], c[1], c[2] - 0.17), 1, 16)
        cyl(f"chochincord{sy}", 0.006, 0.006, (c[0], c[1], c[2] + 0.2), (c[0], c[1], top - 0.02), 1, 6)
    return top


def offering(x):
    """Sanbo stand with a lone sake cup and an incense burner with three sticks."""
    box("sanbo_base", (0.3, 0.3, 0.22), (x, 0, 0.11), 7, 0.004)
    box("sanbo_top", (0.38, 0.38, 0.03), (x, 0, 0.235), 7, 0.004)
    top = 0.25
    # Sake cup (sakazuki): shallow red lacquer dish on a foot, filled.
    cupx, cupy = x - 0.06, 0.08
    lathe("sakazuki", [(0.02, 0.0), (0.022, 0.012), (0.03, 0.014), (0.05, 0.026), (0.058, 0.036), (0.052, 0.034), (0.0, 0.026)], 24, (cupx, cupy, top), 6)
    lathe("sake", [(0.0, 0.031), (0.05, 0.031)], 24, (cupx, cupy, top), 8)
    # Incense burner (koro): bronze bowl with ash, three sticks.
    kx, ky = x + 0.04, -0.07
    lathe("koro", [(0.04, 0.0), (0.06, 0.02), (0.065, 0.06), (0.06, 0.075), (0.0, 0.07)], 24, (kx, ky, top), 10)
    tips = []
    for i, (dx, dy, lean) in enumerate(((-0.012, 0.0, 0.03), (0.01, 0.01, -0.02), (0.008, -0.012, 0.0))):
        a = (kx + dx, ky + dy, top + 0.07)
        b = (kx + dx + lean, ky + dy + lean * 0.5, top + 0.24)
        cyl(f"incense{i}", 0.0025, 0.0025, a, b, 7, 6)
        cyl(f"ember{i}", 0.004, 0.003, b, (b[0] + lean * 0.05, b[1], b[2] + 0.012), 9, 6)
        tips.append(b)
    return tips


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if TORII_ONLY:
        torii(0.0)
        return 0.0, []
    torii(-6.0)
    kitsune((-2.6, -1.25), 1, "jewel")
    kitsune((-2.6, 1.25), -1, "key")
    shrine_top = hokora(2.0)
    tips = offering(0.25)
    return shrine_top, tips


def export(height):
    dg = bpy.context.evaluated_depsgraph_get()
    pos_l, nrm_l, part_l = [], [], []
    for ob in OBJS:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        n = len(me.vertices)
        co = np.zeros(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        nr = np.zeros(n * 3, np.float32)
        me.vertex_normals.foreach_get("vector", nr)
        tris = np.zeros(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tris)
        M = np.array(ob.matrix_world)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        co = co[tris]
        sm = nr.reshape(-1, 3)[tris]
        fn = np.cross(co[1::3] - co[0::3], co[2::3] - co[0::3])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        fnr = np.repeat(fn, 3, axis=0)
        smooth = ob["part"] in (5, 3, 4)
        nrm = sm if smooth else np.where(np.sum(sm * fnr, axis=1, keepdims=True) > 0.93, sm, fnr)
        pos_l.append(co)
        nrm_l.append(nrm)
        part_l.append(np.full(len(co), ob["part"], np.float32))
        ev.to_mesh_clear()
    co = np.concatenate(pos_l)
    nr = np.concatenate(nrm_l)
    part = np.concatenate(part_l)
    idx = np.arange(len(co)).reshape(-1, 3)[:, [0, 2, 1]]
    pos = np.stack([co[:, 0], co[:, 2], co[:, 1]], 1) * SCALE
    nrm = np.stack([nr[:, 0], nr[:, 2], nr[:, 1]], 1)
    hN = np.clip(pos[:, 1] / (height * SCALE), 0, 1)
    ao = np.clip(0.55 + 0.45 * (nrm[:, 1] * 0.5 + 0.5), 0, 1) * (0.75 + 0.25 * np.clip(pos[:, 1] / 0.9, 0, 1))
    moss = np.clip(np.clip(nrm[:, 1], 0, 1) ** 2 * (1 - hN), 0, 1)
    v = np.zeros(len(pos), dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
    v["p"][:, :3] = pos
    v["p"][:, 3] = part
    v["n"][:, :3] = np.clip(np.round(nrm * 127), -127, 127)
    v["t"] = pos[:, [0, 2]]
    v["e"] = np.clip(np.round(np.stack([hN, moss, ao, np.zeros_like(hN)], 1) * 255), 0, 255)
    return v, idx.reshape(-1)


def main():
    top, tips = build()
    if PREVIEW:
        render_preview()
    height = 5.0
    v, idx = export(height)
    if TORII_ONLY:
        with open(os.path.join(OUT, "torii.bin"), "wb") as f:
            f.write(v.tobytes())
            f.write(idx.astype("<u4").tobytes())
        json.dump({"vertexBytes": len(v) * 20, "indexCount": int(idx.size), "credit": "Modeled in Blender (tools/blender/model_shrine.py --torii-only)"},
                  open(os.path.join(OUT, "torii.json"), "w"), indent=2)
        print(f"[torii] {idx.size // 3} tris", flush=True)
        return
    with open(os.path.join(OUT, "shrine.bin"), "wb") as f:
        f.write(v.tobytes())
        f.write(idx.astype("<u4").tobytes())
    ytip = lambda p: [round(p[0] * SCALE, 3), round(p[2] * SCALE, 3), round(p[1] * SCALE, 3)]
    json.dump({
        "vertexBytes": len(v) * 20,
        "indexCount": int(idx.size),
        # Y-up anchors: incense tips (smoke), chochin centers (light).
        "incense": [ytip(t) for t in tips],
        "lights": [[round((2.0 - 0.85 - 0.3) * SCALE, 3), round((top - 0.38) * SCALE, 3), s * 1.1 * SCALE] for s in (-1, 1)],
        "credit": "Modeled in Blender (tools/blender/model_shrine.py)",
    }, open(os.path.join(OUT, "shrine.json"), "w"), indent=2)
    print(f"[shrine] {idx.size // 3} tris", flush=True)


def render_preview():
    scene = bpy.context.scene
    colors = {0: (0.62, 0.08, 0.04), 1: (0.03, 0.03, 0.03), 2: (0.35, 0.34, 0.31), 3: (0.6, 0.5, 0.28), 4: (0.95, 0.92, 0.85),
              5: (0.7, 0.68, 0.62), 6: (0.65, 0.05, 0.04), 7: (0.33, 0.22, 0.13), 8: (0.85, 0.8, 0.6), 9: (1, 0.4, 0.1), 10: (0.3, 0.22, 0.1),
              11: (0.28, 0.17, 0.1)}
    mats = {}
    for k, c in colors.items():
        m = bpy.data.materials.new(f"m{k}")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*c, 1)
        b.inputs["Roughness"].default_value = 0.6
        mats[k] = m
    for ob in OBJS:
        ob.data.materials.append(mats[ob["part"]])
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.0
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(50), 0, math.radians(-60))
    scene.collection.objects.link(so)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.58, 0.7, 1)
    scene.world = world
    cam = bpy.data.cameras.new("cam")
    cam.lens = 32
    co = bpy.data.objects.new("cam", cam)
    co.location = (-11.5, -4.5, 2.4)
    co.rotation_euler = (Vector((-1.0, 0.0, 1.3)) - co.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 900
    scene.render.filepath = os.path.abspath(PREVIEW)
    bpy.ops.render.render(write_still=True)
    # A close-up of the foxes and the offering.
    co.location = (-4.3, -2.4, 1.5)
    co.rotation_euler = (Vector((-2.4, -0.7, 1.2)) - co.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = os.path.abspath(PREVIEW.replace(".png", "_foxes.png"))
    bpy.ops.render.render(write_still=True)


main()
