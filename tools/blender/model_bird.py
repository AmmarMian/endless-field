"""
Models a small meadow bird (a tree sparrow) in Blender and exports two poses for the flocks.

  blender -b --factory-startup --python tools/blender/model_bird.py -- <out dir> [--preview <png>]

Body and head are metaballs remeshed into one smooth skin; beak, eyes, a fanned tail of
tapered feathers, and wings either folded along the back (perched) or open with a scalloped
trailing edge of primaries and secondaries (flying). Colours are painted per vertex: chestnut
crown, white cheek with a black spot and bib, streaked brown back, buff belly, dark flight
feathers with pale edges and a white wing bar.

Vertex layout (shared with the other models): p.xyz position, p.w part (0 body, 1 beak,
2 eye, 3 wing, 4 tail, 5 leg); n normal; t.x signed distance along the wing from the shoulder
(0 off the wing; the shader flaps by it), t.y unused; e.rgb albedo, e.w ambient occlusion.
Variant 0 is perched (origin at the feet), variant 1 is flying (origin at the body's centre).
Y-up, facing +Z. Output: bird.bin + bird.json.
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Vector, noise

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None
os.makedirs(OUT, exist_ok=True)

# Shoulder joint (x from the body's midline), and where the hand starts along the wing.
SHOULDER = 0.022
BODY_Z = 0.042  # body centre height above the feet when perched
# "sparrow" or "kingfisher" (set per build in main).
SPECIES = "sparrow"


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def obj_from_bm(bm, name, part):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob["part"] = part
    return ob


def body(z0):
    """Metaball body + head, converted to a smooth mesh."""
    mb = bpy.data.metaballs.new("bodyball")
    mb.resolution = 0.004
    mb.render_resolution = 0.004
    mb.threshold = 0.6
    ob = bpy.data.objects.new("body", mb)
    bpy.context.scene.collection.objects.link(ob)

    def el(kind, co, r, sx=1, sy=1, sz=1, stiff=2.0):
        e = mb.elements.new(type=kind)
        e.co = Vector(co)
        e.radius = r
        e.stiffness = stiff
        if kind == "ELLIPSOID":
            e.size_x, e.size_y, e.size_z = sx, sy, sz
        return e

    # Plump chest, tapering rump, a short neck into a round head (sizes scale the radius).
    el("ELLIPSOID", (0, 0.004, z0), 0.036, 1.0, 1.55, 0.95)
    el("ELLIPSOID", (0, -0.034, z0 + 0.004), 0.026, 0.85, 1.5, 0.7)
    el("BALL", (0, 0.046, z0 + 0.026), 0.032)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.convert(target="MESH")
    ob = bpy.context.view_layer.objects.active
    rm = ob.modifiers.new("remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = 0.0035
    sm = ob.modifiers.new("smooth", "SMOOTH")
    sm.iterations = 6
    sm.factor = 0.6
    dec = ob.modifiers.new("dec", "DECIMATE")
    dec.ratio = 0.35
    bpy.ops.object.modifier_apply(modifier="remesh")
    bpy.ops.object.modifier_apply(modifier="smooth")
    bpy.ops.object.modifier_apply(modifier="dec")
    bpy.ops.object.shade_smooth()
    ob["part"] = 0
    return ob


def beak(z0, skin):
    bm = bmesh.new()
    kf = SPECIES == "kingfisher"
    # A sparrow's short seed-cracker; a kingfisher's long dagger.
    depth = 0.042 if kf else 0.016
    bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.0085 if kf else 0.0072, radius2=0.0004, depth=depth)
    for v in bm.verts:
        # Cone along +Z -> point it forward (+Y), slightly down; a little flattened.
        x, y, z = v.co
        v.co = Vector((x, z + depth / 2, y * 0.8))
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=__import__("mathutils").Matrix.Rotation(-0.25, 3, "X"))
    co, n = surface(skin, (0, 0.09, z0 + 0.022))
    bmesh.ops.translate(bm, verts=bm.verts, vec=co + Vector((0, -0.002, 0)))
    return obj_from_bm(bm, "beak", 1)


def surface(skin, target):
    """The point on the body skin nearest `target`, and its normal."""
    ok, co, n, _ = skin.closest_point_on_mesh(Vector(target))
    return co, n


def eyes(z0, skin):
    bm = bmesh.new()
    for s in (-1, 1):
        co, n = surface(skin, (s * 0.03, 0.058, z0 + 0.034))
        r = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=8, radius=0.0036)
        bmesh.ops.translate(bm, verts=r["verts"], vec=co - n * 0.0012)
    return obj_from_bm(bm, "eyes", 2)


def feather(bm, length, width, root, yaw, pitch, cup=0.0, roll=0.0):
    """A tapered, slightly cupped feather: a strip of quads from root to rounded tip."""
    from mathutils import Matrix

    n = 6
    rows = []
    for i in range(n + 1):
        u = i / n
        w = width * (0.55 + 0.45 * math.sin(math.pi * min(1.0, u * 0.9 + 0.1))) * (1 - u**6 * 0.85)
        row = []
        for s in (-1, 0, 1):
            x = s * w * 0.5
            z = -cup * (1 - (s * s)) * width * 0.15 + cup * u * 0.002
            row.append(bm.verts.new((x, u * length, z)))
        rows.append(row)
    for i in range(n):
        for j in range(2):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    verts = [v for r in rows for v in r]
    m = Matrix.Rotation(yaw, 3, "Z") @ Matrix.Rotation(pitch, 3, "X") @ Matrix.Rotation(roll, 3, "Y")
    for v in verts:
        v.co = m @ v.co + Vector(root)
    return verts


def tail(z0, perched):
    bm = bmesh.new()
    lift = 0.35 if perched else 0.05
    short = 0.6 if SPECIES == "kingfisher" else 1.0
    for i in range(7):
        k = (i - 3) / 3
        feather(bm, (0.058 - abs(k) * 0.006) * short, 0.013, (k * 0.004, -0.05, z0 + 0.004 + abs(k) * 0.001),
                math.pi + k * 0.22, lift + abs(k) * 0.03, cup=0.5)
    ob = obj_from_bm(bm, "tail", 4)
    sol = ob.modifiers.new("sol", "SOLIDIFY")
    sol.thickness = 0.0012
    return ob


def wing_open(side, z0):
    """Open wing: coverts over a fan of secondaries and primaries, scalloped trailing edge."""
    bm = bmesh.new()
    s = side
    # Feathers are built along +Y; yaw turns +Y to the direction (dx, dy).
    def yaw(dx, dy):
        return math.atan2(-dx, dy)

    # Secondaries along the arm (pointing back, a little outward), primaries fanning from the
    # hand from outward to outward-and-back.
    for i in range(7):
        x = SHOULDER + 0.004 + i * 0.0068
        feather(bm, 0.044 + i * 0.0012, 0.011, (s * x, 0.006, z0 + 0.008), yaw(s * 0.08 * i / 6, -1), 0.0, cup=0.3)
    for i in range(8):
        k = i / 7
        a = 0.12 + k * 1.05  # angle back from straight out
        feather(bm, 0.072 - k * 0.024, 0.012, (s * (SHOULDER + 0.048), 0.012 - k * 0.006, z0 + 0.0085), yaw(s * math.cos(a), -math.sin(a)), 0.0, cup=0.3)
    # Coverts: a smooth lobe over the feather roots (leading edge of the wing).
    cov = []
    n = 10
    for i in range(n + 1):
        u = i / n
        x = SHOULDER + u * 0.075
        lead = 0.016 + 0.006 * math.sin(math.pi * u)
        trail = -0.014 - 0.006 * math.sin(math.pi * u) + u * 0.008
        cov.append((bm.verts.new((s * x, lead, z0 + 0.0105)), bm.verts.new((s * x, (lead + trail) / 2, z0 + 0.0115)), bm.verts.new((s * x, trail, z0 + 0.0102))))
    for i in range(n):
        for j in range(2):
            a, b, c, d = cov[i][j], cov[i][j + 1], cov[i + 1][j + 1], cov[i + 1][j]
            bm.faces.new((a, b, c, d) if s > 0 else (d, c, b, a))
    ob = obj_from_bm(bm, f"wing{side}", 3)
    sol = ob.modifiers.new("sol", "SOLIDIFY")
    sol.thickness = 0.0012
    return ob


def wing_folded(side, z0, skin):
    """Folded wing: a long cupped teardrop along the flank, primaries crossing over the rump."""
    bm = bmesh.new()
    s = side
    rows = []
    n = 12
    for i in range(n + 1):
        u = i / n
        y = 0.03 - u * 0.105
        half = 0.016 * math.sin(math.pi * min(1, u * 1.05)) ** 0.7 + 0.001
        zc = z0 + 0.012 - u * 0.012
        row = []
        for j in range(5):
            v = j / 4 * 2 - 1
            x = s * (0.03 - u * 0.018 + (1 - v * v) * 0.004)
            row.append(bm.verts.new((x, y, zc + v * half)))
        rows.append(row)
    for i in range(n):
        for j in range(4):
            a, b, c, d = rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]
            bm.faces.new((a, b, c, d) if s > 0 else (d, c, b, a))
    ob = obj_from_bm(bm, f"wingf{side}", 3)
    # Lie snugly along the flank (the primaries' tips still reach past the rump).
    sw = ob.modifiers.new("hug", "SHRINKWRAP")
    sw.target = skin
    sw.wrap_method = "NEAREST_SURFACEPOINT"
    sw.wrap_mode = "OUTSIDE_SURFACE"
    sw.offset = 0.0025
    vg = ob.vertex_groups.new(name="hug")
    for v in ob.data.vertices:
        # Fade the wrap out toward the tip so the primaries cross over the rump freely.
        vg.add([v.index], max(0.0, min(1.0, (v.co.y + 0.05) / 0.04)), "REPLACE")
    sw.vertex_group = "hug"
    sol = ob.modifiers.new("sol", "SOLIDIFY")
    sol.thickness = 0.0015
    return ob


def legs():
    bm = bmesh.new()
    for s in (-1, 1):
        r = bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=0.0016, radius2=0.0014, depth=0.024)
        bmesh.ops.translate(bm, verts=r["verts"], vec=(s * 0.008, 0.004, 0.012))
        # Toes: three forward, one back.
        for a in (-0.5, 0.0, 0.5, math.pi):
            t = bmesh.ops.create_cone(bm, cap_ends=True, segments=5, radius1=0.0011, radius2=0.0006, depth=0.014)
            from mathutils import Matrix

            m = Matrix.Rotation(a, 3, "Z") @ Matrix.Rotation(-math.pi / 2, 3, "X")
            for v in t["verts"]:
                v.co = m @ (v.co + Vector((0, 0, 0.007)))
            bmesh.ops.translate(bm, verts=t["verts"], vec=(s * 0.008, 0.004, 0.0008))
    return obj_from_bm(bm, "legs", 5)


def build(perched):
    z0 = BODY_Z if perched else 0.0
    skin = body(z0)
    obs = [skin, beak(z0, skin), eyes(z0, skin), tail(z0, perched)]
    if perched:
        obs += [wing_folded(1, z0, skin), wing_folded(-1, z0, skin), legs()]
    else:
        obs += [wing_open(1, z0), wing_open(-1, z0)]
    return obs, z0


def paint(pos, nrm, part, z0):
    """Per-vertex albedo (Blender space: x right, y forward, z up)."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2] - z0
    col = np.zeros((len(pos), 3), np.float32)
    streak = np.array([noise.noise(Vector((px * 160, py * 40, pz * 160))) for px, py, pz in pos])
    # Body: buff-grey belly, warm brown back with dark streaks, chestnut crown, white cheek
    # with a black spot, black bib under the beak.
    back = np.array([0.42, 0.27, 0.14]) * (0.85 + 0.3 * streak[:, None])
    back = np.where((streak[:, None] > 0.25), np.array([0.12, 0.08, 0.05]), back)
    belly = np.array([0.72, 0.66, 0.56])
    up = np.clip((nrm[:, 2] + 0.15) * 2.0, 0, 1)[:, None]
    c = belly * (1 - up) + back * up
    head = (y > 0.035)
    crown = head & (z > 0.03)
    cheek = head & (np.abs(x) > 0.012) & (z > 0.012) & (z < 0.032)
    spot = cheek & (np.abs(y - 0.048) < 0.007) & (np.abs(z - 0.022) < 0.006)
    bib = head & (np.abs(x) < 0.012) & (z < 0.02) & (y > 0.05)
    c = np.where(crown[:, None], np.array([0.45, 0.2, 0.09]), c)
    c = np.where(cheek[:, None], np.array([0.86, 0.84, 0.8]), c)
    c = np.where(spot[:, None], np.array([0.06, 0.05, 0.05]), c)
    c = np.where(bib[:, None], np.array([0.07, 0.06, 0.06]), c)
    if SPECIES == "kingfisher":
        # Brilliant blue-turquoise above (a paler stripe down the back), warm orange below,
        # an orange cheek and a white throat and neck patch.
        blue = np.array([0.02, 0.36, 0.62]) * (0.85 + 0.3 * streak[:, None])
        blue = np.where(((np.abs(x) < 0.006) & (y < 0.03))[:, None], np.array([0.2, 0.75, 0.85]), blue)
        orange = np.array([0.9, 0.42, 0.1])
        c = orange * (1 - up) + blue * up
        cheek = head & (np.abs(x) > 0.014) & (z > 0.014) & (z < 0.03)
        throat = head & (np.abs(x) < 0.014) & (z < 0.016) & (y > 0.05)
        c = np.where(cheek[:, None], np.array([0.92, 0.45, 0.12]), c)
        c = np.where(throat[:, None], np.array([0.92, 0.9, 0.85]), c)
        col[part == 0] = c[part == 0]
        col[part == 1] = (0.05, 0.04, 0.04)
        col[part == 2] = (0.02, 0.02, 0.025)
        col[part == 5] = (0.85, 0.3, 0.15)
        col[part == 4] = blue[part == 4] * 0.85
        wing = np.array([0.03, 0.3, 0.5]) * (0.85 + 0.3 * streak[:, None])
        coverts = (z > 0.0101) & (y > -0.004)
        wing = np.where(coverts[:, None], np.array([0.06, 0.5, 0.7]), wing)
        col[part == 3] = wing[part == 3]
        return np.clip(col, 0, 1)
    col[part == 0] = c[part == 0]
    col[part == 1] = (0.16, 0.14, 0.13)
    col[part == 2] = (0.02, 0.02, 0.025)
    col[part == 5] = (0.45, 0.36, 0.3)
    # Tail: dark brown, paler edges.
    tail = np.array([0.22, 0.15, 0.1]) * (0.9 + 0.2 * streak[:, None])
    col[part == 4] = tail[part == 4]
    # Wings: chestnut coverts with a white bar, dark flight feathers edged buff.
    r = np.abs(x)
    wing = np.array([0.2, 0.13, 0.08]) * (0.85 + 0.3 * streak[:, None])
    coverts = (z > 0.0101) & (y > -0.004)
    wing = np.where(coverts[:, None], np.array([0.5, 0.28, 0.12]), wing)
    bar = (np.abs(y + 0.003) < 0.0022) & (r < 0.09) & (part == 3)
    wing = np.where(bar[:, None], np.array([0.85, 0.82, 0.76]), wing)
    col[part == 3] = wing[part == 3]
    return np.clip(col, 0, 1)


def export(obs, z0, perched):
    dg = bpy.context.evaluated_depsgraph_get()
    pos_l, nrm_l, part_l, idx_l = [], [], [], []
    base = 0
    for ob in obs:
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
        pos_l.append(co.reshape(-1, 3))
        nrm_l.append(nr.reshape(-1, 3))
        part_l.append(np.full(n, ob["part"], np.float32))
        idx_l.append(tris.reshape(-1, 3) + base)
        base += n
        ev.to_mesh_clear()
    pos = np.concatenate(pos_l)
    nrm = np.concatenate(nrm_l)
    part = np.concatenate(part_l)
    idx = np.concatenate(idx_l)
    col = paint(pos, nrm, part, z0)
    # Occlusion: underside and the wing roots darker.
    ao = np.clip(0.6 + 0.4 * (nrm[:, 2] * 0.5 + 0.5), 0, 1)
    span = np.where(part == 3, np.sign(pos[:, 0]) * np.maximum(np.abs(pos[:, 0]) - SHOULDER, 0), 0)
    # Blender (x, y fwd, z up) -> game (x, y up, z fwd); winding flips with the handedness.
    gp = np.stack([pos[:, 0], pos[:, 2], pos[:, 1]], 1)
    gn = np.stack([nrm[:, 0], nrm[:, 2], nrm[:, 1]], 1)
    idx = idx[:, [0, 2, 1]]
    v = np.zeros(len(gp), dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
    v["p"][:, :3] = gp
    v["p"][:, 3] = part
    v["n"][:, :3] = np.clip(np.round(gn * 127), -127, 127)
    v["t"][:, 0] = span
    v["e"] = np.clip(np.round(np.concatenate([col, ao[:, None]], 1) * 255), 0, 255)
    print(f"[bird] {'perched' if perched else 'flying'}: {len(idx)} tris", flush=True)
    return v, idx.reshape(-1)


def main():
    vbytes, ibytes, variants = [], [], []
    vbase = ioff = 0
    global SPECIES
    for SPECIES, perched in (("sparrow", True), ("sparrow", False), ("kingfisher", True), ("kingfisher", False)):
        reset()
        obs, z0 = build(perched)
        if PREVIEW and SPECIES == "kingfisher":
            render_preview(obs, perched)
        v, idx = export(obs, z0, perched)
        vbytes.append(v.tobytes())
        ibytes.append((idx + vbase).astype("<u4").tobytes())
        variants.append({"name": f"{SPECIES}-{'perched' if perched else 'flying'}", "firstIndex": ioff, "indexCount": int(idx.size)})
        vbase += len(v)
        ioff += idx.size
    with open(os.path.join(OUT, "bird.bin"), "wb") as f:
        f.write(b"".join(vbytes))
        f.write(b"".join(ibytes))
    json.dump({"vertexBytes": vbase * 20, "indexCount": ioff, "variants": variants, "shoulder": SHOULDER,
               "credit": "Modeled in Blender (tools/blender/model_bird.py)"},
              open(os.path.join(OUT, "bird.json"), "w"), indent=2)


def render_preview(obs, perched):
    scene = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in obs:
        mat = bpy.data.materials.new(f"m{ob.name}")
        mat.use_nodes = True
        b = mat.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = {0: (0.45, 0.32, 0.2, 1), 1: (0.1, 0.1, 0.1, 1), 2: (0, 0, 0, 1), 3: (0.3, 0.18, 0.1, 1), 4: (0.2, 0.14, 0.1, 1), 5: (0.5, 0.4, 0.3, 1)}[ob["part"]]
        ob.data.materials.append(mat)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(50), 0, math.radians(40))
    scene.collection.objects.link(so)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.65, 0.75, 1)
    scene.world = world
    cam = bpy.data.cameras.new("cam")
    cam.lens = 60
    co = bpy.data.objects.new("cam", cam)
    co.location = (0.22, 0.2, 0.18 if perched else 0.2)
    target = Vector((0, 0, BODY_Z if perched else 0))
    co.rotation_euler = (target - co.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 700
    scene.render.resolution_y = 500
    path = PREVIEW.replace(".png", "-perched.png" if perched else "-flying.png")
    scene.render.filepath = os.path.abspath(path)
    bpy.ops.render.render(write_still=True)
    print("[bird] preview", path, flush=True)


main()
