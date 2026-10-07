"""
Models a barn swallow (Hirundo rustica) in flight for the player's bird, in Blender:

  blender -b --factory-startup --python tools/blender/model_swallow.py -- <out dir> [--preview <png>]

Built for close viewing (it is the protagonist):
  body     metaballs (head, breast, belly, rump) remeshed into one smooth skin, then a light
           smoothing pass; small wide-gaped bill; glossy eyes snapped onto the head
  wings    long and scythe-shaped: 9 primaries swept back from the hand, 9 secondaries along
           the short arm, tertials, then three rows of coverts overlapping toward a clean
           leading edge (each feather a tapered, cambered, slightly twisted blade)
  tail     12 rectrices in a deep fork: long thin outer streamers, short inner feathers, each
           with a white spot near the tip
  paint    glossy steel-blue upperparts, chestnut-red forehead and throat, a blue-black breast
           band, cream-buff underparts and underwing coverts (vertex colours)

Vertex layout (shared with the other birds): p.xyz position, p.w part (0 body, 1 bill, 2 eye,
3 wing, 4 tail); n normal; t.x = signed distance along the wing from the shoulder (0 off the
wing), t.y = 1 on glossy upperparts; e.rgb albedo, e.w ambient occlusion. Y-up, facing +Z,
real size (about 0.18 m long, 0.33 m span); the game scales it. Output: swallow.bin/.json.
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector, noise

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None
os.makedirs(OUT, exist_ok=True)

SHOULDER = 0.016
ELBOW = 0.045  # span distance where the hand begins


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


# ---------------------------------------------------------------------------------------
# Body


def body():
    mb = bpy.data.metaballs.new("skin")
    mb.resolution = 0.0028
    mb.render_resolution = 0.0028
    mb.threshold = 0.6
    ob = bpy.data.objects.new("body", mb)
    bpy.context.scene.collection.objects.link(ob)

    def el(kind, co, r, sx=1.0, sy=1.0, sz=1.0, stiff=2.0):
        e = mb.elements.new(type=kind)
        e.co = Vector(co)
        e.radius = r
        e.stiffness = stiff
        if kind == "ELLIPSOID":
            e.size_x, e.size_y, e.size_z = sx, sy, sz

    # Streamlined: a deep breast, a slim belly and rump tapering into the tail, a flat
    # broad head on a short neck (swallows look all wing and tail).
    el("ELLIPSOID", (0, 0.012, 0.0), 0.026, 0.95, 1.6, 0.85)
    el("ELLIPSOID", (0, -0.022, 0.001), 0.02, 0.8, 1.7, 0.7)
    el("ELLIPSOID", (0, -0.05, 0.003), 0.012, 0.75, 1.8, 0.55)
    el("ELLIPSOID", (0, 0.046, 0.008), 0.02, 1.0, 1.05, 0.88)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.convert(target="MESH")
    ob = bpy.context.view_layer.objects.active
    rm = ob.modifiers.new("remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = 0.0022
    bpy.ops.object.modifier_apply(modifier="remesh")
    sm = ob.modifiers.new("smooth", "SMOOTH")
    sm.iterations = 8
    sm.factor = 0.7
    bpy.ops.object.modifier_apply(modifier="smooth")
    dec = ob.modifiers.new("dec", "DECIMATE")
    dec.ratio = 0.3
    bpy.ops.object.modifier_apply(modifier="dec")
    bpy.ops.object.shade_smooth()
    ob["part"] = 0
    return ob


def surface(skin, target):
    ok, co, n, _ = skin.closest_point_on_mesh(Vector(target))
    return co, n


def bill(skin):
    """Short and broad: a flattened wedge with a wide gape."""
    bm = bmesh.new()
    tip = Vector((0, 0.0, 0))
    pts = [Vector((-0.0065, -0.008, 0.001)), Vector((0.0065, -0.008, 0.001)), Vector((0, -0.008, 0.0045)), Vector((0, -0.008, -0.0025))]
    t = bm.verts.new(tip)
    vs = [bm.verts.new(p) for p in pts]
    for a, b in ((0, 2), (2, 1), (1, 3), (3, 0)):
        bm.faces.new((t, vs[a], vs[b]))
    bm.faces.new((vs[0], vs[3], vs[1], vs[2]))
    co, _ = surface(skin, (0, 0.08, 0.006))
    bmesh.ops.translate(bm, verts=bm.verts, vec=co + Vector((0, 0.0065, -0.0005)))
    return obj_from_bm(bm, "bill", 1)


def eyes(skin):
    bm = bmesh.new()
    for s in (-1, 1):
        co, n = surface(skin, (s * 0.03, 0.058, 0.013))
        r = bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.0024)
        bmesh.ops.translate(bm, verts=r["verts"], vec=co - n * 0.0012)
    return obj_from_bm(bm, "eyes", 2)


# ---------------------------------------------------------------------------------------
# Feathers


def feather(bm, length, width, root, direction, lift=0.0, twist=0.0, camber=0.15, rows=7):
    """A tapered blade: rounded tip, shaft along the middle, cambered, optionally twisted.
    `direction` is a unit vector (in the wing plane) from root to tip."""
    d = Vector(direction).normalized()
    up = Vector((0, 0, 1))
    side = d.cross(up).normalized()
    grid = []
    for i in range(rows + 1):
        u = i / rows
        # Narrow at the root, widest near two thirds, rounded at the tip.
        w = width * (0.45 + 0.55 * math.sin(math.pi * min(1.0, 0.2 + u * 0.85))) * (1 - u ** 8 * 0.7)
        tw = twist * u
        row = []
        for j in (-1, 0, 1):
            h = camber * width * (1 - j * j) * 0.5 + lift * u * length
            p = Vector(root) + d * (u * length) + side * (j * w * 0.5 * math.cos(tw)) + up * (h + j * w * 0.5 * math.sin(tw))
            row.append(bm.verts.new(p))
        grid.append(row)
    for i in range(rows):
        for j in range(2):
            bm.faces.new((grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]))


def wing(side):
    """Open wing, spread for gliding: the shader flaps it about the shoulder."""
    s = side
    bm = bmesh.new()
    z = 0.006
    # Leading edge curve: shoulder -> wrist (short arm) -> hand sweeping back to a long tip.
    wrist = Vector((s * (SHOULDER + 0.04), 0.012, z))

    def ang(deg):
        r = math.radians(deg)
        return Vector((s * math.cos(r), -math.sin(r), 0))

    # Primaries: overlapping into one solid, pointed hand; the outermost is the longest
    # (the swallow's scythe tip), each swept a little further back.
    for i in range(9):
        k = i / 8  # 0 inner .. 1 outer
        length = 0.062 + 0.058 * k ** 0.8
        a = 10 + (1 - k) * 26
        root = wrist + Vector((s * k * 0.004, -k * 0.003, 0.00025 * i))
        feather(bm, length, 0.021, root, ang(a + 6), lift=0.006, twist=-0.12 * s * k, camber=0.22, rows=9)
    # Secondaries along the arm, pointing back, overlapping.
    for i in range(10):
        k = i / 9
        root = Vector((s * (SHOULDER + 0.003 + k * 0.038), 0.006 - k * 0.002, z - 0.0004))
        feather(bm, 0.04 - k * 0.004, 0.016, root, ang(82 - k * 16), camber=0.2)
    # Tertials, near the body.
    for i in range(3):
        root = Vector((s * (SHOULDER + 0.002 + i * 0.004), 0.004, z + 0.0008))
        feather(bm, 0.034, 0.013, root, ang(86 - i * 3), camber=0.2)
    # Coverts: a smooth cambered sheet over the feather roots from the shoulder to beyond the
    # wrist, its trailing edge softly scalloped (feather tips); a rounded leading edge.
    nu, nv = 24, 6
    grid = []
    for iu in range(nu + 1):
        u = iu / nu
        x = SHOULDER - 0.002 + u * 0.062
        lead = 0.016 - u * 0.004 - max(0.0, u - 0.7) * 0.02
        depth = (0.022 + 0.004 * math.sin(math.pi * u)) * (1 - max(0.0, u - 0.75) * 2.4)
        row = []
        for iv in range(nv + 1):
            v = iv / nv
            scallop = 0.0025 * abs(math.sin(u * math.pi * 9)) * v ** 3
            y = lead - v * depth + scallop
            zz = z + 0.0032 * math.sin(math.pi * (0.15 + v * 0.85)) + 0.0012 - v * 0.0008
            row.append(bm.verts.new((s * x, y, zz)))
        grid.append(row)
    for iu in range(nu):
        for iv in range(nv):
            a, b, c, d = grid[iu][iv], grid[iu][iv + 1], grid[iu + 1][iv + 1], grid[iu + 1][iv]
            bm.faces.new((a, b, c, d) if s > 0 else (d, c, b, a))
    ob = obj_from_bm(bm, f"wing{side}", 3)
    sol = ob.modifiers.new("sol", "SOLIDIFY")
    sol.thickness = 0.0009
    return ob


def tail():
    bm = bmesh.new()
    root = Vector((0, -0.052, 0.003))
    for i in range(12):
        k = (i - 5.5) / 5.5  # -1 .. 1
        outer = abs(k)
        # Deep fork: the outer pair are long streamers; the middle ones short.
        length = 0.04 + 0.012 * outer + (0.06 if outer > 0.95 else 0.0)
        width = 0.0085 if outer < 0.95 else 0.0055
        a = math.radians(k * 24)
        d = Vector((math.sin(a), -math.cos(a), 0.03))
        feather(bm, length, width, root + Vector((k * 0.004, 0, 0.0002 * (6 - abs(i - 5.5)))), d, camber=0.15, rows=8)
    ob = obj_from_bm(bm, "tail", 4)
    sol = ob.modifiers.new("sol", "SOLIDIFY")
    sol.thickness = 0.0008
    return ob


# ---------------------------------------------------------------------------------------
# Paint (Blender space: x right, y forward, z up)


BLUE = np.array([0.03, 0.06, 0.2])
RED = np.array([0.5, 0.11, 0.05])
CREAM = np.array([0.88, 0.8, 0.66])


def paint(pos, nrm, part):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    n = len(pos)
    col = np.zeros((n, 3), np.float32)
    gloss = np.zeros(n, np.float32)
    fine = np.array([noise.noise(Vector((px * 300, py * 300, pz * 300))) for px, py, pz in pos])
    up = np.clip((nrm[:, 2] + 0.05) * 2.5, 0, 1)
    # Body: blue above, cream below; red forehead and throat, a dark breast band.
    body = CREAM * (1 - up[:, None]) + BLUE * up[:, None]
    head = y > 0.04
    throat = head & (z < 0.004) & (nrm[:, 2] < 0.2)
    forehead = (y > 0.062) & (z > 0.004)
    band = (y > 0.026) & (y < 0.04) & (nrm[:, 2] < 0.3)
    body = np.where(throat[:, None], RED, body)
    body = np.where(forehead[:, None], RED * 0.9, body)
    body = np.where((band & ~throat)[:, None], BLUE * 1.2, body)
    body = body * (0.92 + 0.12 * fine[:, None])
    col[part == 0] = body[part == 0]
    gloss[part == 0] = up[part == 0]
    col[part == 1] = (0.02, 0.02, 0.02)
    col[part == 2] = (0.01, 0.01, 0.012)
    # Wings: blue-black above with a steel gloss; the underwing coverts pale, the flight
    # feathers greyer below.
    under = nrm[:, 2] < 0
    wingc = np.where(under[:, None], np.where((np.abs(x) < 0.06)[:, None], CREAM * 0.85, np.array([0.18, 0.18, 0.2])), BLUE * (0.9 + 0.2 * fine[:, None]))
    col[part == 3] = wingc[part == 3]
    gloss[part == 3] = (~under)[part == 3]
    # Tail: blue-black, a white spot on each feather except the middle pair.
    spot = (np.abs(x) > 0.006) & (y < -0.075) & (y > -0.092)
    tailc = np.where(spot[:, None], np.array([0.85, 0.84, 0.8]), BLUE * 0.85)
    col[part == 4] = tailc[part == 4]
    gloss[part == 4] = 1
    return np.clip(col, 0, 1), gloss


def collect(obs):
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
    return np.concatenate(pos_l), np.concatenate(nrm_l), np.concatenate(part_l), np.concatenate(idx_l)


def export(pos, nrm, part, idx, col, gloss):
    ao = np.clip(0.6 + 0.4 * (nrm[:, 2] * 0.5 + 0.5), 0, 1)
    span = np.where(part == 3, np.sign(pos[:, 0]) * np.maximum(np.abs(pos[:, 0]) - SHOULDER, 0), 0)
    gp = np.stack([pos[:, 0], pos[:, 2], pos[:, 1]], 1)
    gn = np.stack([nrm[:, 0], nrm[:, 2], nrm[:, 1]], 1)
    idx = idx[:, [0, 2, 1]]
    v = np.zeros(len(gp), dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
    v["p"][:, :3] = gp
    v["p"][:, 3] = part
    v["n"][:, :3] = np.clip(np.round(gn * 127), -127, 127)
    v["t"][:, 0] = span
    v["t"][:, 1] = gloss
    v["e"] = np.clip(np.round(np.concatenate([col, ao[:, None]], 1) * 255), 0, 255)
    with open(os.path.join(OUT, "swallow.bin"), "wb") as f:
        f.write(v.tobytes())
        f.write(idx.reshape(-1).astype("<u4").tobytes())
    json.dump({"vertexBytes": len(v) * 20, "indexCount": int(idx.size), "shoulder": SHOULDER, "elbow": ELBOW,
               "credit": "Modeled in Blender (tools/blender/model_swallow.py)"},
              open(os.path.join(OUT, "swallow.json"), "w"), indent=2)
    print(f"[swallow] {idx.shape[0]} tris", flush=True)


def preview(pos, nrm, idx, col, gloss):
    """Renders the painted mesh (vertex colours) from two angles, to judge it."""
    me = bpy.data.meshes.new("prev")
    me.from_pydata([tuple(p) for p in pos], [], [tuple(t) for t in idx])
    attr = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    for i, c in enumerate(col):
        attr.data[i].color = (c[0] ** 2.2, c[1] ** 2.2, c[2] ** 2.2, 1)
    me.shade_smooth() if hasattr(me, "shade_smooth") else None
    ob = bpy.data.objects.new("prev", me)
    bpy.context.scene.collection.objects.link(ob)
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    ca = nt.nodes.new("ShaderNodeVertexColor")
    ca.layer_name = "Col"
    nt.links.new(ca.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.45
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
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.2
    scene.world = world
    cam = bpy.data.cameras.new("cam")
    cam.lens = 70
    co = bpy.data.objects.new("cam", cam)
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 600
    for name, loc, tgt in (("top", (0.25, -0.35, 0.45), (0, -0.01, 0)), ("below", (-0.3, 0.3, -0.25), (0, 0, 0)), ("side", (0.45, 0.05, 0.08), (0, -0.01, 0))):
        co.location = loc
        co.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.abspath(PREVIEW.replace(".png", f"-{name}.png"))
        bpy.ops.render.render(write_still=True)


def main():
    reset()
    skin = body()
    obs = [skin, bill(skin), eyes(skin), wing(1), wing(-1), tail()]
    pos, nrm, part, idx = collect(obs)
    col, gloss = paint(pos, nrm, part)
    export(pos, nrm, part, idx, col, gloss)
    if PREVIEW:
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob)
        preview(pos, nrm, idx, col, gloss)


main()
