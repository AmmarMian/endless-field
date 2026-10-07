"""
Models the wind's small secrets in Blender and exports them for the meadow:

  blender -b --factory-startup --python tools/blender/model_secrets.py -- <out dir> [--preview <png>]

  pinwheel  a Japanese kazaguruma: four folded paper sails around a pin, on a bamboo stick
  furin     a frame of bamboo (two posts, a lashed crossbar) hung with three glass wind bells,
            each with a clapper and a long paper strip (tanzaku) that catches the wind

Vertex layout (shared with the other models): p.xyz position, p.w part (0 bamboo/wood,
1 paper sail, 2 pin, 3 glass bell, 4 paper strip, 5 clapper, 6 cord); n normal;
t = pivot (x, y) of the moving part in model space (sails spin about (t.x, t.y) along +Z;
bells, clappers and strips swing about it); e.rgb albedo, e.w ambient occlusion.
Y-up, the pinwheel faces +Z. Output: secrets.bin + secrets.json.
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None
os.makedirs(OUT, exist_ok=True)

HUB_Y = 0.62  # pinwheel hub height (Blender z)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def obj(bm, name, part, color, pivot=(0.0, 0.0), solidify=0.0):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob["part"] = part
    ob["color"] = color
    ob["pivot"] = pivot
    if solidify:
        m = ob.modifiers.new("sol", "SOLIDIFY")
        m.thickness = solidify
    return ob


def tube(bm, a, b, r, segs=8):
    """A cylinder from point a to b."""
    a, b = Vector(a), Vector(b)
    d = b - a
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r, radius2=r * 0.92, depth=d.length)
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    for v in res["verts"]:
        v.co = rot @ v.co + (a + b) / 2
    return res["verts"]


def bamboo(bm, a, b, r):
    """Bamboo: a cane with slightly swollen nodes every ~25 cm."""
    tube(bm, a, b, r)
    a, b = Vector(a), Vector(b)
    n = max(1, int((b - a).length / 0.25))
    for i in range(1, n + 1):
        p = a + (b - a) * (i / (n + 1))
        d = (b - a).normalized() * 0.008
        tube(bm, p - d, p + d, r * 1.18, 8)


# ---------------------------------------------------------------------------------------
# Pinwheel


PAPERS = [(0.86, 0.16, 0.14), (0.95, 0.75, 0.2), (0.2, 0.45, 0.78), (0.96, 0.94, 0.9), (0.85, 0.42, 0.62), (0.3, 0.62, 0.42)]


def pinwheel(seed):
    rng = np.random.default_rng(seed)
    obs = []
    bm = bmesh.new()
    bamboo(bm, (0, 0, 0), (0, 0, HUB_Y - 0.01), 0.006)
    obs.append(obj(bm, "stick", 0, (0.62, 0.55, 0.32)))
    # Four sails: each is half of a square's quarter folded to the pin, a curved triangle that
    # cups the wind. Two colours alternate (paper printed differently on each face).
    c0, c1 = PAPERS[rng.integers(len(PAPERS))], PAPERS[rng.integers(len(PAPERS))]
    R = 0.11
    for k in range(4):
        bm = bmesh.new()
        n = 8
        rows = []
        for i in range(n + 1):
            u = i / n
            row = []
            for j in range(n + 1 - i):
                w = j / n
                # Triangle (pin, outer corner, side corner) bent so the free corner curls back
                # to the pin: the classic sail.
                x = u * R + w * R * 0.05
                y = w * R
                z = 0.035 * (w ** 1.5) * (1 - u) + 0.012 * u * w
                row.append(bm.verts.new((x, y, z)))
            rows.append(row)
        for i in range(n):
            for j in range(n - i):
                a, b, c = rows[i][j], rows[i + 1][j], rows[i][j + 1]
                bm.faces.new((a, b, c))
                if j + 1 < len(rows[i + 1]):
                    bm.faces.new((b, rows[i + 1][j + 1], c))
        m = Matrix.Translation((0, -0.008, HUB_Y)) @ Matrix.Rotation(-math.pi / 2, 4, "X") @ Matrix.Rotation(k * math.pi / 2, 4, "Z")
        for v in bm.verts:
            v.co = m @ v.co
        obs.append(obj(bm, f"sail{k}", 1, c0 if k % 2 == 0 else c1, (0.0, HUB_Y), 0.0008))
    bm = bmesh.new()
    tube(bm, (0, 0.008, HUB_Y), (0, -0.026, HUB_Y), 0.0045, 8)
    r = bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.007)
    bmesh.ops.translate(bm, verts=r["verts"], vec=(0, -0.026, HUB_Y))
    obs.append(obj(bm, "pin", 2, (0.9, 0.85, 0.6), (0.0, HUB_Y)))
    return obs


# ---------------------------------------------------------------------------------------
# Wind bells


def furin():
    obs = []
    bm = bmesh.new()
    H, W = 1.75, 0.9
    bamboo(bm, (-W / 2, 0, 0), (-W / 2, 0, H), 0.022)
    bamboo(bm, (W / 2, 0, 0), (W / 2, 0, H), 0.022)
    bamboo(bm, (-W / 2 - 0.12, 0, H - 0.08), (W / 2 + 0.12, 0, H - 0.08), 0.018)
    # Rope lashings at the joints.
    for s in (-1, 1):
        for i in range(4):
            tube(bm, (s * W / 2 - 0.03, 0, H - 0.1 + i * 0.012), (s * W / 2 + 0.03, 0, H - 0.1 + i * 0.012), 0.004, 6)
    obs.append(obj(bm, "frame", 0, (0.58, 0.52, 0.32)))
    bm = bmesh.new()
    for s in (-1, 1):
        for i in range(4):
            tube(bm, (s * W / 2 - 0.03, -0.024, H - 0.1 + i * 0.012), (s * W / 2 + 0.03, -0.024, H - 0.1 + i * 0.012), 0.004, 6)
    obs.append(obj(bm, "lash", 6, (0.82, 0.74, 0.55)))

    for b, x in enumerate((-0.25, 0.02, 0.27)):
        top = H - 0.1 - 0.02
        drop = 0.14 + 0.05 * (b % 2)
        bell_y = top - drop
        pivot = (x, top)
        # Cord.
        bm = bmesh.new()
        tube(bm, (x, 0, top), (x, 0, bell_y + 0.06), 0.0015, 5)
        obs.append(obj(bm, f"cord{b}", 6, (0.85, 0.15, 0.12), pivot))
        # Glass bell: a lathed dome, open at the bottom.
        bm = bmesh.new()
        prof = [(0.004, 0.065), (0.02, 0.062), (0.034, 0.05), (0.042, 0.03), (0.045, 0.01), (0.046, 0.0)]
        segs = 18
        rings = []
        for r, z in prof:
            rings.append([bm.verts.new((x + r * math.cos(2 * math.pi * k / segs), r * math.sin(2 * math.pi * k / segs), bell_y + z)) for k in range(segs)])
        for i in range(len(rings) - 1):
            for k in range(segs):
                bm.faces.new((rings[i][k], rings[i + 1][k], rings[i + 1][(k + 1) % segs], rings[i][(k + 1) % segs]))
        bm.faces.new(rings[0][::-1])
        tint = [(0.55, 0.78, 0.9), (0.9, 0.62, 0.55), (0.62, 0.85, 0.7)][b]
        obs.append(obj(bm, f"bell{b}", 3, tint, pivot, 0.002))
        # Clapper and the paper strip below it.
        bm = bmesh.new()
        tube(bm, (x, 0, bell_y + 0.05), (x, 0, bell_y - 0.02), 0.0012, 5)
        r = bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=0.006)
        bmesh.ops.translate(bm, verts=r["verts"], vec=(x, 0, bell_y + 0.005))
        obs.append(obj(bm, f"clapper{b}", 5, (0.75, 0.72, 0.68), pivot))
        bm = bmesh.new()
        n = 8
        rows = []
        for i in range(n + 1):
            z = bell_y - 0.02 - i / n * 0.2
            rows.append((bm.verts.new((x - 0.022, 0, z)), bm.verts.new((x + 0.022, 0, z))))
        for i in range(n):
            bm.faces.new((rows[i][0], rows[i][1], rows[i + 1][1], rows[i + 1][0]))
        paper = [(0.95, 0.92, 0.82), (0.88, 0.3, 0.3), (0.35, 0.55, 0.85)][b]
        obs.append(obj(bm, f"strip{b}", 4, paper, pivot, 0.0006))
    return obs


# ---------------------------------------------------------------------------------------


def export(obs):
    dg = bpy.context.evaluated_depsgraph_get()
    vs, ids = [], []
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
        co = co.reshape(-1, 3)
        nr = nr.reshape(-1, 3)
        v = np.zeros(n, dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
        # Blender (x, y, z up) -> game (x, z up -> y, y -> -z): the pinwheel faces Blender -Y,
        # which becomes game +Z.
        v["p"][:, 0] = co[:, 0]
        v["p"][:, 1] = co[:, 2]
        v["p"][:, 2] = -co[:, 1]
        v["p"][:, 3] = ob["part"]
        v["n"][:, 0] = np.round(nr[:, 0] * 127)
        v["n"][:, 1] = np.round(nr[:, 2] * 127)
        v["n"][:, 2] = np.round(-nr[:, 1] * 127)
        v["t"][:, 0] = ob["pivot"][0]
        v["t"][:, 1] = ob["pivot"][1]
        ao = np.clip(0.65 + 0.35 * (nr[:, 2] * 0.5 + 0.5), 0, 1)
        col = np.array(ob["color"], np.float32)
        v["e"][:, :3] = np.clip(np.round(col * 255), 0, 255)
        v["e"][:, 3] = np.round(ao * 255)
        vs.append(v)
        ids.append(tris.reshape(-1, 3) + base)
        base += n
        ev.to_mesh_clear()
    return np.concatenate(vs), np.concatenate(ids).reshape(-1)


def main():
    vb, ib, variants = [], [], []
    vbase = ioff = 0
    builds = [("pinwheel-a", lambda: pinwheel(3)), ("pinwheel-b", lambda: pinwheel(11)), ("pinwheel-c", lambda: pinwheel(29)), ("furin", furin)]
    for name, fn in builds:
        reset()
        obs = fn()
        if PREVIEW and name in ("pinwheel-a", "furin"):
            render_preview(obs, name)
        v, idx = export(obs)
        vb.append(v.tobytes())
        ib.append((idx + vbase).astype("<u4").tobytes())
        variants.append({"name": name, "firstIndex": ioff, "indexCount": int(idx.size)})
        print(f"[secrets] {name}: {idx.size // 3} tris", flush=True)
        vbase += len(v)
        ioff += idx.size
    with open(os.path.join(OUT, "secrets.bin"), "wb") as f:
        f.write(b"".join(vb))
        f.write(b"".join(ib))
    json.dump({"vertexBytes": vbase * 20, "indexCount": ioff, "variants": variants, "hubY": HUB_Y,
               "credit": "Modeled in Blender (tools/blender/model_secrets.py)"},
              open(os.path.join(OUT, "secrets.json"), "w"), indent=2)


def render_preview(obs, name):
    scene = bpy.context.scene
    for ob in obs:
        mat = bpy.data.materials.new(f"m{ob.name}")
        mat.use_nodes = True
        b = mat.node_tree.nodes["Principled BSDF"]
        c = ob["color"]
        b.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1)
        if ob["part"] == 3:
            b.inputs["Transmission Weight"].default_value = 0.7
            b.inputs["Roughness"].default_value = 0.1
        ob.data.materials.append(mat)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(so)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.66, 0.75, 1)
    scene.world = world
    cam = bpy.data.cameras.new("cam")
    cam.lens = 50
    co = bpy.data.objects.new("cam", cam)
    if name == "furin":
        co.location, target = (1.1, -2.4, 1.4), Vector((0, 0, 1.3))
    else:
        co.location, target = (0.35, -0.55, 0.75), Vector((0, 0, HUB_Y))
    co.rotation_euler = (target - co.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 600
    scene.render.resolution_y = 600
    scene.render.filepath = os.path.abspath(PREVIEW.replace(".png", f"-{name}.png"))
    bpy.ops.render.render(write_still=True)


main()
