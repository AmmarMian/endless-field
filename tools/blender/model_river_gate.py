"""
Models a river gate for the kingfisher's course in Blender and exports it:

  blender -b --factory-startup --python tools/blender/model_river_gate.py -- <out dir> [--preview <png>]

A hoop of twisted straw rope (shimenawa) held between two bamboo poles standing in the water,
hung with zigzag paper streamers (shide) that flutter. The hoop's centre is at (0, RING_Y, 0);
the wind passes through along +Z.

Vertex layout (shared with the other models): p.xyz position, p.w part (0 bamboo, 1 rope,
2 paper streamer, 3 rope binding); n normal; t.x = 0 / streamer index + 1, t.y = distance
below the streamer's top (the shader flutters by it); e.rgb albedo, e.w ambient occlusion.
Output: gate.bin + gate.json. Y-up.
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None
os.makedirs(OUT, exist_ok=True)

RING_Y = 2.3
RADIUS = 1.5
POLE_X = RADIUS + 0.12


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def obj(bm, name, part, color, extra=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob["part"] = part
    ob["color"] = color
    ob["extra"] = extra or [0.0, 0.0]
    return ob


def tube(bm, a, b, r, segs=8):
    a, b = Vector(a), Vector(b)
    d = b - a
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r, radius2=r * 0.9, depth=d.length)
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    for v in res["verts"]:
        v.co = rot @ v.co + (a + b) / 2
    return res["verts"]


def poles():
    bm = bmesh.new()
    for s in (-1, 1):
        tube(bm, (s * POLE_X, 0, -1.2), (s * POLE_X, 0, RING_Y + 0.35), 0.045, 10)
        for i in range(1, 8):
            z = -1.2 + i * 0.5
            tube(bm, (s * POLE_X, 0, z - 0.012), (s * POLE_X, 0, z + 0.012), 0.052, 10)
    return obj(bm, "poles", 0, (0.6, 0.55, 0.33))


def rope():
    """Two strands twisted around each other, swept round the hoop (and thicker at the top)."""
    bm = bmesh.new()
    n, m = 160, 8
    for strand in range(2):
        rings = []
        for i in range(n):
            u = i / n * 2 * math.pi
            centre = Vector((RADIUS * math.cos(u), 0, RING_Y + RADIUS * math.sin(u)))
            radial = Vector((math.cos(u), 0, math.sin(u)))
            tangent = Vector((-math.sin(u), 0, math.cos(u)))
            side = Vector((0, 1, 0))
            twist = u * 9 + strand * math.pi
            thick = 0.05 + 0.02 * max(0.0, math.sin(u))
            off = (radial * math.cos(twist) + side * math.sin(twist)) * thick * 0.55
            ring = []
            for k in range(m):
                a = k / m * 2 * math.pi
                ring.append(bm.verts.new(centre + off + (radial * math.cos(a) + side * math.sin(a)) * thick * 0.6))
            rings.append(ring)
        for i in range(n):
            r0, r1 = rings[i], rings[(i + 1) % n]
            for k in range(m):
                bm.faces.new((r0[k], r0[(k + 1) % m], r1[(k + 1) % m], r1[k]))
    return obj(bm, "rope", 1, (0.78, 0.68, 0.42))


def bindings():
    bm = bmesh.new()
    for s in (-1, 1):
        for i in range(5):
            z = RING_Y - 0.08 + i * 0.04
            tube(bm, (s * (POLE_X - 0.13), 0, z), (s * (POLE_X + 0.06), 0, z), 0.012, 6)
    return obj(bm, "bind", 3, (0.85, 0.15, 0.12))


def shide():
    """Zigzag paper streamers hanging from the top of the hoop."""
    obs = []
    for idx, u in enumerate((0.3, 0.42, 0.58, 0.7)):
        a = u * math.pi
        top = Vector((RADIUS * math.cos(a), 0.0, RING_Y + RADIUS * math.sin(a) - 0.06))
        bm = bmesh.new()
        w = 0.11
        steps = 4
        seg = 0.17
        verts = []
        x = 0.0
        for i in range(steps * 2 + 1):
            # A lightning-bolt zigzag: shift sideways every half step.
            z = -i * seg / 2
            x = (w * 0.5) * (1 if (i // 2) % 2 == 0 else -1) * (1 if i % 2 else 0.4)
            verts.append((bm.verts.new(top + Vector((x - w / 2, -0.004 * i, z))), bm.verts.new(top + Vector((x + w / 2, -0.004 * i, z)))))
        for i in range(len(verts) - 1):
            bm.faces.new((verts[i][0], verts[i][1], verts[i + 1][1], verts[i + 1][0]))
        obs.append(obj(bm, f"shide{idx}", 2, (0.97, 0.96, 0.92), [idx + 1, top.z]))
    return obs


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
        # Blender (x, y, z up) -> game (x, y up, z): the hoop's opening faces game +Z.
        v["p"][:, 0] = co[:, 0]
        v["p"][:, 1] = co[:, 2]
        v["p"][:, 2] = -co[:, 1]
        v["p"][:, 3] = ob["part"]
        v["n"][:, 0] = np.round(nr[:, 0] * 127)
        v["n"][:, 1] = np.round(nr[:, 2] * 127)
        v["n"][:, 2] = np.round(-nr[:, 1] * 127)
        ex = ob["extra"]
        v["t"][:, 0] = ex[0]
        v["t"][:, 1] = np.where(ex[0] > 0, ex[1] - co[:, 2], 0)
        col = np.array(ob["color"], np.float32)
        # Straw rope: alternate light and dark along the twist for texture.
        if ob["part"] == 1:
            ang = np.arctan2(co[:, 2] - RING_Y, co[:, 0])
            stripes = 0.85 + 0.15 * np.sin(ang * 18 * 9 / 2)
            cols = np.outer(stripes, col)
        else:
            cols = np.tile(col, (n, 1))
        v["e"][:, :3] = np.clip(np.round(cols * 255), 0, 255)
        v["e"][:, 3] = np.round(np.clip(0.7 + 0.3 * (nr[:, 2] * 0.5 + 0.5), 0, 1) * 255)
        vs.append(v)
        ids.append(tris.reshape(-1, 3) + base)
        base += n
        ev.to_mesh_clear()
    return np.concatenate(vs), np.concatenate(ids).reshape(-1)


def main():
    reset()
    obs = [poles(), rope(), bindings(), *shide()]
    if PREVIEW:
        render_preview(obs)
    v, idx = export(obs)
    with open(os.path.join(OUT, "gate.bin"), "wb") as f:
        f.write(v.tobytes())
        f.write(idx.astype("<u4").tobytes())
    json.dump({"vertexBytes": len(v) * 20, "indexCount": int(idx.size), "ringY": RING_Y, "radius": RADIUS,
               "credit": "Modeled in Blender (tools/blender/model_river_gate.py)"},
              open(os.path.join(OUT, "gate.json"), "w"), indent=2)
    print(f"[gate] {idx.size // 3} tris", flush=True)


def render_preview(obs):
    scene = bpy.context.scene
    for ob in obs:
        mat = bpy.data.materials.new(f"m{ob.name}")
        mat.use_nodes = True
        c = ob["color"]
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (c[0], c[1], c[2], 1)
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
    cam.lens = 40
    co = bpy.data.objects.new("cam", cam)
    co.location = (2.5, -6.0, 2.6)
    co.rotation_euler = (Vector((0, 0, 1.8)) - co.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 700
    scene.render.resolution_y = 600
    scene.render.filepath = os.path.abspath(PREVIEW)
    bpy.ops.render.render(write_still=True)


main()
