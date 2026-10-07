"""
Models a river fish (a rainbow trout) in Blender and exports it for the river:

  blender -b --factory-startup --python tools/blender/model_fish.py -- <out dir> [--preview <png>]

The body is lofted from elliptical cross-sections along a profile (blunt head, deepest just
behind the gills, a slim tail stalk), then smoothed with a subdivision pass. Fins are thin
shaped plates: forked tail, dorsal, small adipose, pectorals, pelvics and anal fin. Eyes are
small spheres. Painted per vertex: olive-green back with dark spots, a rosy band along the
flank, silver-white belly.

Vertex layout (shared with the other models): p.xyz position, p.w part (0 body, 1 fin,
2 eye); n normal; t.x = position along the body (0 snout .. 1 tail tip: the shader swims
by it), t.y unused; e.rgb albedo, e.w ambient occlusion. Length 1 m (scaled per fish in the
game), facing +Z, Y-up. Output: fish.bin + fish.json.
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Vector, noise

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None
os.makedirs(OUT, exist_ok=True)

L = 1.0  # body length, snout (y = +L/2) to tail tip (y = -L/2)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def obj(bm, name, part):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob["part"] = part
    return ob


def body():
    """Lofted body: (u along the length, half height, half width, centre height)."""
    prof = [
        (0.0, 0.005, 0.004, 0.0),
        (0.03, 0.045, 0.035, 0.002),
        (0.08, 0.075, 0.055, 0.004),
        (0.16, 0.1, 0.07, 0.006),
        (0.28, 0.115, 0.075, 0.008),
        (0.42, 0.11, 0.07, 0.008),
        (0.56, 0.095, 0.058, 0.006),
        (0.7, 0.07, 0.042, 0.004),
        (0.82, 0.048, 0.028, 0.002),
        (0.9, 0.04, 0.02, 0.0),
    ]
    bm = bmesh.new()
    seg = 20
    rings = []
    for u, hh, hw, zc in prof:
        y = L / 2 - u * L
        ring = []
        for k in range(seg):
            a = k / seg * 2 * math.pi
            # Slightly flat-bottomed, rounder back.
            s = math.sin(a)
            hz = hh * (1.0 if s > 0 else 0.92)
            ring.append(bm.verts.new((hw * math.cos(a), y, zc + hz * s)))
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(seg):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % seg], rings[i + 1][(k + 1) % seg], rings[i + 1][k]))
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])
    ob = obj(bm, "body", 0)
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = 1
    return ob


def fin(name, outline, thickness=0.004):
    """A flat fin from an outline polygon in the (y, z) plane at x = 0, or given 3D points."""
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in outline]
    face = bm.faces.new(vs)
    bmesh.ops.triangulate(bm, faces=[face])
    ob = obj(bm, name, 1)
    sol = ob.modifiers.new("sol", "SOLIDIFY")
    sol.thickness = thickness
    sol.offset = 0
    return ob


def fins():
    obs = []
    tail_y = L / 2 - 0.88 * L
    # Forked tail, slightly concave.
    t = []
    for i in range(13):
        k = i / 12
        a = (k - 0.5) * 2  # -1 .. 1, lower lobe to upper lobe
        fork = 0.14 - 0.05 * (1 - abs(a)) ** 2
        t.append((0, tail_y - fork, a * 0.12))
    obs.append(fin("tail", [(0, tail_y + 0.02, 0.03), *t[::-1], (0, tail_y + 0.02, -0.03)]))
    # Dorsal: a rounded sail on the back.
    d0 = L / 2 - 0.36 * L
    obs.append(fin("dorsal", [(0, d0 + 0.05, 0.105), (0, d0 + 0.02, 0.19), (0, d0 - 0.06, 0.18), (0, d0 - 0.11, 0.1)]))
    # Adipose: a small nub near the tail.
    a0 = L / 2 - 0.72 * L
    obs.append(fin("adipose", [(0, a0 + 0.02, 0.06), (0, a0, 0.09), (0, a0 - 0.03, 0.06)]))
    # Anal fin underneath.
    n0 = L / 2 - 0.66 * L
    obs.append(fin("anal", [(0, n0 + 0.04, -0.06), (0, n0 + 0.01, -0.12), (0, n0 - 0.06, -0.1), (0, n0 - 0.07, -0.05)]))
    # Paired fins: pectorals behind the gills, pelvics mid-belly, angled out and back.
    for s in (-1, 1):
        p0 = L / 2 - 0.2 * L
        obs.append(fin(f"pect{s}", [(s * 0.05, p0, -0.06), (s * 0.13, p0 - 0.06, -0.09), (s * 0.1, p0 - 0.11, -0.085), (s * 0.05, p0 - 0.05, -0.07)]))
        v0 = L / 2 - 0.5 * L
        obs.append(fin(f"pelv{s}", [(s * 0.04, v0, -0.08), (s * 0.09, v0 - 0.06, -0.11), (s * 0.06, v0 - 0.09, -0.1), (s * 0.035, v0 - 0.04, -0.085)]))
    return obs


def eyes():
    bm = bmesh.new()
    for s in (-1, 1):
        r = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=8, radius=0.016)
        bmesh.ops.translate(bm, verts=r["verts"], vec=(s * 0.04, L / 2 - 0.075, 0.022))
    return obj(bm, "eyes", 2)


def paint(pos, nrm, part):
    """Blender space: x side, y forward (+ = head), z up."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    u = (L / 2 - y) / L
    col = np.zeros((len(pos), 3), np.float32)
    spots = np.array([noise.noise(Vector((px * 40, py * 40, pz * 40))) for px, py, pz in pos])
    back = np.array([0.22, 0.28, 0.14])
    belly = np.array([0.88, 0.87, 0.82])
    band = np.array([0.82, 0.38, 0.4])
    # Vertical position relative to the body's local half height.
    h = np.clip(z / 0.1, -1, 1)
    c = belly + (back - belly) * np.clip((h + 0.15) * 1.8, 0, 1)[:, None]
    # Rosy lateral band along the flank (the "rainbow").
    bandw = np.exp(-((h - 0.05) ** 2) / 0.04)[:, None] * (np.abs(x) > 0.02)[:, None]
    c = c + (band - c) * bandw * 0.7
    # Dark spots above the band and on the tail and dorsal.
    dark = (spots > 0.35) & (h > -0.1)
    c = np.where(dark[:, None], c * 0.3, c)
    col[:] = c
    fins = part == 1
    fincol = np.array([0.42, 0.4, 0.3]) * (0.8 + 0.3 * (spots[:, None] < 0.3))
    col[fins] = fincol[fins]
    col[part == 2] = (0.02, 0.02, 0.02)
    return np.clip(col, 0, 1), np.clip(u, 0, 1)


def export(obs):
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
    col, along = paint(pos, nrm, part)
    ao = np.clip(0.65 + 0.35 * (nrm[:, 2] * 0.5 + 0.5), 0, 1)
    gp = np.stack([pos[:, 0], pos[:, 2], pos[:, 1]], 1)
    gn = np.stack([nrm[:, 0], nrm[:, 2], nrm[:, 1]], 1)
    idx = idx[:, [0, 2, 1]]
    v = np.zeros(len(gp), dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
    v["p"][:, :3] = gp
    v["p"][:, 3] = part
    v["n"][:, :3] = np.clip(np.round(gn * 127), -127, 127)
    v["t"][:, 0] = along
    v["e"] = np.clip(np.round(np.concatenate([col, ao[:, None]], 1) * 255), 0, 255)
    return v, idx.reshape(-1)


def main():
    reset()
    obs = [body(), *fins(), eyes()]
    if PREVIEW:
        render_preview(obs)
    v, idx = export(obs)
    with open(os.path.join(OUT, "fish.bin"), "wb") as f:
        f.write(v.tobytes())
        f.write(idx.astype("<u4").tobytes())
    json.dump({"vertexBytes": len(v) * 20, "indexCount": int(idx.size), "credit": "Modeled in Blender (tools/blender/model_fish.py)"},
              open(os.path.join(OUT, "fish.json"), "w"), indent=2)
    print(f"[fish] {idx.size // 3} tris", flush=True)


def render_preview(obs):
    scene = bpy.context.scene
    for ob in obs:
        mat = bpy.data.materials.new(f"m{ob.name}")
        mat.use_nodes = True
        c = {0: (0.45, 0.5, 0.35, 1), 1: (0.4, 0.38, 0.3, 1), 2: (0, 0, 0, 1)}[ob["part"]]
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = c
        ob.data.materials.append(mat)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(45), 0, math.radians(30))
    scene.collection.objects.link(so)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.66, 0.75, 1)
    scene.world = world
    cam = bpy.data.cameras.new("cam")
    cam.lens = 50
    co = bpy.data.objects.new("cam", cam)
    co.location = (1.6, 0.4, 0.5)
    co.rotation_euler = (Vector((0, 0, 0)) - co.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 450
    scene.render.filepath = os.path.abspath(PREVIEW)
    bpy.ops.render.render(write_still=True)


main()
