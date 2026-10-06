"""
Models a Kasuga-style stone lantern (ishi-doro) in Blender and exports it for the lantern path.

  blender -b --factory-startup --python tools/blender/model_lantern.py -- <out dir> [--preview <png>]

Parts, bottom to top: hexagonal plinth (kiso) with a lotus cushion, banded pole (sao),
platform (chudai), fire box (hibukuro) with open windows between corner posts and a paper
lantern inside, hexagonal roof (kasa) with concave pitch and upturned corner curls (warabite),
and an onion jewel finial (hoju). Stone parts are bevelled with Blender's Bevel modifier.

Vertex attributes (same layout as the plants): p.w part (0 stone, 1 paper), e.x height in
[0, 1], e.y moss weight (upward faces, crevices), e.z ambient occlusion, e.w unused.
Output: lantern.bin + lantern.json (two variants, two LODs each). Y-up.
"""
import json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None
os.makedirs(OUT, exist_ok=True)
HEX = 6


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(bm, name, part, bevel=0.0, segs=2, smooth_angle=35):
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
        m.angle_limit = math.radians(smooth_angle)
    return ob


def prism(name, r0, r1, z0, z1, part=0, bevel=0.012, segs=HEX, rot=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r0, radius2=r1, depth=z1 - z0,
                          matrix=Matrix.Translation((0, 0, (z0 + z1) / 2)) @ Matrix.Rotation(rot, 4, "Z"))
    return link(bm, name, part, bevel)


def lathe(name, profile, segs, part=0, bevel=0.0):
    """Surface of revolution from (r, z) pairs, bottom to top."""
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        ring = []
        for k in range(segs):
            a = k / segs * 2 * math.pi
            ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(segs):
            a, b = rings[i][k], rings[i][(k + 1) % segs]
            c, d = rings[i + 1][(k + 1) % segs], rings[i + 1][k]
            bm.faces.new((a, b, c, d))
    if profile[0][0] > 1e-4:
        bm.faces.new(list(reversed(rings[0])))
    if profile[-1][0] > 1e-4:
        bm.faces.new(rings[-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    return link(bm, name, part, bevel)


def hex_radius(theta, r):
    """Distance from the center to a hexagon's edge (corners at multiples of 60 degrees)."""
    seg = math.pi / 3
    t = (theta % seg) - seg / 2
    return r * math.cos(seg / 2) / math.cos(t)


def roof(name, r_eave, z_eave, rise, upturn, thick, n_ang, n_rad):
    """Hexagonal roof: concave pitch to a peak, corners curling upward at the eave."""
    bm = bmesh.new()
    top, bot = [], []
    for i in range(n_ang):
        th = i / n_ang * 2 * math.pi
        rh = hex_radius(th, r_eave)
        seg = math.pi / 3
        corner = 1 - abs(((th % seg) - seg / 2) / (seg / 2))  # 1 at corners
        corner = corner ** 4
        rt, rb = [], []
        for j in range(n_rad + 1):
            rho = j / n_rad  # 0 peak, 1 eave
            r = rh * rho * (1 + 0.1 * corner * rho ** 3)
            z = z_eave + rise * (1 - rho) ** 1.7 + upturn * corner * rho ** 4
            rt.append(bm.verts.new((r * math.cos(th), r * math.sin(th), z)))
            rb.append(bm.verts.new((r * math.cos(th), r * math.sin(th), z - thick * (0.6 + 0.4 * rho))))
        top.append(rt)
        bot.append(rb)
    for i in range(n_ang):
        a, b = top[i], top[(i + 1) % n_ang]
        c, d = bot[i], bot[(i + 1) % n_ang]
        for j in range(n_rad):
            bm.faces.new((a[j], b[j], b[j + 1], a[j + 1]))
            bm.faces.new((c[j + 1], d[j + 1], d[j], c[j]))
        bm.faces.new((a[n_rad], b[n_rad], d[n_rad], c[n_rad]))  # eave edge
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    return link(bm, name, 0)


def firebox(name, r, z0, z1, post, lod):
    """Hexagonal fire box: floor and lintel rings joined by six corner posts; open windows."""
    obs = [prism(name + "_sill", r, r, z0, z0 + 0.035), prism(name + "_lintel", r * 1.04, r, z1 - 0.04, z1)]
    for k in range(HEX):
        a = k / HEX * 2 * math.pi
        c = Vector((r * 0.93 * math.cos(a), r * 0.93 * math.sin(a), 0))
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((c.x, c.y, (z0 + z1) / 2)) @ Matrix.Rotation(a, 4, "Z") @ Matrix.Diagonal((post, post * 1.3, z1 - z0, 1)))
        obs.append(link(bm, f"{name}_post{k}", 0, 0.006 if lod == 0 else 0.0))
    # Paper lantern inside: a hexagonal shoji drum that glows through the windows.
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=HEX, radius1=r * 0.72, radius2=r * 0.72, depth=(z1 - z0) * 0.78,
                          matrix=Matrix.Translation((0, 0, (z0 + z1) / 2)) @ Matrix.Rotation(math.pi / 6, 4, "Z"))
    paper = link(bm, name + "_paper", 1)
    obs.append(paper)
    return obs


def build(variant, lod):
    """Variant 0: classic Kasuga-doro (~1.5 m). Variant 1: a squatter, older one (~1.15 m)."""
    s = 1.0 if variant == 0 else 0.78
    w = 1.0 if variant == 0 else 1.15
    segs_round = 24 if lod == 0 else 10
    obs = []
    z = 0.0
    obs.append(prism("plinth", 0.3 * w * s, 0.27 * w * s, z, z + 0.12 * s, bevel=0.015 if lod == 0 else 0))
    z += 0.12 * s
    # Lotus cushion: a swelling ring of petals (scalloped profile) under the pole.
    lotus = [(0.2 * w * s, z), (0.24 * w * s, z + 0.03 * s), (0.2 * w * s, z + 0.07 * s), (0.1 * w * s, z + 0.09 * s)]
    obs.append(lathe("lotus", lotus, segs_round, bevel=0))
    z += 0.09 * s
    pole_h = 0.55 * s
    pole = [(0.085 * w * s, z), (0.08 * w * s, z + pole_h * 0.45), (0.095 * w * s, z + pole_h * 0.48),
            (0.095 * w * s, z + pole_h * 0.52), (0.078 * w * s, z + pole_h * 0.55), (0.072 * w * s, z + pole_h)]
    obs.append(lathe("pole", pole, segs_round))
    z += pole_h
    obs.append(prism("platform", 0.17 * w * s, 0.26 * w * s, z, z + 0.1 * s, bevel=0.01 if lod == 0 else 0))
    z += 0.1 * s
    fb_h = 0.27 * s
    obs += firebox("firebox", 0.17 * w * s, z, z + fb_h, 0.035 * s, lod)
    z += fb_h
    roof_ang, roof_rad = (72, 8) if lod == 0 else (24, 3)
    obs.append(roof("roof", 0.36 * w * s, z + 0.02 * s, 0.17 * s, 0.07 * s, 0.035 * s, roof_ang, roof_rad))
    z += 0.02 * s + 0.17 * s
    jewel = [(0.05 * s, z - 0.01), (0.06 * s, z + 0.02 * s), (0.035 * s, z + 0.035 * s), (0.065 * s, z + 0.07 * s),
             (0.06 * s, z + 0.1 * s), (0.03 * s, z + 0.135 * s), (0.0, z + 0.16 * s)]
    obs.append(lathe("hoju", jewel, segs_round))
    return obs, z + 0.16 * s


def export(obs, height):
    """Applies modifiers, joins, triangulates; returns packed vertices and indices (Y-up)."""
    dg = bpy.context.evaluated_depsgraph_get()
    pos_l, nrm_l, uv_l, part_l, idx_l = [], [], [], [], []
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
        # Flat-ish shading for faceted stone: split per triangle so bevels catch light.
        co = co.reshape(-1, 3)[tris]
        fn = np.cross(co[1::3] - co[0::3], co[2::3] - co[0::3])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
        sm = nr.reshape(-1, 3)[tris]
        fnr = np.repeat(fn, 3, axis=0)
        # Keep smooth normals where the surface is curved (lathes, roof), facets elsewhere.
        cosang = np.sum(sm * fnr, axis=1, keepdims=True)
        nrm = np.where(cosang > 0.93, sm, fnr)
        pos_l.append(co)
        nrm_l.append(nrm)
        part_l.append(np.full(len(co), ob["part"], np.float32))
        idx_l.append(np.arange(len(co)) + base)
        base += len(co)
        ev.to_mesh_clear()
    co = np.concatenate(pos_l)
    nr = np.concatenate(nrm_l)
    part = np.concatenate(part_l)
    idx = np.concatenate(idx_l).reshape(-1, 3)
    pos = np.stack([co[:, 0], co[:, 2], co[:, 1]], 1)
    nrm = np.stack([nr[:, 0], nr[:, 2], nr[:, 1]], 1)
    idx = idx[:, [0, 2, 1]]
    hN = np.clip(pos[:, 1] / height, 0, 1)
    up = np.clip(nrm[:, 1], 0, 1)
    # Moss settles on upward faces, more toward the base and in the roof's hollows.
    moss = np.clip(up ** 2 * (0.6 + 0.4 * (1 - hN)), 0, 1)
    # Occlusion: undersides (roof soffit, platform) and the inside of the fire box are dark.
    radial = np.linalg.norm(pos[:, [0, 2]], axis=1)
    ao = np.clip(0.55 + 0.45 * (nrm[:, 1] * 0.5 + 0.5), 0, 1)
    ao = np.where((part == 0) & (radial < 0.12) & (hN > 0.5) & (hN < 0.8), ao * 0.6, ao)
    v = np.zeros(len(pos), dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
    v["p"][:, :3] = pos
    v["p"][:, 3] = part
    v["n"][:, :3] = np.clip(np.round(nrm * 127), -127, 127)
    # UV: angle around the axis and height (paper lattice and stone grain).
    v["t"] = np.stack([np.arctan2(pos[:, 2], pos[:, 0]) / (2 * np.pi) + 0.5, pos[:, 1]], 1)
    v["e"] = np.clip(np.round(np.stack([hN, moss, ao, np.zeros_like(hN)], 1) * 255), 0, 255)
    return v, idx.reshape(-1)


def main():
    vbytes, ibytes, variants = [], [], []
    vbase = ioff = 0
    preview = None
    for variant in range(2):
        lods = []
        for lod in range(2):
            reset()
            obs, height = build(variant, lod)
            if variant == 0 and lod == 0 and PREVIEW:
                render_preview(obs)
                reset()
                obs, height = build(variant, lod)
            v, idx = export(obs, height)
            vbytes.append(v.tobytes())
            ibytes.append((idx + vbase).astype("<u4").tobytes())
            lods.append({"firstIndex": ioff, "indexCount": int(idx.size)})
            print(f"[lantern] variant {variant} lod {lod}: {idx.size // 3} tris, h {height:.2f}", flush=True)
            vbase += len(v)
            ioff += idx.size
        # Light center: middle of the fire box.
        light_y = (0.12 + 0.09 + 0.55 + 0.1 + 0.135) * (1.0 if variant == 0 else 0.78)
        variants.append({"height": round(height, 3), "lightY": round(light_y, 3), "lods": lods})
    with open(os.path.join(OUT, "lantern.bin"), "wb") as f:
        f.write(b"".join(vbytes))
        f.write(b"".join(ibytes))
    json.dump({"vertexBytes": vbase * 20, "indexCount": ioff, "variants": variants,
               "credit": "Modeled in Blender (tools/blender/model_lantern.py)"},
              open(os.path.join(OUT, "lantern.json"), "w"), indent=2)


def render_preview(obs):
    scene = bpy.context.scene
    stone = bpy.data.materials.new("stone")
    stone.use_nodes = True
    b = stone.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.32, 0.31, 0.28, 1)
    b.inputs["Roughness"].default_value = 0.85
    paper = bpy.data.materials.new("paper")
    paper.use_nodes = True
    p = paper.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (1.0, 0.75, 0.4, 1)
    p.inputs["Emission Color"].default_value = (1.0, 0.6, 0.25, 1)
    p.inputs["Emission Strength"].default_value = 6.0
    for ob in obs:
        ob.data.materials.append(paper if ob["part"] == 1 else stone)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 0.6
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(60), 0, math.radians(30))
    scene.collection.objects.link(so)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.06, 0.12, 1)
    scene.world = world
    cam = bpy.data.cameras.new("cam")
    cam.lens = 50
    co = bpy.data.objects.new("cam", cam)
    co.location = (1.6, -2.6, 1.3)
    co.rotation_euler = (Vector((0, 0, 0.8)) - co.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 1100
    scene.render.filepath = os.path.abspath(PREVIEW)
    bpy.ops.render.render(write_still=True)
    print("[lantern] preview", PREVIEW, flush=True)


main()
