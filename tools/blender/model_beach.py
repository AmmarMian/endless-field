"""
Builds the beach's props in Blender and exports each family as a glTF whose top-level nodes
are its variants (the layout tools/blender/build_plants.py expects):

  blender -b --factory-startup --python tools/blender/model_beach.py -- <assets-src/props> [--only a,b]

Families:
  rockshelf   wave-cut rock platforms (Poly Haven coast_rocks_01/02/03 scans), decimated;
              their textures packed into one atlas
  searocks    boulder groups and ledges (Poly Haven coast_rocks_05, coast_land_rocks_02/04)
  driftwood   sea-bleached logs, forked branches, a root stump and sticks, plus chestnut-paling
              sand fence; modelled here, the silvered grain baked from a procedural material
  shells      cockle, scallop, mussel, whelk, razor clam, limpet (palette texture)
  seaweed     bladderwrack, kelp straps and heaped wrack (palette texture)
  marram      marram grass tussocks: rolled grey-green leaves arching out, straw tips
"""
import bpy, bmesh, math, os, random, sys
import numpy as np
from mathutils import Matrix, Vector, noise

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = argv[0]
ONLY = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
rnd = random.Random(7)
V = Vector


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def export(family):
    d = os.path.join(OUT, family)
    os.makedirs(d, exist_ok=True)
    for o in bpy.context.scene.objects:
        o.select_set(o.type == "MESH")
    bpy.ops.export_scene.gltf(filepath=os.path.join(d, f"{family}.gltf"), export_format="GLTF_SEPARATE", export_image_format="JPEG", use_selection=True)
    obs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in obs)
    print(f"[beach] {family}: {len(obs)} variants, {tris} tris", flush=True)


def mesh_from(name, verts, faces, uvs=None, offset=(0, 0, 0), mat=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                uv.data[li].uv = uvs[me.loops[li].vertex_index]
    for p in me.polygons:
        p.use_smooth = smooth
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = offset
    bpy.context.scene.collection.objects.link(ob)
    return ob


def image_material(name, img):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    t = m.node_tree.nodes.new("ShaderNodeTexImage")
    t.image = img
    m.node_tree.links.new(t.outputs["Color"], m.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def save_image(name, px, size):
    """px: (h, w, 4) float in Blender's bottom-up row order."""
    img = bpy.data.images.new(name, size[0], size[1])
    img.pixels.foreach_set(px.astype(np.float32).ravel())
    path = os.path.join(bpy.app.tempdir, f"{name}.png")
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return img


# ---------------------------------------------------------------------------------------
# Scanned rocks: import, join, decimate, recentre on the base, pack textures in a 2x2 atlas.


def scanned(family, sources, target_tris):
    reset()
    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    atlas = np.zeros((2048, 2048, 4), np.float32)
    obs = []
    for i, (src, keep_h) in enumerate(sources):
        bpy.ops.import_scene.gltf(filepath=os.path.abspath(f"assets-src/{src}/{src}.gltf"))
        parts = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        bpy.ops.object.select_all(action="DESELECT")
        for o in parts:
            o.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        if len(parts) > 1:
            bpy.ops.object.join()
        ob = bpy.context.view_layer.objects.active
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # The texture before the material is replaced by the atlas.
        mat = ob.data.materials[0]
        tex = next(n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and "diff" in n.image.name)
        tex.scale(1024, 1024)
        px = np.array(tex.pixels[:], np.float32).reshape(1024, 1024, 4)
        cx, cy = cells[i]
        atlas[cy * 1024 : (cy + 1) * 1024, cx * 1024 : (cx + 1) * 1024] = px
        n = len(ob.data.polygons)
        m = ob.modifiers.new("dec", "DECIMATE")
        m.ratio = min(1.0, target_tris / max(n, 1))
        bpy.ops.object.modifier_apply(modifier=m.name)
        # UVs into this rock's atlas cell (inset half a texel against bleeding).
        uvl = ob.data.uv_layers.active.data
        uv = np.zeros(len(uvl) * 2, np.float32)
        uvl.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        uv = np.clip(uv - np.floor(uv), 0.001, 0.999)
        uv = (uv * 0.998 + 0.001 + np.array([cx, cy])) * 0.5
        uvl.foreach_set("uv", uv.ravel())
        # Base at z = 0 (the lowest `keep_h` m are what sinks into the sand), centred.
        co = np.zeros(len(ob.data.vertices) * 3, np.float32)
        ob.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        co[:, 0] -= (co[:, 0].max() + co[:, 0].min()) / 2
        co[:, 1] -= (co[:, 1].max() + co[:, 1].min()) / 2
        co[:, 2] -= co[:, 2].min() + keep_h
        ob.data.vertices.foreach_set("co", co.ravel())
        ob.data.update()
        ob.name = f"{family}_{i}"
        obs.append(ob)
        print(f"[beach] {src}: {n} -> {len(ob.data.polygons)} faces, {tuple(round(d, 1) for d in ob.dimensions)}", flush=True)
    img = save_image(f"{family}_atlas", atlas, (2048, 2048))
    mat = image_material(f"{family}_mat", img)
    for i, ob in enumerate(obs):
        ob.data.materials.clear()
        ob.data.materials.append(mat)
        ob.location = (i * 100, 0, 0)
    for img in list(bpy.data.images):
        if img.name != f"{family}_atlas":
            bpy.data.images.remove(img)
    export(family)


# ---------------------------------------------------------------------------------------
# Driftwood: tapered, bent, knotted logs and branches, silvered grain baked in Cycles.


def log_mesh(bm, path, radii, segs=10, twist=0.0, jag=(0.0, 0.0)):
    """A tube along path with per-ring radius; ends roughened (broken wood)."""
    pts = [V(p) for p in path]
    rings = []
    for i, p in enumerate(pts):
        d = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        o = d.orthogonal().normalized()
        o2 = d.cross(o)
        ring = []
        for k in range(segs):
            a = 2 * math.pi * k / segs + twist * i
            # Weathered, grooved surface: ridges along the grain and lumpy knots.
            groove = 1 + 0.06 * math.sin(a * 5 + i * 0.4) + 0.05 * noise.noise(p * 3.1 + V((math.cos(a), math.sin(a), 0)))
            r = radii[i] * groove
            q = p + (o * math.cos(a) + o2 * math.sin(a)) * r
            if i == 0 and jag[0]:
                q += d * (-jag[0] * (0.5 + 0.5 * math.sin(a * 3 + 1)))
            if i == len(pts) - 1 and jag[1]:
                q += d * (jag[1] * (0.5 + 0.5 * math.sin(a * 2 + 2.5)))
            ring.append(bm.verts.new(q))
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(segs):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % segs], rings[i + 1][(k + 1) % segs], rings[i + 1][k]))
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])


def bent(a, b, n, sway, seed):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        p = V(a).lerp(V(b), t)
        p += V((0, noise.noise(V((t * 2.3, seed, 0))) * sway, abs(noise.noise(V((t * 1.7, seed + 3, 1)))) * sway * 0.4))
        pts.append(p)
    return pts


def driftwood():
    reset()
    specs = []
    # Long log, broken at both ends, with two snapped branch stubs.
    specs.append(("log", [(bent((-1.6, 0, 0.16), (1.7, 0, 0.13), 14, 0.12, 1), [0.17 - 0.04 * t / 13 for t in range(14)], (0.08, 0.12)),
                          (bent((0.3, 0, 0.2), (0.75, 0.42, 0.42), 5, 0.04, 2), [0.06 - 0.008 * t for t in range(5)], (0, 0.03)),
                          (bent((-0.8, 0, 0.18), (-1.0, -0.38, 0.2), 4, 0.03, 3), [0.05 - 0.008 * t for t in range(4)], (0, 0.03))]))
    # Forked branch, lying flat.
    specs.append(("fork", [(bent((-1.1, 0, 0.06), (0.2, 0, 0.06), 9, 0.08, 4), [0.07 - 0.002 * t for t in range(9)], (0.04, 0)),
                           (bent((0.2, 0, 0.06), (1.2, 0.35, 0.05), 7, 0.06, 5), [0.05 - 0.004 * t for t in range(7)], (0, 0.02)),
                           (bent((0.2, 0, 0.06), (1.0, -0.5, 0.05), 7, 0.06, 6), [0.045 - 0.004 * t for t in range(7)], (0, 0.02))]))
    # Root stump: a short trunk and roots reaching out like fingers.
    roots = [(bent((-0.3, 0, 0.32), (0.5, 0, 0.3), 6, 0.05, 7), [0.3, 0.29, 0.28, 0.26, 0.25, 0.24], (0, 0.1))]
    for k in range(7):
        a = k / 7 * math.pi * 2 + 0.3
        end = (-0.35 + math.cos(a) * 0.15 - 0.6, math.sin(a) * 1.0, 0.25 + 0.35 * math.sin(a) ** 2 * (k % 2))
        roots.append((bent((-0.35, math.sin(a) * 0.1, 0.32), end, 6, 0.1, 8 + k), [0.09 - 0.013 * t for t in range(6)], (0, 0.02)))
    specs.append(("stump", roots))
    # Short weathered chunk.
    specs.append(("chunk", [(bent((-0.45, 0, 0.11), (0.45, 0, 0.12), 6, 0.03, 20), [0.12, 0.125, 0.12, 0.118, 0.11, 0.1], (0.05, 0.07))]))
    # A tangle of sticks.
    sticks = []
    for k in range(6):
        a = rnd.uniform(0, math.pi)
        L = rnd.uniform(0.5, 1.1)
        c = V((rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2), 0.03 + k * 0.012))
        d = V((math.cos(a), math.sin(a), 0)) * L / 2
        sticks.append((bent(c - d, c + d, 5, 0.04, 30 + k), [0.022] * 5, (0.01, 0.01)))
    specs.append(("sticks", sticks))
    # Chestnut paling sand fence: split stakes wired together, leaning, a few gone.
    pal = []
    for k in range(34):
        if rnd.random() < 0.12:
            continue
        x = -2 + k * 4 / 33
        h = rnd.uniform(0.85, 1.05)
        lean = rnd.uniform(-0.06, 0.06)
        pal.append(([(x, 0, -0.25), (x + lean * 0.5, 0.01, h * 0.5), (x + lean, 0.0, h)], [0.025, 0.024, 0.02], (0, 0.03)))
    for k in (0, 1):
        pal.append(([(-2.05, 0, 0.1 + k * 0.6), (2.05, 0, 0.12 + k * 0.6)], [0.004, 0.004], (0, 0)))
    specs.append(("fence", pal))
    obs = []
    for i, (name, parts) in enumerate(specs):
        bm = bmesh.new()
        for path, radii, jag in parts:
            segs = 5 if radii[0] < 0.01 else (6 if radii[0] < 0.03 else 10)
            log_mesh(bm, path, radii, segs=segs, twist=0.05, jag=jag)
        ob = mesh_from(f"driftwood_{name}", [], [])
        bm.to_mesh(ob.data)
        bm.free()
        for p in ob.data.polygons:
            p.use_smooth = True
        ob.location = (i * 10, 0, 0)
        obs.append(ob)
    bake_wood(obs)
    export("driftwood")


def bake_wood(obs):
    """Smart-UV all variants into one layout, bake a procedural silvered-wood colour."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")
    img = bpy.data.images.new("driftwood_albedo", 1024, 1024)
    m = bpy.data.materials.new("driftwood")
    m.use_nodes = True
    nt = m.node_tree
    N = nt.nodes
    L = nt.links
    coord = N.new("ShaderNodeTexCoord")
    mapping = N.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 7.0, 7.0)
    L.new(coord.outputs["Object"], mapping.inputs["Vector"])
    # Grain: fine bands across the wood's length (stretched along x), wavy.
    wave = N.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 3.0
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 4.0
    L.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    grain = N.new("ShaderNodeValToRGB")
    grain.color_ramp.elements[0].position = 0.25
    grain.color_ramp.elements[0].color = (0.33, 0.3, 0.26, 1)
    grain.color_ramp.elements[1].position = 0.75
    grain.color_ramp.elements[1].color = (0.7, 0.67, 0.6, 1)
    L.new(wave.outputs["Fac"], grain.inputs["Fac"])
    blot = N.new("ShaderNodeTexNoise")
    blot.inputs["Scale"].default_value = 2.5
    L.new(coord.outputs["Object"], blot.inputs["Vector"])
    warm = N.new("ShaderNodeMix")
    warm.data_type = "RGBA"
    warm.inputs["Factor"].default_value = 0.0
    L.new(blot.outputs["Fac"], warm.inputs["Factor"])
    L.new(grain.outputs["Color"], warm.inputs["A"])
    warm.inputs["B"].default_value = (0.5, 0.42, 0.32, 1)
    emit = N.new("ShaderNodeEmission")
    L.new(warm.outputs["Result"], emit.inputs["Color"])
    out = N["Material Output"]
    L.new(emit.outputs["Emission"], out.inputs["Surface"])
    tex = N.new("ShaderNodeTexImage")
    tex.image = img
    nt.nodes.active = tex
    for o in obs:
        o.data.materials.clear()
        o.data.materials.append(m)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.cycles.device = "CPU"
    bpy.ops.object.bake(type="EMIT", margin=6, use_clear=True)
    # Export a plain textured material.
    for l in list(out.inputs["Surface"].links):
        L.remove(l)
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    img.filepath_raw = os.path.join(bpy.app.tempdir, "driftwood_albedo.png")
    img.file_format = "PNG"
    img.save()


# ---------------------------------------------------------------------------------------
# Palette-textured families: each variant gets its own rows of a small texture; u runs along
# growth lines / strands, v picks the variant's row.


def palette_image(name, rows):
    """rows: list of functions u -> (r, g, b) in sRGB 0..1; each fills 16 rows of a 256 px wide image."""
    h = 16 * len(rows)
    px = np.zeros((h, 256, 4), np.float32)
    for i, f in enumerate(rows):
        for x in range(256):
            px[i * 16 : (i + 1) * 16, x, :3] = f(x / 255)
            px[i * 16 : (i + 1) * 16, x, 3] = 1
    return save_image(name, px, (256, h)), len(rows)


def row_v(i, n):
    return (i + 0.5) / n


def shell_surface(fn, nu, nv, name, offset, mat, row, nrows, closed_u=False, base=True):
    """A parametric surface fn(u, v) -> point, u along growth lines (texture u), v outward."""
    verts, faces, uvs = [], [], []
    for j in range(nv + 1):
        for i in range(nu + 1):
            u, v = i / nu, j / nv
            verts.append(fn(u, v))
            uvs.append((v * 0.98 + 0.01, row_v(row, nrows)))
    for j in range(nv):
        for i in range(nu):
            a = j * (nu + 1) + i
            faces.append((a, a + 1, a + nu + 2, a + nu + 1))
    return mesh_from(name, verts, faces, uvs, offset, mat)


def shells():
    reset()
    def bands(c0, c1, freq, dark=0.0):
        return lambda u: tuple(np.array(c0) + (np.array(c1) - np.array(c0)) * (0.5 + 0.5 * math.sin(u * freq)) * (1 - dark * (u > 0.85)))
    rows = [
        bands((0.86, 0.8, 0.68), (0.7, 0.55, 0.4), 40),  # cockle
        bands((0.9, 0.62, 0.45), (0.75, 0.35, 0.25), 18),  # scallop
        lambda u: (0.12 + 0.2 * u ** 6, 0.13 + 0.15 * u ** 6, 0.2 + 0.25 * u ** 6),  # mussel (nacre at rim)
        bands((0.82, 0.74, 0.6), (0.55, 0.45, 0.35), 55),  # whelk
        bands((0.84, 0.76, 0.62), (0.6, 0.5, 0.42), 8),  # razor
        bands((0.66, 0.6, 0.5), (0.45, 0.4, 0.34), 30),  # limpet
    ]
    img, n = palette_image("shells_palette", rows)
    mat = image_material("shells", img)
    S = 0.01  # cm -> m
    # Cockle: a ribbed dome, convex side up.
    def cockle(u, v):
        a = math.pi * (0.08 + 0.84 * u)
        r = 3.0 * v * (1 + 0.035 * math.sin(a * 28))
        return V((math.cos(a) * r * S, (math.sin(a) * r - 1.0) * S, 1.7 * (1 - v * v) ** 0.7 * S * min(1, v * 6 + 0.3)))
    def scallop(u, v):
        a = math.pi * (0.12 + 0.76 * u)
        r = 3.6 * v * (1 + 0.06 * math.sin(a * 18) ** 2)
        return V((math.cos(a) * r * S, (math.sin(a) * r - 1.2) * S, 0.9 * (1 - v * v) * S))
    def mussel(u, v):
        a = math.pi * u
        return V((math.cos(a) * 1.4 * math.sin(math.pi * v) ** 0.8 * S, (v * 5.5 - 2.7) * S, math.sin(a) * 1.0 * math.sin(math.pi * v) ** 0.7 * S))
    def razor(u, v):
        a = math.pi * u
        return V((math.cos(a) * 0.8 * S, (v * 12 - 6) * S, math.sin(a) * 0.45 * S))
    def limpet(u, v):
        a = 2 * math.pi * u
        r = 2.4 * v * (1 + 0.05 * math.sin(a * 22))
        return V((math.cos(a) * r * S, math.sin(a) * r * 0.85 * S, 1.6 * (1 - v) ** 1.3 * S))
    def whelk(u, v):
        # A conical spiral: whorls shrinking toward the apex, lying on its side.
        turns = 4.5
        t = v
        ang = t * turns * 2 * math.pi
        R = 1.7 * math.exp(-t * 2.2)
        cr = 1.25 * math.exp(-t * 2.2)
        centre = V((math.cos(ang) * R, math.sin(ang) * R, t * 6.0))
        a = 2 * math.pi * u
        p = centre + V((math.cos(ang) * math.cos(a) * cr, math.sin(ang) * math.cos(a) * cr, math.sin(a) * cr))
        return V((p.x * S, (p.z - 2.5) * S, (p.y + 1.6) * S))
    shapes = [(cockle, 18, 10), (scallop, 22, 8), (mussel, 10, 14), (whelk, 10, 36), (razor, 6, 14), (limpet, 20, 8)]
    for i, (fn, nu, nv) in enumerate(shapes):
        shell_surface(fn, nu, nv, f"shell_{i}", (i * 0.3, 0, 0), mat, i, n)
    export("shells")


def ribbon(bm, uvs, path, widths, normal, row, nrows, thick=0.0):
    """A flat strap along path (lying in the plane with `normal` up); u = along."""
    pts = [V(p) for p in path]
    left, right = [], []
    for i, p in enumerate(pts):
        d = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        s = d.cross(V(normal)).normalized()
        u = i / (len(pts) - 1)
        for side, lst in ((-1, left), (1, right)):
            v = bm.verts.new(p + s * widths[i] * 0.5 * side)
            uvs[v] = (u * 0.98 + 0.01, row_v(row, nrows))
            lst.append(v)
    for i in range(len(pts) - 1):
        bm.faces.new((left[i], right[i], right[i + 1], left[i + 1]))


def wavy(a, b, n, amp, freq, seed, lift=0.0):
    out = []
    for i in range(n):
        t = i / (n - 1)
        p = V(a).lerp(V(b), t)
        d = (V(b) - V(a)).normalized()
        s = V((-d.y, d.x, 0))
        p += s * math.sin(t * freq + seed) * amp + V((0, 0, lift * math.sin(t * math.pi) + 0.005 + 0.01 * abs(noise.noise(V((t * 5, seed, 0))))))
        out.append(p)
    return out


def bm_with_uvs(name, bm, uvs, offset, mat):
    me = bpy.data.meshes.new(name)
    lay = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for l in f.loops:
            l[lay].uv = uvs[l.vert]
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = offset
    bpy.context.scene.collection.objects.link(ob)
    return ob


def seaweed():
    reset()
    rows = [
        lambda u: (0.48 - 0.1 * u, 0.4 - 0.08 * u, 0.16),  # bladderwrack: olive, darkening
        lambda u: (0.52 + 0.06 * math.sin(u * 30), 0.4, 0.17),  # kelp: golden brown
        lambda u: (0.3, 0.26, 0.12),  # old wrack, dark
        lambda u: (0.56, 0.47, 0.2),  # bladders
    ]
    img, n = palette_image("seaweed_palette", rows)
    mat = image_material("seaweed", img)
    up = (0, 0, 1)
    # Bladderwrack: dichotomous forking fronds lying on the sand, with paired bladders.
    for vi in range(2):
        bm, uvs = bmesh.new(), {}
        def frond(p, ang, L, w, depth):
            e = p + V((math.cos(ang), math.sin(ang), 0)) * L
            ribbon(bm, uvs, wavy(p, e, 6, 0.02, 4, depth + vi, 0.01), [w] * 6, up, 0, n)
            if depth < 3:
                for s in (-1, 1):
                    frond(e, ang + s * rnd.uniform(0.25, 0.5), L * 0.8, w * 0.9, depth + 1)
            if depth >= 1:
                for s in (-1, 1):
                    c = p.lerp(e, 0.5) + V((-math.sin(ang), math.cos(ang), 0)) * s * w * 0.6 + V((0, 0, 0.008))
                    res = bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=w * 0.45)
                    bmesh.ops.translate(bm, verts=res["verts"], vec=c)
                    for v in res["verts"]:
                        uvs[v] = (0.5, row_v(3, n))
        for k in range(3 + vi):
            frond(V((0, 0, 0)), k * 2.1 + vi, 0.2, 0.045, 0)
        bm_with_uvs(f"seaweed_wrack{vi}", bm, uvs, (vi * 3, 0, 0), mat)
    # Kelp: a holdfast, a stipe and long wavy straps.
    bm, uvs = bmesh.new(), {}
    for k in range(4):
        a = k * 0.5 - 0.7
        ribbon(bm, uvs, wavy((0, 0, 0.02), (math.cos(a) * 1.6, math.sin(a) * 1.6, 0), 14, 0.06, 9, k, 0.03), [0.05 + 0.08 * math.sin(math.pi * min(1, t / 3)) for t in range(14)], up, 1, n)
    res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.06)
    for v in res["verts"]:
        v.co.z *= 0.6
        uvs[v] = (0.1, row_v(2, n))
    bm_with_uvs("seaweed_kelp", bm, uvs, (6, 0, 0), mat)
    # A heap of old wrack: many short dark strands piled up.
    bm, uvs = bmesh.new(), {}
    for k in range(40):
        a = rnd.uniform(0, math.pi * 2)
        c = V((rnd.gauss(0, 0.25), rnd.gauss(0, 0.12), 0))
        L = rnd.uniform(0.15, 0.4)
        d = V((math.cos(a), math.sin(a), 0)) * L / 2
        h = max(0.0, 0.08 - c.length * 0.15) + 0.004 * k
        ribbon(bm, uvs, wavy(c - d + V((0, 0, h)), c + d + V((0, 0, h)), 5, 0.02, 6, k, 0.02), [0.04] * 5, up, 2 if k % 3 else 0, n)
    bm_with_uvs("seaweed_heap", bm, uvs, (9, 0, 0), mat)
    export("seaweed")


def marram():
    reset()
    rows = [
        lambda u: tuple(np.array((0.62, 0.7, 0.55)) * (0.6 + 0.4 * min(1, u * 2.5)) * (1 - u ** 2) + np.array((0.86, 0.8, 0.62)) * u ** 2),  # living leaf
        lambda u: tuple(np.array((0.66, 0.6, 0.46)) * (1 - u) + np.array((0.86, 0.8, 0.62)) * u),  # dead, straw
        lambda u: tuple(np.array((0.48, 0.6, 0.4)) * (0.6 + 0.4 * min(1, u * 2.5)) * (1 - u ** 2 * 0.6) + np.array((0.75, 0.7, 0.5)) * u ** 2 * 0.6),  # young, greener
    ]
    img, n = palette_image("marram_palette", rows)
    mat = image_material("marram", img)
    for vi, (count, spread, tall) in enumerate(((220, 0.28, 0.95), (150, 0.2, 0.75), (320, 0.4, 1.1), (110, 0.15, 0.6))):
        bm, uvs = bmesh.new(), {}
        for k in range(count):
            # Leaves rise from a tight crown and arch outward (a fountain), the outer ones more.
            a = rnd.uniform(0, math.pi * 2)
            r0 = math.sqrt(rnd.random()) * spread
            base = V((math.cos(a) * r0, math.sin(a) * r0, -0.05))
            L = tall * rnd.uniform(0.6, 1.15) * (1.0 - 0.3 * r0 / spread)
            outward = 0.15 + 0.85 * (r0 / spread) ** 1.3 + rnd.uniform(-0.1, 0.2)
            dirx = V((math.cos(a), math.sin(a), 0))
            pts = []
            for i in range(6):
                t = i / 5
                p = base + V((0, 0, L * t * (1 - 0.35 * outward * t))) + dirx * L * outward * 0.55 * t * t
                pts.append(p)
            w = rnd.uniform(0.01, 0.017)
            dead = rnd.random() < (0.25 if vi != 2 else 0.4)
            row = 1 if dead else (2 if rnd.random() < 0.3 else 0)
            # The blade's flat side faces sideways to its arch (a rolled leaf reads as a thin line).
            ribbon(bm, uvs, pts, [w * (1 - 0.85 * (i / 5) ** 1.5) for i in range(6)], (dirx.y, -dirx.x, 0.0) if outward > 0.05 else (1, 0, 0), row, n)
        bm_with_uvs(f"marram_{vi}", bm, uvs, (vi * 2, 0, 0), mat)
    export("marram")


FAMILIES = {
    "rockshelf": lambda: scanned("rockshelf", [("coast_rocks_01", 0.6), ("coast_rocks_02", 0.6), ("coast_rocks_03", 0.6)], 5000),
    "searocks": lambda: scanned("searocks", [("coast_rocks_05", 0.2), ("coast_land_rocks_02", 0.3), ("coast_land_rocks_04", 0.2)], 2400),
    "driftwood": driftwood,
    "shells": shells,
    "seaweed": seaweed,
    "marram": marram,
}

for name, fn in FAMILIES.items():
    if ONLY and name not in ONLY:
        continue
    fn()
