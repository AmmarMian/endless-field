"""
Generates eroded mountain massifs as heightfields (run inside Blender).

  blender -b --factory-startup --python tools/blender/build_mountains.py -- <out_dir> [count]
  MOUNTAIN_PREVIEW=1 also renders a shaded review image per massif.

Each massif: ridged multifractal (Blender's mathutils.noise) under a radial falloff, so a
stamp is an island of peaks that blends into the surrounding hills. Then droplet-based
hydraulic erosion and thermal (talus) erosion carve valleys, ridges and scree slopes.

Output per massif: massif_<i>.bin, float16 heights normalized to [0, 1] (row-major, N x N),
plus mountains.json with size/extent. The runtime scales heights to meters.
"""
import bpy, json, math, os, sys
import numpy as np
from mathutils import Vector, noise

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
COUNT = int(argv[1]) if len(argv) > 1 else 3
N = 512                     # heightfield resolution
EXTENT = 2400.0             # meters covered by one massif
os.makedirs(OUT, exist_ok=True)


def base_field(seed):
    rng = np.random.default_rng(seed)
    off = Vector(rng.uniform(-500, 500, 3))
    h = np.zeros((N, N), np.float64)
    coords = (np.arange(N) / (N - 1)) * 2 - 1
    # Domain warp gives the ridges natural bends instead of noise-lattice alignment.
    for j, y in enumerate(coords):
        for i, x in enumerate(coords):
            p = Vector((x * 2.6, y * 2.6, 0.0)) + off
            w = noise.noise_vector(p * 0.7)
            q = p + w * 0.45
            h[j, i] = noise.ridged_multi_fractal(q, 0.95, 2.1, 7, 1.0, 2.1, noise_basis="PERLIN_ORIGINAL")
    h = (h - h.min()) / (h.max() - h.min())
    # Radial massif falloff with a wobbly outline, so peaks cluster and edges meet the hills.
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    ang = np.arctan2(yy, xx)
    wob = 0.82 + 0.12 * np.sin(ang * 3 + rng.uniform(0, 6)) + 0.06 * np.sin(ang * 7 + rng.uniform(0, 6))
    r = np.sqrt(xx * xx + yy * yy) / wob
    fall = np.clip(1 - r, 0, 1) ** 1.4
    # A main summit off-center makes each massif read as one mountain with satellites.
    cx, cy = rng.uniform(-0.25, 0.25, 2)
    summit = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.18)
    return (h ** 1.5) * fall * (0.45 + 0.55 * summit)


def stream_power(h, iters=60, k=0.0032, m=0.5):
    """Fluvial incision: each cell is lowered by k * A^m * S (A = drainage area, S = slope
    to its steepest-descent neighbor), never below that neighbor. Stable, and it carves
    dendritic valleys with sharp ridges between them."""
    n = h.shape[0]
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([math.hypot(dy, dx) for dy, dx in offs])
    flat_idx = np.arange(n * n).reshape(n, n)
    hh = h.copy()
    for it in range(iters):
        pad = np.pad(hh, 1, mode="edge")
        pidx = np.pad(flat_idx, 1, mode="edge")
        drops = np.stack([(hh - pad[1 + dy:1 + dy + n, 1 + dx:1 + dx + n]) / dist[i] for i, (dy, dx) in enumerate(offs)])
        best = np.argmax(drops, axis=0)
        slope = np.maximum(np.take_along_axis(drops, best[None], 0)[0], 0.0)
        rcv = np.stack([pidx[1 + dy:1 + dy + n, 1 + dx:1 + dx + n] for dy, dx in offs])
        rcv = np.take_along_axis(rcv, best[None], 0)[0].ravel()
        rcv = np.where(slope.ravel() > 0, rcv, flat_idx.ravel())
        # Drainage area: accumulate from the highest cells down.
        order = np.argsort(-hh.ravel()).tolist()
        area = [1.0] * (n * n)
        rl = rcv.tolist()
        for c in order:
            r = rl[c]
            if r != c:
                area[r] += area[c]
        area = np.array(area).reshape(n, n)
        incision = k * np.power(area, m) * slope
        floor = hh.ravel()[rcv].reshape(n, n)
        hh = np.maximum(hh - incision, np.minimum(hh, floor + 1e-5))
        if it % 6 == 0:
            hh = thermal(hh, iters=1, talus=0.008)
    return hh


def gaussian_smooth(h, sigma):
    """Separable Gaussian blur (edge-clamped): rounds ridges and peaks, keeps valleys."""
    r = int(sigma * 3)
    x = np.arange(-r, r + 1)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    pad = np.pad(h, r, mode="edge")
    tmp = sum(k[i] * pad[:, i:i + h.shape[1]] for i in range(2 * r + 1))
    tmp = tmp[r:-r, :] if r else tmp
    pad = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
    return sum(k[i] * pad[i:i + h.shape[0], :] for i in range(2 * r + 1))


def thermal(h, iters=40, talus=0.0035, k=0.4):
    """Material slides off slopes steeper than the talus angle, leaving scree aprons."""
    hh = h.copy()
    for _ in range(iters):
        total = np.zeros_like(hh)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(np.roll(hh, dy, 0), dx, 1)
            d = hh - nb
            move = np.where(d > talus, (d - talus) * k * 0.25, 0.0)
            total -= move
            total += np.roll(np.roll(move, -dy, 0), -dx, 1)
        hh += total
    return hh


meta = {"size": N, "extent": EXTENT, "massifs": []}
for i in range(COUNT):
    h = base_field(100 + i * 17)
    if not os.environ.get("SKIP_EROSION"):
        h = stream_power(h)
        h = thermal(h, iters=4, talus=0.007)
    h = np.clip(h, 0, None)
    # Soft, rounded alpine forms: blur away the knife-edge ridges left by noise + erosion.
    h = gaussian_smooth(h, float(os.environ.get("MOUNTAIN_SMOOTH", "3.2")))
    # Keep the rim exactly at zero so stamps blend seamlessly.
    coords = (np.arange(N) / (N - 1)) * 2 - 1
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    rim = np.clip((1 - np.maximum(abs(xx), abs(yy))) / 0.08, 0, 1)
    h = h * rim
    h = h / h.max()
    path = os.path.join(OUT, f"massif_{i}.bin")
    h.astype("<f2").tofile(path)
    meta["massifs"].append(os.path.basename(path))
    print(f"[mountains] massif {i}: mean {h.mean():.3f}", flush=True)

    if os.environ.get("MOUNTAIN_PREVIEW"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        me = bpy.data.meshes.new("m")
        R = 256
        sub = h[:: N // R, :: N // R][:R, :R]
        xs = np.linspace(-EXTENT / 2, EXTENT / 2, R)
        verts = [(x, y, sub[j, i2] * 420.0) for j, y in enumerate(xs) for i2, x in enumerate(xs)]
        faces = [(j * R + i2, j * R + i2 + 1, (j + 1) * R + i2 + 1, (j + 1) * R + i2) for j in range(R - 1) for i2 in range(R - 1)]
        me.from_pydata(verts, [], faces)
        me.update()
        for p in me.polygons:
            p.use_smooth = True
        ob = bpy.data.objects.new("m", me)
        scene.collection.objects.link(ob)
        mat = bpy.data.materials.new("alpine")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Normal"], sep.inputs[0])
        pos = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Position"], pos.inputs[0])
        ramp_h = nt.nodes.new("ShaderNodeValToRGB")
        ramp_h.color_ramp.elements[0].color = (0.25, 0.3, 0.12, 1)
        ramp_h.color_ramp.elements[1].position = 0.75
        ramp_h.color_ramp.elements[1].color = (0.42, 0.4, 0.36, 1)
        mapr = nt.nodes.new("ShaderNodeMapRange")
        mapr.inputs[2].default_value = 420.0
        nt.links.new(pos.outputs["Z"], mapr.inputs[0])
        nt.links.new(mapr.outputs[0], ramp_h.inputs[0])
        snow = nt.nodes.new("ShaderNodeMath"); snow.operation = "MULTIPLY"
        hi = nt.nodes.new("ShaderNodeMapRange"); hi.inputs[1].default_value = 230; hi.inputs[2].default_value = 300
        nt.links.new(pos.outputs["Z"], hi.inputs[0])
        flat = nt.nodes.new("ShaderNodeMapRange"); flat.inputs[1].default_value = 0.6; flat.inputs[2].default_value = 0.85
        nt.links.new(sep.outputs["Z"], flat.inputs[0])
        nt.links.new(hi.outputs[0], snow.inputs[0]); nt.links.new(flat.outputs[0], snow.inputs[1])
        mix = nt.nodes.new("ShaderNodeMixRGB")
        nt.links.new(snow.outputs[0], mix.inputs[0])
        nt.links.new(ramp_h.outputs[0], mix.inputs[1])
        mix.inputs[2].default_value = (0.92, 0.94, 0.98, 1)
        nt.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
        me.materials.append(mat)
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        sun.data.energy = 4
        sun.rotation_euler = (math.radians(60), 0, math.radians(-40))
        scene.collection.objects.link(sun)
        world = bpy.data.worlds.new("w"); scene.world = world
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.65, 0.85, 1)
        cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        cam.data.clip_end = 20000
        scene.collection.objects.link(cam)
        cam.location = (0, -2300, 650)
        cam.rotation_euler = (Vector((0, 0, 120)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.camera = cam
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.resolution_x, scene.render.resolution_y = 1000, 560
        scene.render.filepath = os.path.abspath(os.path.join("tools/out", f"mountain_{i}.png"))
        bpy.ops.render.render(write_still=True)

json.dump(meta, open(os.path.join(OUT, "mountains.json"), "w"), indent=2)
print("[mountains] done", flush=True)
