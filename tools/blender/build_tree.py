"""
Builds a web-ready tree from a Poly Haven glTF (run inside Blender for decimation and baking).

  blender -b --factory-startup --python tools/blender/build_tree.py -- <config.json>

Config keys: src (gltf path), alpha (leaves alpha png), out (output dir), name,
lod0/lod1 bark decimation ratios per material, lod1_leaf_keep, impostor settings.

Outputs in <out>/:
  tree.json             manifest: bounds, LOD ranges, impostor layout, texture names
  lod0.bin, lod1.bin    interleaved vertices (20 B) + uint32 indices, bark then leaves
  leaves.png            diffuse + alpha, colors bled into transparent texels
  trunk_diff.jpg ...    bark textures copied from the source
  impostor_albedo.png, impostor_normal.png   N views around the tree (glTF Y-up frame)

Coordinates stay in glTF space (Y up, meters). Leaves: every loose leaf/frond card in the
source (a curved ~24-46 triangle strip) is refit as one bilinear quad over its UV rectangle.
"""
import bpy, bmesh, json, math, os, shutil, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gltfnp import load, components, weld  # noqa: E402
from texutil import img_np, save_png, bleed  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[tree {time.time() - T0:6.1f}s]", *a, flush=True)


cfg = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = cfg["out"]
os.makedirs(OUT, exist_ok=True)
gltf, prims = load(cfg["src"])
src_dir = os.path.dirname(cfg["src"])
rng = np.random.default_rng(cfg.get("seed", 7))


def material_textures(name):
    """Return {role: (uri, transform)} for a material, applying KHR_texture_transform if any."""
    m = next(m for m in gltf["materials"] if m["name"] == name)
    out = {}
    for role, ref in (("diff", m["pbrMetallicRoughness"].get("baseColorTexture")), ("nor", m.get("normalTexture"))):
        if not ref:
            continue
        img = gltf["images"][gltf["textures"][ref["index"]]["source"]]["uri"]
        tr = ref.get("extensions", {}).get("KHR_texture_transform")
        out[role] = (img, tr)
    return out


def apply_uv_transform(uv, tr):
    if not tr:
        return uv
    s = np.array(tr.get("scale", [1, 1]), np.float32)
    o = np.array(tr.get("offset", [0, 0]), np.float32)
    if tr.get("rotation"):
        raise SystemExit("texture rotation not supported")
    return uv * s + o


# ---------------------------------------------------------------- leaves -> cards
leaf = next(p for p in prims if "leaves" in p["material"])
w = weld(leaf["pos"])
tris = w[leaf["idx"]]
lab = components(w.max() + 1, tris)
tri_lab = lab[tris[:, 0]]
order = np.argsort(tri_lab, kind="stable")
sorted_lab = tri_lab[order]
starts = np.flatnonzero(np.r_[True, sorted_lab[1:] != sorted_lab[:-1]])
ends = np.r_[starts[1:], len(order)]
log("leaf cards", len(starts))

cards_pos = np.zeros((len(starts), 4, 3), np.float32)
cards_uv = np.zeros((len(starts), 4, 2), np.float32)
cards_n = np.zeros((len(starts), 3), np.float32)
for ci, (a, b) in enumerate(zip(starts, ends)):
    vid = np.unique(leaf["idx"][order[a:b]].reshape(-1))
    P = leaf["pos"][vid].astype(np.float64)
    U = leaf["uv"][vid].astype(np.float64)
    lo, hi = U.min(0), U.max(0)
    pad = (hi - lo) * 0.02
    lo, hi = lo - pad, hi + pad
    # Bilinear least-squares fit P(u, v) = c0 + c1 u + c2 v + c3 u v
    A = np.stack([np.ones(len(U)), U[:, 0], U[:, 1], U[:, 0] * U[:, 1]], 1)
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    Ac = np.stack([np.ones(4), corners[:, 0], corners[:, 1], corners[:, 0] * corners[:, 1]], 1)
    cards_pos[ci] = Ac @ coef
    cards_uv[ci] = corners
    n = leaf["nrm"][vid].mean(0)
    cards_n[ci] = n / (np.linalg.norm(n) + 1e-9)

# Tree frame: trunk axis at x=z=0 in the source; canopy = leaf card centers.
centers = cards_pos.mean(1)
canopy_c = centers.mean(0)
canopy_r = np.percentile(np.linalg.norm(centers - canopy_c, axis=1), 90)
all_pos = np.concatenate([p["pos"] for p in prims])
bmin, bmax = all_pos.min(0), all_pos.max(0)
height = float(bmax[1])
log("bounds", bmin.round(2), bmax.round(2), "canopy", canopy_c.round(2), round(float(canopy_r), 2))


def leaf_extras(c_pos):
    """Per-vertex extras: x = wind flex (distance from trunk), y = phase, z = ao, w = part (255 = leaf)."""
    ctr = c_pos.mean(1)
    radial = np.linalg.norm(ctr[:, [0, 2]], axis=1) / (np.abs(bmax[[0, 2]]).max() + 1e-6)
    flex = np.clip(0.35 + 0.65 * radial * (ctr[:, 1] / height), 0, 1)
    depth = np.linalg.norm(ctr - canopy_c, axis=1) / (canopy_r * 1.15)
    ao = np.clip(0.35 + 0.65 * depth ** 1.5, 0.3, 1.0)
    phase = rng.random(len(ctr))
    ex = np.stack([flex, phase, ao, np.ones_like(ao)], 1)
    return np.repeat(ex[:, None, :], 4, axis=1)


def cards_to_mesh(c_pos, c_uv, c_n):
    n = len(c_pos)
    pos = c_pos.reshape(-1, 3)
    uv = c_uv.reshape(-1, 2)
    # Blend card normals toward the canopy sphere: soft, volumetric foliage shading.
    sph = c_pos - canopy_c
    sph /= np.linalg.norm(sph, axis=2, keepdims=True) + 1e-9
    nrm = (np.repeat(c_n[:, None, :], 4, 1) * 0.35 + sph * 0.65)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-9
    base = (np.arange(n) * 4)[:, None]
    idx = (base + np.array([0, 1, 2, 0, 2, 3])).reshape(-1)
    return pos, nrm.reshape(-1, 3), uv, leaf_extras(c_pos).reshape(-1, 4), idx


def subset_cards(keep):
    k = max(1, int(len(cards_pos) * keep))
    sel = rng.choice(len(cards_pos), k, replace=False)
    s = 1.0 / math.sqrt(keep)
    c = cards_pos[sel]
    ctr = c.mean(1, keepdims=True)
    return (c - ctr) * s + ctr, cards_uv[sel], cards_n[sel]


# ---------------------------------------------------------------- bark decimation in Blender
def np_mesh(name, pos, idx, uv):
    """Create a Blender mesh from glTF-space arrays (welded positions, per-corner UVs)."""
    wv = weld(pos, 4)
    uniq = np.zeros((wv.max() + 1, 3), np.float32)
    uniq[wv] = pos
    corners = wv[idx].reshape(-1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(uniq))
    me.vertices.foreach_set("co", uniq.ravel())
    me.loops.add(len(corners))
    me.loops.foreach_set("vertex_index", corners.astype(np.int32))
    me.polygons.add(len(idx))
    me.polygons.foreach_set("loop_start", np.arange(0, len(corners), 3, dtype=np.int32))
    uvl = me.uv_layers.new(name="UV")
    uvl.data.foreach_set("uv", uv[idx.reshape(-1)].ravel())
    me.update()
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def read_mesh(ob):
    deps = bpy.context.evaluated_depsgraph_get()
    eo = ob.evaluated_get(deps)
    me = eo.to_mesh()
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    loops = np.zeros(nt * 3, np.int32)
    me.loop_triangles.foreach_get("loops", loops)
    lv = np.zeros(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    co = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    cn = np.zeros(len(me.loops) * 3, np.float32)
    me.corner_normals.foreach_get("vector", cn)
    cn = cn.reshape(-1, 3)
    uv = np.zeros(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    pos = co[lv[loops]]
    nrm = cn[loops]
    uvs = uv[loops]
    eo.to_mesh_clear()
    # Re-index unique corners.
    key = np.concatenate([pos, nrm * 0.25, uvs], 1).round(5)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return uniq[:, :3], uniq[:, 3:6] * 4.0, uniq[:, 6:8], inv.reshape(-1).astype(np.int64)



def simplify_part(p, idx_src, ratio, err):
    """meshoptimizer simplify (keeps original vertices, so normals/UVs stay exact)."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.bin")
        dst = os.path.join(td, "out.bin")
        with open(src, "wb") as f:
            f.write(np.array([len(p["pos"]), idx_src.size], "<u4").tobytes())
            f.write(p["pos"].astype("<f4").tobytes())
            f.write(idx_src.astype("<u4").tobytes())
        res = subprocess.run(["node", "tools/simplify.mjs", src, dst, str(ratio), str(err)], capture_output=True, text=True, check=True)
        log("  meshopt", res.stdout.strip())
        out = np.fromfile(dst, "<u4").astype(np.int64)
    used, inv = np.unique(out, return_inverse=True)
    nrm = p["nrm"][used]
    return p["pos"][used], nrm, p["uv"][used], inv.reshape(-1)


bark_parts = []  # (material name, texture prefix, prim)
for p in prims:
    if "leaves" in p["material"]:
        continue
    texs = material_textures(p["material"])
    part = "branches" if "branches" in p["material"] else "trunk"
    p["uv"] = apply_uv_transform(p["uv"], texs["diff"][1])
    bark_parts.append((p["material"], part, p, texs))

lods = {}
for lod_name in ("lod0", "lod1"):
    lods[lod_name] = []
    for mat, part, p, texs in bark_parts:
        ratio = cfg[lod_name][part]
        idx_src = p["idx"]
        min_twig = cfg[lod_name].get(f"{part}_min_size", 0)
        if min_twig > 0:
            # Collapse decimation cannot go below one piece per loose twig, so drop the
            # smallest twigs outright; leaves hide them at this LOD's distances.
            wv = weld(p["pos"], 4)
            tl = components(wv.max() + 1, wv[idx_src])[wv[idx_src[:, 0]]]
            ids, inv = np.unique(tl, return_inverse=True)
            P = p["pos"][idx_src[:, 0]]
            lo = np.full((len(ids), 3), np.inf); hi = np.full((len(ids), 3), -np.inf)
            np.minimum.at(lo, inv, P); np.maximum.at(hi, inv, P)
            keep = np.linalg.norm(hi - lo, axis=1) >= min_twig
            idx_src = idx_src[keep[inv]]
            log(lod_name, part, "twigs kept", int(keep.sum()), "of", len(ids))
        pos, nrm, uv, idx = simplify_part(p, idx_src, ratio, cfg[lod_name].get("error", 0.01))
        log(lod_name, part, "tris", len(idx_src), "->", len(idx) // 3)
        partflag = 0.0 if part == "trunk" else 0.5
        h = np.clip(pos[:, 1] / height, 0, 1)
        radial = np.linalg.norm(pos[:, [0, 2]], axis=1) / (np.abs(bmax[[0, 2]]).max() + 1e-6)
        flex = np.clip(radial * h * 0.8, 0, 1)
        ex = np.stack([flex, np.zeros_like(h), np.ones_like(h), np.full_like(h, partflag)], 1)
        lods[lod_name].append(("bark", pos, nrm, uv, ex, idx, part))

lods["lod0"].append(("leaves", *cards_to_mesh(cards_pos, cards_uv, cards_n), "leaves"))
lods["lod1"].append(("leaves", *cards_to_mesh(*subset_cards(cfg["lod1_leaf_keep"])), "leaves"))


def pack_lod(parts, path):
    """20-byte vertices: float16x4 pos, snorm8x4 normal, float16x2 uv, unorm8x4 extras."""
    groups = {}
    vbytes = []
    ibytes = []
    vbase = 0
    ioff = 0
    for kind in ("bark", "leaves"):
        sel = [p for p in parts if p[0] == kind]
        first = ioff
        for _, pos, nrm, uv, ex, idx, _part in sel:
            n = len(pos)
            v = np.zeros(n, dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
            v["p"][:, :3] = pos
            v["n"][:, :3] = np.clip(np.round(nrm * 127), -127, 127)
            v["t"] = uv
            v["e"] = np.clip(np.round(ex * 255), 0, 255)
            vbytes.append(v.tobytes())
            ibytes.append((idx + vbase).astype("<u4").tobytes())
            vbase += n
            ioff += len(idx)
        groups[kind] = {"firstIndex": first, "indexCount": ioff - first}
    vb = b"".join(vbytes)
    ib = b"".join(ibytes)
    with open(path, "wb") as f:
        f.write(vb)
        f.write(ib)
    return {"file": os.path.basename(path), "vertexCount": vbase, "vertexBytes": len(vb), "indexCount": ioff, "groups": groups}


manifest_lods = [pack_lod(lods[n], os.path.join(OUT, f"{n}.bin")) for n in ("lod0", "lod1")]
for m in manifest_lods:
    log(m["file"], "verts", m["vertexCount"], "tris", m["indexCount"] // 3, "bytes", m["vertexBytes"] + m["indexCount"] * 4)


# ---------------------------------------------------------------- textures
leaf_tex = material_textures(leaf["material"])
diff = img_np(os.path.join(src_dir, leaf_tex["diff"][0]))
alpha = img_np(cfg["alpha"])[..., 0]
diff[..., 3] = alpha
save_png(bleed(diff), os.path.join(OUT, "leaves.png"))
textures = {"leaves": "leaves.png"}
for mat, part, p, texs in bark_parts:
    for role, (uri, _) in texs.items():
        name = f"{part}_{role}.jpg"
        shutil.copy(os.path.join(src_dir, uri), os.path.join(OUT, name))
        textures[f"{part}_{role}"] = name
log("textures", textures)

# ---------------------------------------------------------------- impostor bake (Eevee)
IMP = cfg["impostor"]
frames, tile = IMP["frames"], IMP["tile"]
cols = IMP["cols"]
rows = math.ceil(frames / cols)
radial_max = float(np.linalg.norm(all_pos[:, [0, 2]], axis=1).max())
ortho = max(radial_max * 2, height - bmin[1]) * 1.04
cam_cy = bmin[1] + ortho / 2


def to_blender(pos):
    return np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], 1)


def build_render_mesh(name, pos, idx, uv, mat):
    me = bpy.data.meshes.new(name)
    bp = to_blender(pos)
    me.vertices.add(len(bp))
    me.vertices.foreach_set("co", bp.ravel())
    corners = idx.reshape(-1).astype(np.int32)
    me.loops.add(len(corners))
    me.loops.foreach_set("vertex_index", corners)
    me.polygons.add(len(corners) // 3)
    me.polygons.foreach_set("loop_start", np.arange(0, len(corners), 3, dtype=np.int32))
    uvl = me.uv_layers.new(name="UV")
    buv = uv.copy()
    buv[:, 1] = 1 - buv[:, 1]
    uvl.data.foreach_set("uv", buv[corners].ravel())
    me.update()
    for poly in me.polygons:
        poly.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def make_material(name, diff_path, alpha_path=None):
    """Material with a switchable output: albedo (emission of the texture) or normal."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.abspath(diff_path))
    emit_alb = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tex.outputs["Color"], emit_alb.inputs["Color"])
    # Normal in glTF frame: (bx, bz, -by) * 0.5 + 0.5
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sep.inputs[0])
    neg = nt.nodes.new("ShaderNodeMath"); neg.operation = "MULTIPLY"; neg.inputs[1].default_value = -1
    nt.links.new(sep.outputs["Y"], neg.inputs[0])
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], comb.inputs[0])
    nt.links.new(sep.outputs["Z"], comb.inputs[1])
    nt.links.new(neg.outputs[0], comb.inputs[2])
    remap = nt.nodes.new("ShaderNodeVectorMath"); remap.operation = "MULTIPLY_ADD"
    remap.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap.inputs[2].default_value = (0.5, 0.5, 0.5)
    nt.links.new(comb.outputs[0], remap.inputs[0])
    emit_nrm = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(remap.outputs[0], emit_nrm.inputs["Color"])
    pick = nt.nodes.new("ShaderNodeMixShader"); pick.name = "pick"
    pick.inputs[0].default_value = 0.0
    nt.links.new(emit_alb.outputs[0], pick.inputs[1])
    nt.links.new(emit_nrm.outputs[0], pick.inputs[2])
    final = pick
    if alpha_path:
        atex = nt.nodes.new("ShaderNodeTexImage")
        atex.image = bpy.data.images.load(os.path.abspath(alpha_path))
        atex.image.colorspace_settings.name = "Non-Color"
        thr = nt.nodes.new("ShaderNodeMath"); thr.operation = "GREATER_THAN"; thr.inputs[1].default_value = 0.5
        nt.links.new(atex.outputs["Color"], thr.inputs[0])
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        cut = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(thr.outputs[0], cut.inputs[0])
        nt.links.new(transp.outputs[0], cut.inputs[1])
        nt.links.new(pick.outputs[0], cut.inputs[2])
        final = cut
    nt.links.new(final.outputs[0], out.inputs["Surface"])
    return mat


scene = bpy.context.scene
for ob in list(scene.objects):
    bpy.data.objects.remove(ob)
mats = []
for mat, part, p, texs in bark_parts:
    m = make_material(f"imp_{part}", os.path.join(src_dir, texs["diff"][0]))
    mats.append(m)
    # Render from the LOD0 bark (identical silhouette at impostor distances, much faster).
    lp = next(x for x in lods["lod0"] if x[6] == part)
    build_render_mesh(f"imp_{part}", lp[1], lp[5].reshape(-1, 3), lp[3], m)
lm = make_material("imp_leaves", os.path.join(src_dir, leaf_tex["diff"][0]), cfg["alpha"])
mats.append(lm)
build_render_mesh("imp_leaves", leaf["pos"], leaf["idx"], leaf["uv"], lm)

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = tile
scene.render.resolution_y = tile
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.look = "None"
scene.render.filter_size = 1.2
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.type = "ORTHO"
cam.data.ortho_scale = ortho
cam.data.clip_start = 0.1
cam.data.clip_end = ortho * 6
scene.collection.objects.link(cam)
scene.camera = cam


def render_atlas(mode, path):
    scene.view_settings.view_transform = "Standard"
    for m in mats:
        m.node_tree.nodes["pick"].inputs[0].default_value = 0.0 if mode == "albedo" else 1.0
    atlas = np.zeros((rows * tile, cols * tile, 4), np.float32)
    tmp = os.path.join(OUT, "_frame.png")
    for k in range(frames):
        th = k * 2 * math.pi / frames
        # glTF camera position (sin th, cy, cos th) * R -> Blender (x, -z, y)
        R = ortho * 2
        gx, gz = math.sin(th) * R, math.cos(th) * R
        cam.location = (gx, -gz, cam_cy)
        from mathutils import Vector
        cam.rotation_euler = (Vector((0, 0, cam_cy)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        fr = img_np(tmp)
        r, c = divmod(k, cols)
        # Blender pixels are bottom-up; atlas rows are stored top-down.
        atlas[(rows - 1 - r) * tile:(rows - r) * tile, c * tile:(c + 1) * tile] = fr
    os.remove(tmp)
    if mode == "normal":
        # Emission went through the sRGB view; undo it so normals are linear.
        rgb = atlas[..., :3]
        atlas[..., :3] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    save_png(bleed(atlas), path, "sRGB" if mode == "albedo" else "Non-Color")


render_atlas("albedo", os.path.join(OUT, "impostor_albedo.png"))
render_atlas("normal", os.path.join(OUT, "impostor_normal.png"))
log("impostor baked", frames, "frames", tile, "px")

manifest = {
    "name": cfg["name"],
    "source": cfg.get("credit", ""),
    "bounds": {"min": bmin.round(3).tolist(), "max": bmax.round(3).tolist()},
    "height": round(height, 3),
    "canopy": {"center": canopy_c.round(3).tolist(), "radius": round(float(canopy_r), 3)},
    "lods": manifest_lods,
    "lodDistances": cfg["lod_distances"],
    "impostor": {"frames": frames, "cols": cols, "rows": rows, "tile": tile, "size": round(float(ortho), 3), "centerY": round(float(cam_cy), 3)},
    "textures": textures,
}
json.dump(manifest, open(os.path.join(OUT, "tree.json"), "w"), indent=2)
log("done", OUT)
