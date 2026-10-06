"""
Builds instanced ground-cover plants from a Poly Haven glTF holding several variants.

  blender -b --factory-startup --python tools/blender/build_plants.py -- <config.json>

Each top-level mesh node becomes one variant, re-centered on its base and simplified with
meshoptimizer. Diffuse + alpha are merged into one color-bled RGBA texture.
"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gltfnp import load_nodes  # noqa: E402
from texutil import img_np, save_png, bleed  # noqa: E402

cfg = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = cfg["out"]
os.makedirs(OUT, exist_ok=True)
gltf, nodes = load_nodes(cfg["src"])
rng = np.random.default_rng(3)


def simplify(pos, idx, target_tris):
    ratio = min(1.0, target_tris / max(1, len(idx)))
    if ratio >= 1.0:
        return idx
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        with open(a, "wb") as f:
            f.write(np.array([len(pos), idx.size], "<u4").tobytes())
            f.write(pos.astype("<f4").tobytes())
            f.write(idx.astype("<u4").tobytes())
        r = subprocess.run(["node", "tools/simplify.mjs", a, b, str(ratio), str(cfg.get("error", 0.01))], capture_output=True, text=True, check=True)
        print("  meshopt", r.stdout.strip())
        return np.fromfile(b, "<u4").astype(np.int64).reshape(-1, 3)


variants, vbytes, ibytes = [], [], []
vbase = ioff = 0
for name, T, prims in nodes:
    pos = np.concatenate([p["pos"] for p in prims])
    nrm = np.concatenate([p["nrm"] for p in prims])
    uv = np.concatenate([p["uv"] for p in prims])
    idx, off = [], 0
    for p in prims:
        idx.append(p["idx"] + off)
        off += len(p["pos"])
    idx = np.concatenate(idx)
    # Re-center on the base: nodes are laid out side by side in the source file.
    pos = pos - np.array([T[0], 0, T[2]], np.float32)
    pos *= cfg.get("scale", 1.0)
    tris = simplify(pos, idx, cfg["target_tris"])
    used, inv = np.unique(tris, return_inverse=True)
    pos, nrm, uv = pos[used], nrm[used], uv[used]
    tris = inv.reshape(-1)
    height = float(pos[:, 1].max())
    radius = float(np.linalg.norm(pos[:, [0, 2]], axis=1).max())
    hN = np.clip(pos[:, 1] / max(height, 1e-3), 0, 1)
    ex = np.stack([hN, np.full_like(hN, rng.random()), 0.45 + 0.55 * hN, np.ones_like(hN)], 1)
    v = np.zeros(len(pos), dtype=[("p", "<f2", 4), ("n", "i1", 4), ("t", "<f2", 2), ("e", "u1", 4)])
    v["p"][:, :3] = pos
    v["n"][:, :3] = np.clip(np.round(nrm * 127), -127, 127)
    v["t"] = uv
    v["e"] = np.clip(np.round(ex * 255), 0, 255)
    vbytes.append(v.tobytes())
    ibytes.append((tris + vbase).astype("<u4").tobytes())
    variants.append({"name": name, "firstIndex": ioff, "indexCount": len(tris), "height": round(height, 3), "radius": round(radius, 3)})
    print(f"[plants] {cfg['name']} {name}: tris {len(idx)} -> {len(tris) // 3}, h {height:.2f} r {radius:.2f}")
    vbase += len(pos)
    ioff += len(tris)

vb, ib = b"".join(vbytes), b"".join(ibytes)
with open(os.path.join(OUT, "plants.bin"), "wb") as f:
    f.write(vb)
    f.write(ib)

src_dir = os.path.dirname(cfg["src"])
diff = img_np(os.path.join(src_dir, gltf["images"][gltf["textures"][gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]["uri"]))
diff[..., 3] = img_np(cfg["alpha"])[..., 0]
save_png(bleed(diff), os.path.join(OUT, "albedo.png"))
nor_uri = gltf["images"][gltf["textures"][gltf["materials"][0]["normalTexture"]["index"]]["source"]]["uri"]
shutil.copy(os.path.join(src_dir, nor_uri), os.path.join(OUT, "normal.jpg"))
json.dump({
    "name": cfg["name"],
    "source": cfg["credit"],
    "vertexBytes": len(vb),
    "indexCount": ioff,
    "variants": variants,
    "bloomColor": cfg["bloom_color"],
}, open(os.path.join(OUT, "plants.json"), "w"), indent=2)
print(f"[plants] {cfg['name']} done: {vbase} verts, {ioff // 3} tris, {len(vb) + len(ib)} bytes")
