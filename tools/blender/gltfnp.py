"""Minimal numpy glTF reader: primitives as (positions, normals, uvs, indices, material name)."""
import json, os
import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path):
    g = json.load(open(path))
    base = os.path.dirname(path)
    bufs = [np.fromfile(os.path.join(base, b["uri"]), dtype=np.uint8) for b in g["buffers"]]

    def acc(i):
        a = g["accessors"][i]
        bv = g["bufferViews"][a["bufferView"]]
        dt = np.dtype(COMP[a["componentType"]])
        n = NCOMP[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or dt.itemsize * n
        raw = bufs[bv["buffer"]]
        if stride == dt.itemsize * n:
            arr = np.frombuffer(raw.data, dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n)
        else:
            arr = np.lib.stride_tricks.as_strided(
                np.frombuffer(raw.data, dtype=dt, offset=off), shape=(a["count"], n), strides=(stride, dt.itemsize))
        return np.array(arr)

    prims = []
    for m in g["meshes"]:
        for p in m["primitives"]:
            at = p["attributes"]
            prims.append({
                "material": g["materials"][p["material"]]["name"],
                "pos": acc(at["POSITION"]).astype(np.float32),
                "nrm": acc(at["NORMAL"]).astype(np.float32) if "NORMAL" in at else None,
                "uv": acc(at["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in at else None,
                "idx": acc(p["indices"]).reshape(-1, 3).astype(np.int64),
            })
    return g, prims


def components(nverts, tris):
    """Connected components over shared vertex indices (label propagation + pointer jumping)."""
    lab = np.arange(nverts, dtype=np.int64)
    while True:
        m = lab[tris].min(axis=1)
        new = lab.copy()
        for k in range(3):
            np.minimum.at(new, tris[:, k], m)
        np.minimum.at(new, new, new)  # no-op guard
        new = new[new]
        while True:
            nn = new[new]
            if np.array_equal(nn, new):
                break
            new = nn
        if np.array_equal(new, lab):
            return lab
        lab = new


def weld(pos, decimals=5):
    """Map each vertex to a representative with identical position (splits from UV seams)."""
    q = np.round(pos, decimals)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.reshape(-1)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def load_nodes(path):
    """Top-level mesh nodes with their TRS applied: list of (node name, [prims])."""
    g, prims = load(path)
    # Map mesh index -> its primitives in load() order.
    by_mesh, k = [], 0
    for m in g["meshes"]:
        by_mesh.append(prims[k:k + len(m["primitives"])])
        k += len(m["primitives"])
    out = []
    for node in g["nodes"]:
        if "mesh" not in node:
            continue
        R = quat_to_mat(node.get("rotation", [0, 0, 0, 1]))
        S = np.array(node.get("scale", [1, 1, 1]), np.float64)
        T = np.array(node.get("translation", [0, 0, 0]), np.float64)
        M = R * S
        N = R / S  # inverse-transpose for normals (R orthonormal)
        ps = []
        for p in by_mesh[node["mesh"]]:
            q = dict(p)
            q["pos"] = (p["pos"] @ M.T + T).astype(np.float32)
            if p["nrm"] is not None:
                n = p["nrm"] @ N.T
                q["nrm"] = (n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
            ps.append(q)
        out.append((node.get("name", "node"), T, ps))
    return g, out
