import sys, numpy as np
sys.path.insert(0, "tools/blender")
from gltfnp import load, components, weld
g, prims = load(sys.argv[1])
for p in prims:
    if "leaves" not in p["material"]:
        continue
    w = weld(p["pos"])
    tris = w[p["idx"]]
    lab = components(w.max() + 1, tris)
    tl = lab[tris[:, 0]]
    ids, counts = np.unique(tl, return_counts=True)
    print(p["material"], "tris", len(tris), "components", len(ids))
    print("  tris/component percentiles 5/50/95/max:", np.percentile(counts, [5, 50, 95]).astype(int), counts.max())
    # Extent of the median-size components
    order = np.argsort(counts)
    for pick in [order[len(order) // 2], order[-1]]:
        sel = p["idx"][tl == ids[pick]].reshape(-1)
        P = p["pos"][sel]; U = p["uv"][sel]
        print("  comp tris", counts[pick], "extent m", np.round(P.max(0) - P.min(0), 3), "uv bbox", np.round(U.min(0), 3), np.round(U.max(0), 3))
