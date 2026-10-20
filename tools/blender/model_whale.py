"""
Models, rigs and animates a humpback whale in Blender and bakes its animation for the GPU
(the birds' format; runtime in src/world/whales.ts, src/shaders/whales.wgsl):

  blender -b --factory-startup --python tools/blender/model_whale.py -- <out dir> \
      [--preview <dir>] [--clips a,b] [--blend <file.blend>]

Blender axes: x right, y forward (toward the head), z up; the origin is at the waterline when
the whale lies resting at the surface (its back and blowhole just out of the water).

Body: one smooth metaball skin, ~13 m: a broad head studded with tubercles (the knobs of a
humpback's rostrum), a throat with ventral pleats (painted), a long tail stock. Pectoral fins
nearly a third of the body long, scalloped on the leading edge; a small dorsal hump; broad
flukes with a serrated trailing edge, black above and patterned white beneath.

Parts: 0 skin, 1 flukes (their underside pattern), 2 eyes, 3 pectoral fins (white beneath).
Clips: swim (slow up-and-down strokes), rest (logging at the surface, a slow breath),
dive (the terminal dive: back arches, the tail stock rises, the flukes lift clear of the
water and slip under), surface (rising up to rest).
"""
import math, os, sys
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(__file__))
from rigkit import *  # noqa: E402,F401,F403

V = Vector
reset()

L = 13.0  # body length
Z = -0.55  # body axis below the waterline when resting (the back just breaks the surface)

# ---------------------------------------------------------------------------------------
# Body: ellipsoids along the axis; radius profile of a humpback (stout, tapering to a narrow
# tail stock that flattens sideways).
skin = Skin(res=0.09)
# A metaball's surface lies at ~0.57 of its radius (threshold 0.6, stiffness 2): radii below
# are the visible half sizes.
_ball = skin.ball
skin.ball = lambda co, r, size=(1, 1, 1), stiff=2.0: _ball(co, r * 1.75, size, stiff)
# A humpback: a long, slender, flat-topped head (the rostrum awash when it rests), a deep
# throat, the back highest over the flippers, and a tail stock deeper than it is wide.
prof = [  # (y along the body, half width, half height, z offset)
    (6.4, 0.22, 0.16, -0.3), (5.9, 0.45, 0.3, -0.32), (5.2, 0.72, 0.5, -0.35), (4.3, 0.98, 0.75, -0.42),
    (3.2, 1.25, 1.05, -0.45), (2.0, 1.45, 1.25, -0.38), (0.6, 1.5, 1.3, -0.25), (-0.8, 1.38, 1.2, -0.12),
    (-2.2, 1.1, 1.02, -0.02), (-3.4, 0.78, 0.86, 0.04), (-4.5, 0.46, 0.7, 0.06), (-5.4, 0.26, 0.55, 0.06),
    (-6.1, 0.17, 0.4, 0.04), (-6.5, 0.13, 0.28, 0.02),
]
# Sampled every 0.4 m along a smooth spline through the profile (sparser ellipsoids leave
# beads along the tail stock).
dense = catmull(prof, 34)
for y, w, h, dz in dense:
    skin.ball((0, y, Z + dz), max(w, h), (w / max(w, h), 0.55, h / max(w, h)), stiff=2.0)
# The lower jaw and the deep, pleated throat beneath the head.
skin.ball((0, 4.2, Z - 1.05), 0.95, (1.0, 2.2, 0.5), stiff=2.0)
skin.ball((0, 2.6, Z - 1.25), 1.1, (1.0, 1.8, 0.5), stiff=2.0)
# Splashguard: the raised ridge in front of the blowhole.
skin.ball((0, 3.75, Z + 0.32), 0.22, (0.7, 1.2, 0.45), stiff=3)
# Dorsal fin on its hump, two thirds of the way back, then the knuckles of the tail stock.
skin.ball((0, -3.0, Z + 0.98), 0.3, (0.3, 1.6, 0.9), stiff=2.5)
skin.ball((0, -3.35, Z + 1.12), 0.16, (0.3, 1.0, 1.0), stiff=3)
for k in range(4):
    skin.ball((0, -3.9 - k * 0.5, Z + 0.78 - k * 0.11), 0.12, (0.4, 1.0, 0.8), stiff=3)
# Tubercles: knobs on top of the head and along the jaw.
rng = np.random.default_rng(4)
for k in range(34):
    y = 4.0 + rng.random() * 2.3
    side = rng.choice([-1, 1]) * rng.random() * 0.55 * (6.6 - y) / 2.6
    top = Z - 0.3 - 0.35 + (0.5 + (0.75 - 0.5) * (5.2 - y) / 0.9 if y < 5.2 else 0.5 - 0.35 * (y - 5.2) / 1.2)
    skin.ball((side, y, top + 0.02), 0.07, stiff=4)
body = skin.build(voxel=0.07, ratio=0.3, smooth=6)
body["bone"] = ""

obs = [body]

# Pectoral fins: long, narrow, curved back, knobbly leading edge (scalloped), white beneath.
def flipper(side):
    root = V((side * 1.25, 2.6, Z - 0.65))
    pts = []
    n = 14
    for i in range(n + 1):
        t = i / n
        sweep = V((side * 3.9 * t, -0.9 * t - 0.5 * t * t, -0.55 * t))
        chord = 0.95 * (1 - t) ** 0.7 + 0.12
        scal = 1 + 0.08 * math.sin(t * 40)
        pts.append((root + sweep, chord * scal))
    verts, faces = [], []
    for i, (p, c) in enumerate(pts):
        for k, (a, th) in enumerate(((0.0, 0.0), (0.35, 0.08), (1.0, 0.0), (0.35, -0.08))):
            verts.append(tuple(p + V((0, 0.5 * c - a * c, th * c))))
    for i in range(n):
        for k in range(4):
            a = i * 4 + k
            b = i * 4 + (k + 1) % 4
            f = (a, b, b + 4, a + 4)
            faces.append(f if side > 0 else f[::-1])
    ob = mesh_obj("pec", verts, faces, 3)
    ob["bone"] = "pec.R" if side > 0 else "pec.L"
    return ob


obs += [flipper(1), flipper(-1)]

# Flukes: a broad crescent, notched in the middle, serrated trailing edge.
def flukes():
    verts, faces, fs, fc, up = [], [], [], [], []
    span = 2.1
    n = 20
    for layer, sgn in enumerate((1, -1)):
        for i in range(n + 1):
            u = i / n * 2 - 1  # -1 left tip .. 1 right tip
            x = u * span
            lead = -6.55 - 0.55 * abs(u) ** 1.6
            trail = -7.35 - 0.25 * abs(u) + 0.35 * (1 - abs(u)) ** 6 + 0.06 * math.sin(abs(u) * 30) * (abs(u) > 0.15)
            tipround = math.sqrt(max(0.0, 1 - abs(u) ** 8))
            for j, v in enumerate((0.0, 0.5, 1.0)):
                y = lead + (trail - lead) * v * tipround
                th = 0.11 * (1 - abs(u)) * (1 - v) * math.sin(math.pi * (0.2 + v * 0.8))
                verts.append((x, y, Z + 0.0 + sgn * th))
                fs.append(abs(u))
                fc.append(v)
                up.append(1.0 if sgn > 0 else 0.0)
        o = layer * (n + 1) * 3
        for i in range(n):
            for j in range(2):
                a = o + i * 3 + j
                f = (a, a + 3, a + 4, a + 1)
                faces.append(f if sgn > 0 else f[::-1])
    # Close the leading and trailing edges.
    m = (n + 1) * 3
    for i in range(n):
        for j in (0, 2):
            a, b = i * 3 + j, (i + 1) * 3 + j
            q = (a, b, m + b, m + a)
            faces.append(q if j == 2 else q[::-1])
    ob = mesh_obj("flukes", verts, faces, 1, dict(fs=fs, fc=fc, up=up))
    ob["bone"] = "flukes"
    return ob


obs.append(flukes())
for s in (-1, 1):
    e = sphere((s * 0.98, 3.9, Z - 0.75), 0.07, 2, segs=(8, 6))
    e["bone"] = "head"
    obs.append(e)

# ---------------------------------------------------------------------------------------
# Rig: a spine from the head back to the flukes; flippers.

rig = Rig()
rig.bone("body", (0, -0.2, Z), (0, 1.8, Z), radius=1.8)
rig.bone("head", (0, 1.8, Z), (0, 5.9, Z), "body", radius=1.6)
rig.bone("tail1", (0, -0.2, Z), (0, -2.6, Z), "body", radius=1.4)
rig.bone("tail2", (0, -2.6, Z), (0, -4.6, Z), "tail1", radius=1.0)
rig.bone("tail3", (0, -4.6, Z), (0, -6.3, Z), "tail2", radius=0.6)
rig.bone("flukes", (0, -6.3, Z), (0, -7.4, Z), "tail3", radius=0.5)
rig.bone("pec.R", (1.25, 2.6, Z - 0.65), (5.0, 1.3, Z - 1.2), "body", radius=0.5)

# Bones run backward along the tail (y negative): rx + on them lifts the tail tip (checked
# against the preview sheets; head/body point forward, rx + raises the nose).


def paint(pos, nrm, part, fs, fc, up):
    n = len(pos)
    top = np.clip(nrm[:, 2] * 0.9 + 0.45, 0, 1)
    mott = fine_noise(pos, 0.9) * 0.5 + 0.5
    scars = np.clip((fine_noise(pos, 3.3) - 0.62) * 6, 0, 1)
    dark = np.array((0.16, 0.18, 0.22))
    belly = np.array((0.78, 0.78, 0.76))
    col = dark[None] * (0.85 + 0.3 * mott[:, None])
    # Pale throat and belly (patchy, as on North Atlantic humpbacks).
    pale = np.clip((0.15 - nrm[:, 2]) * 3, 0, 1) * np.clip((pos[:, 1] + 1.0) / 3.0, 0, 1) * np.clip(mott * 1.6 - 0.2, 0, 1)
    col = col * (1 - pale[:, None]) + belly[None] * pale[:, None]
    # Ventral pleats: pale grooves running back from the chin.
    pleat = (np.sin(pos[:, 0] * 22) > 0.6) & (nrm[:, 2] < -0.3) & (pos[:, 1] > 1.5)
    col[pleat] = col[pleat] * 0.6 + belly * 0.4
    # Barnacle scars and white scratches.
    col = col * (1 - scars[:, None] * 0.5) + np.array((0.7, 0.7, 0.68))[None] * scars[:, None] * 0.5
    # Pectorals: white beneath and toward the tips.
    fin = part == 3
    whiteFin = np.clip((0.3 - nrm[:, 2]) * 2, 0, 1)
    col[fin] = dark * (1 - whiteFin[fin, None]) + belly * whiteFin[fin, None]
    # Flukes: black above, beneath white with a black rim and blotches (each whale's own).
    fl = part == 1
    under = up < 0.5
    blot = np.clip((fine_noise(pos, 2.2) - 0.25) * 4, 0, 1)
    rim = (fc > 0.85) | (fs > 0.9)
    fcol = np.where((under & ~rim)[:, None], belly[None] * (1 - blot[:, None] * 0.85) + dark[None] * blot[:, None] * 0.85, dark[None])
    col[fl] = fcol[fl]
    col[part == 2] = (0.02, 0.02, 0.02)
    ao = np.clip(0.6 + 0.4 * top, 0, 1)
    gloss = np.where(part == 2, 1.0, 0.65)
    return np.clip(col, 0, 1), ao, gloss


# ---------------------------------------------------------------------------------------
# Animation

def stroke(u, amp=1.0):
    """One slow fluke beat (phase u): the wave runs back down the tail."""
    s = lambda d: math.sin(2 * math.pi * (u - d))
    return {
        "body": (1.5 * amp * s(0.0), 0, 0),
        "head": (-1.0 * amp * s(0.05), 0, 0),
        "tail1": (4 * amp * s(0.1), 0, 0),
        "tail2": (7 * amp * s(0.2), 0, 0),
        "tail3": (9 * amp * s(0.3), 0, 0),
        "flukes": (12 * amp * s(0.42), 0, 0),
        "pec": (0, 0, 4 * amp * s(0.25)),
        "loc": (0, 0, 0.12 * amp * s(0.15)),
    }


def rest(u):
    """Logging: lying still, breathing slowly, the tail hanging, fins idling."""
    b = math.sin(2 * math.pi * u)
    return {
        "body": (0.6 * b, 0, 0), "head": (-0.8 * b, 0, 0),
        "tail1": (3 + 1.5 * b, 0, 0), "tail2": (5 + 2 * b, 0, 0), "tail3": (4 + 2 * b, 0, 0), "flukes": (3 + 3 * b, 0, 0),
        "pec": (8 * b, 0, 3 + 4 * b), "loc": (0, 0, 0.06 * b),
    }


def dive_keys():
    """The terminal dive (12 s): a breath, the back arches high and rolls forward, the body tips
    nose-down about its middle so the tail stock rises and the flukes lift clear of the water
    (vertical, dripping), then the whole whale slides under. (Tail bones point backward: a
    negative rx raises them.)"""
    def ss(a, b, x):
        t = min(1.0, max(0.0, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    keys = []
    for i in range(61):
        t = i / 60 * 12.0
        u = t / 12.0
        arch = math.sin(math.pi * ss(0.0, 0.75, u))
        lift = ss(0.3, 0.6, u) * (1 - ss(0.82, 1.0, u))
        p = {
            "body": (-74 * ss(0.12, 0.66, u), 0, 0),
            "head": (-6 * arch, 0, 0),
            "tail1": (-10 * arch, 0, 0),
            "tail2": (-12 * arch, 0, 0),
            "tail3": (-8 * lift, 0, 0),
            # The flukes trail, then hang dripping as the tail rises (a slow flex).
            "flukes": (-18 * lift + 8 * math.sin(u * 9) * lift, 0, 0),
            "pec": (-25 * ss(0.1, 0.5, u), 0, -12 * ss(0.1, 0.5, u)),
            "loc": (0, 2.5 * u, 0.35 * arch - 1.0 * ss(0.0, 0.3, u) - 3.3 * ss(0.25, 0.7, u) - 9.0 * ss(0.72, 1.0, u)),
        }
        keys.append((t, p))
    return keys


def surface_keys():
    keys = []
    for i in range(25):
        t = i / 24 * 6.0
        u = t / 6.0
        e = 1 - (1 - u) ** 2
        p = rest(0.0)
        p.update({"body": (18 * (1 - e) - 6 * math.sin(math.pi * u), 0, 0), "tail2": (8 * (1 - e), 0, 0), "loc": (0, -2 * (1 - e), -7.0 * (1 - e) + 0.25 * math.sin(math.pi * u))})
        keys.append((t, p))
    return keys


def finwave_keys():
    """Rolling onto its side and raising a long white flipper, which it waves and slaps down
    on the water, twice (10 s)."""
    def ss(a, b, x):
        t = min(1.0, max(0.0, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    keys = []
    for i in range(41):
        t = i / 40 * 10.0
        u = t / 10.0
        roll = ss(0.0, 0.2, u) * (1 - ss(0.82, 1.0, u))
        slap = (math.sin(2 * math.pi * ss(0.25, 0.8, u) * 2 - math.pi / 2) * 0.5 + 0.5) * roll
        p = rest(u * 1.2)
        p.update({"body": (0, 65 * roll, 0), "pec.L": (0, 0, -70 * slap - 10 * roll), "pec.R": (0, 0, -6 * roll), "loc": (0, 0, -0.25 * roll)})
        keys.append((t, p))
    return keys


clips = {
    "finwave": (8, False, finwave_keys()),
    "swim": (12, True, cycle(lambda u: stroke(u), 5.0, 30)),
    "rest": (8, True, cycle(rest, 8.0, 32)),
    "dive": (8, False, dive_keys()),
    "surface": (8, False, surface_keys()),
}

allowed = {0: ["body", "head", "tail1", "tail2", "tail3"], 1: None, 2: None, 3: None}
finish("whale", obs, rig, allowed, 2.0, paint, clips, {"length": L}, size=9.0, look=(0, -1.0, -0.5), views=("side", "back"))
