"""
Models, rigs and animates the shore crab (Carcinus maenas) in Blender and bakes its animation
for the GPU (the birds' format; runtime in src/world/crabs.ts, src/shaders/crabs.wgsl):

  blender -b --factory-startup --python tools/blender/model_crab.py -- <out dir> \
      [--preview <dir>] [--clips a,b] [--blend <file.blend>]

An exoskeleton: every segment is rigid and follows exactly one bone (no skin bending), so the
carapace, the claws' arm / hand / finger and each walking leg's thigh / shin are separate
pieces jointed at the bones' heads. Blender axes: x right, y forward, z up; the crab walks
sideways (along x). Ground at z = 0.

Vertex parts: 0 carapace and legs (mottled shell), 1 claw fingers (dark, glossy), 2 eyes.
Clips: idle (feeding: the claws pick at the sand and lift to the mouth), walk and run
(sideways, alternating tetrapod gait), threat (claws raised and open), burrow (digging down
out of sight) and emerge.
"""
import math, os, sys
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(__file__))
from rigkit import *  # noqa: E402,F401,F403
import rigkit  # noqa: E402

V = Vector
reset()

# Proportions (m): a mature shore crab, carapace ~6.5 cm across.
CW, CD, CH = 0.034, 0.027, 0.011  # carapace half width, half depth, half height
BODY_Z = 0.019
LEG_Y = [0.009, -0.001, -0.010, -0.018]  # leg attachment, front to back
LEG_LEN = [(0.019, 0.024), (0.021, 0.026), (0.02, 0.025), (0.017, 0.022)]

# ---------------------------------------------------------------------------------------
# Mesh

skin = Skin(res=0.0012, scale=10)
# Carapace: a broad, flattened hexagon-ish dome, widest at the front third; the front edge
# carries five teeth each side (the shore crab's serrated margin).
# Wide across the front, narrowing to the back: a trapezoid with rounded corners.
skin.ball((0, 0.004, BODY_Z), CW * 0.92, (1.0, 0.62, 0.3), stiff=2.5)
skin.ball((0, -0.01, BODY_Z - 0.001), CW * 0.62, (1.0, 0.75, 0.38), stiff=2.2)
skin.ball((0, 0.0, BODY_Z + 0.003), CW * 0.6, (1.0, 0.8, 0.32), stiff=2.0)
for s in (-1, 1):
    for k in range(5):
        a = math.radians(30 + k * 13)
        skin.ball((s * math.sin(a) * CW * 0.95, 0.004 + math.cos(a) * CD * 0.72, BODY_Z + 0.0005), 0.0048, (1.1, 0.7, 0.42), stiff=3.5)
    # The brow lobes between the eyes.
    skin.ball((s * 0.004, CD * 0.86, BODY_Z + 0.001), 0.004, (1, 0.8, 0.5), stiff=3)
# Underside: the sternum and the folded abdomen.
skin.ball((0, -0.004, BODY_Z - 0.006), CW * 0.62, (1.0, 0.8, 0.35), stiff=2.0)
body = skin.build(voxel=0.0011, ratio=0.35, smooth=6)

body["bone"] = "body"
obs = [body]
FORCE = []


def piece(ob, bone):
    ob["bone"] = bone
    obs.append(ob)
    return ob


def seg(a, b, r0, r1, part=0, segs=8, name="seg"):
    """A tapered leg segment (slightly flattened, like a crab's)."""
    a, b = V(a), V(b)
    pts = [a.lerp(b, t / 4) for t in range(5)]
    radii = [r0 + (r1 - r0) * t / 4 for t in range(5)]
    return tube(pts, radii, part, segs=segs, name=name)


rig = Rig()
rig.bone("body", (0, -CD, BODY_Z), (0, CD, BODY_Z), radius=0.03)
# Eyes on short stalks at the front notch.
rig.bone("eyes", (0, CD * 0.8, BODY_Z + 0.004), (0, CD * 1.05, BODY_Z + 0.006), "body", radius=0.006)
for s in (-1, 1):
    base = V((s * 0.005, CD * 0.92, BODY_Z + 0.005))
    tip = base + V((s * 0.002, 0.003, 0.003))
    piece(seg(base, tip, 0.0011, 0.001, 0, 6, "stalk"), "eyes")
    piece(sphere(tip, 0.0017, 2, segs=(8, 6)), "eyes")

# Walking legs: thigh (merus) rising from the side to a raised knee, shin (carpus + propodus
# + dactyl) bending down to a pointed tip on the ground.
for i, (y, (l1, l2)) in enumerate(zip(LEG_Y, LEG_LEN)):
    root = V((CW * (0.86 - 0.08 * i), y, BODY_Z - 0.003))
    splay = math.radians(-12 + i * 14)  # front legs angle forward, back legs back
    out = V((math.cos(splay), -math.sin(splay) * 0.8, 0)).normalized()
    knee = root + out * l1 * 0.92 + V((0, 0, l1 * 0.38))
    tip = knee + out * l2 * 0.55 + V((0, 0, -(knee.z - 0.0005)))
    mid = knee.lerp(tip, 0.45) + out * 0.004 + V((0, 0, 0.003))
    rig.bone(f"thigh{i}.R", root, knee, "body", radius=0.006)
    rig.bone(f"shin{i}.R", knee, tip, f"thigh{i}.R", radius=0.005)
    for sgn in (1, -1):
        m = (lambda p: V((p.x * sgn, p.y, p.z)))
        side = "R" if sgn > 0 else "L"
        piece(seg(m(root), m(knee), 0.0037, 0.003, 0, 8, "thigh"), f"thigh{i}.{side}")
        piece(seg(m(knee), m(mid), 0.0029, 0.0024, 0, 8, "shin"), f"shin{i}.{side}")
        piece(seg(m(mid), m(tip), 0.0022, 0.0004, 0, 7, "dactyl"), f"shin{i}.{side}")

# Claws (chelipeds): arm from the front corner, a swollen hand, the movable finger.
# Folded in front of the face: arm out and forward, forearm turning in, hand across the front.
shoulder = V((CW * 0.58, CD * 0.7, BODY_Z - 0.003))
elbow = shoulder + V((0.011, 0.009, 0.0))
wrist = elbow + V((-0.003, 0.010, 0.0))
palm_end = wrist + V((-0.015, 0.007, -0.002))
rig.bone("arm.R", shoulder, elbow, "body", radius=0.007)
rig.bone("fore.R", elbow, wrist, "arm.R", radius=0.007)
rig.bone("hand.R", wrist, palm_end, "fore.R", radius=0.008)
finger_base = wrist + V((-0.011, 0.006, 0.003))
finger_tip = finger_base + V((-0.011, 0.004, -0.003))
rig.bone("finger.R", finger_base, finger_tip, "hand.R", radius=0.004)
for sgn in (1, -1):
    m = (lambda p: V((p.x * sgn, p.y, p.z)))
    side = "R" if sgn > 0 else "L"
    piece(seg(m(shoulder), m(elbow), 0.0042, 0.0046, 0, 8, "arm"), f"arm.{side}")
    piece(seg(m(elbow), m(wrist), 0.0048, 0.0052, 0, 8, "fore"), f"fore.{side}")
    # The hand: a swollen, flattened palm narrowing into the fixed finger.
    hand = sphere(m(wrist.lerp(palm_end, 0.4)), 0.0078, 0, scale=(1.35, 0.78, 0.82), segs=(14, 10))
    piece(hand, f"hand.{side}")
    fixed_a = m(wrist.lerp(palm_end, 0.8))
    piece(seg(fixed_a, m(palm_end + V((-0.009, 0.002, -0.001))), 0.0032, 0.0006, 1, 7, "fixed"), f"hand.{side}")
    piece(seg(m(finger_base), m(finger_tip), 0.0024, 0.0005, 1, 7, "finger"), f"finger.{side}")

# ---------------------------------------------------------------------------------------
# Paint: olive-green mottled carapace, yellow-orange underneath and at the joints, dark
# claw tips; a per-vertex AO from height and facing.


def paint(pos, nrm, part, fs, fc, up):
    n = len(pos)
    mott = fine_noise(pos, 260.0) * 0.5 + 0.5
    big = fine_noise(pos, 90.0) * 0.5 + 0.5
    top = np.clip(nrm[:, 2] * 0.8 + 0.5, 0, 1)
    olive = np.array((0.44, 0.46, 0.23))
    dark = np.array((0.22, 0.25, 0.11))
    pale = np.array((0.66, 0.6, 0.38))
    belly = np.array((0.86, 0.66, 0.3))
    col = olive[None] * (1 - mott[:, None] * 0.6) + dark[None] * mott[:, None] * 0.6
    col = col * (0.85 + 0.3 * big[:, None])
    spots = np.clip((fine_noise(pos, 520.0) - 0.35) * 4, 0, 1)
    col = col * (1 - spots[:, None] * 0.5) + pale[None] * spots[:, None] * 0.5
    col = col * top[:, None] + belly[None] * (1 - top[:, None])
    # Joints and leg tips warm toward orange.
    low = np.clip(1 - pos[:, 2] / 0.006, 0, 1)
    col = col * (1 - 0.5 * low[:, None]) + np.array((0.75, 0.5, 0.22))[None] * 0.5 * low[:, None]
    col[part == 1] = np.array((0.1, 0.07, 0.05))
    col[part == 2] = np.array((0.04, 0.035, 0.03))
    ao = np.clip(0.55 + 0.45 * top * np.clip(pos[:, 2] / BODY_Z, 0.3, 1.2), 0, 1)
    gloss = np.where(part == 0, 0.55, 0.9)
    return np.clip(col, 0, 1), ao, gloss


# ---------------------------------------------------------------------------------------
# Animation. Leg poses: thigh (rx lifts the knee up/down about the leg's long axis frame,
# rz swings it forward/back), shin (rx opens/closes the knee). Signs were tuned against the
# preview contact sheets.

LEGS = range(4)
# Alternating tetrapod: legs 0, 2 of one side step with 1, 3 of the other.
PHASE = {("R", 0): 0.0, ("R", 2): 0.0, ("L", 1): 0.0, ("L", 3): 0.0, ("R", 1): 0.5, ("R", 3): 0.5, ("L", 0): 0.5, ("L", 2): 0.5}


def leg_angles(ph, reach_amp, lift_amp, leading):
    """One leg over its cycle: stance (0..0.55) the foot pulls in (leading side) or pushes
    out (trailing side); swing (0.55..1) it lifts and returns."""
    ph = ph % 1.0
    if ph < 0.55:
        t = ph / 0.55
        reach = 1 - 2 * t
        lift = 0.0
    else:
        t = (ph - 0.55) / 0.45
        reach = -1 + 2 * (0.5 - 0.5 * math.cos(math.pi * t))
        lift = math.sin(math.pi * t)
    if not leading:
        reach = -reach
    knee = reach * reach_amp
    return (-knee * 0.45 + lift * lift_amp, 0.0, 0.0), (knee, 0.0, 0.0)


def gait_pose(ph, reach_amp, lift_amp, claw_tuck, bob):
    p = {}
    for (side, i), off in PHASE.items():
        th, sh = leg_angles(ph + off + i * 0.04, reach_amp, lift_amp, side == "R")
        if side == "L":
            # apply_pose mirrors left (rx, -ry, -rz): rx is shared.
            pass
        p[f"thigh{i}.{side}"] = th
        p[f"shin{i}.{side}"] = sh
    p["arm"] = (claw_tuck, 0, 0)
    p["fore"] = (0, 0, -claw_tuck * 0.5)
    p["loc"] = (0, 0, bob * abs(math.sin(2 * math.pi * ph * 2)))
    return p


def feeding(t):
    """Idle: the claws take turns to pinch at the sand and lift a morsel to the mouth."""
    p = {}
    for side, off in (("R", 0.0), ("L", 0.5)):
        u = (t + off) % 1.0
        reach = max(0.0, math.sin(2 * math.pi * u))
        p[f"arm.{side}"] = (-18 * reach, 0, 0)
        p[f"fore.{side}"] = (0, 0, 22 * reach * (1 if side == "R" else -1) * 0)
        p[f"hand.{side}"] = (-25 * reach, 0, 0)
        p[f"finger.{side}"] = (-18 * (1 - reach) * (0.5 + 0.5 * math.sin(2 * math.pi * u * 3)), 0, 0)
    for i in LEGS:
        w = 2.0 * math.sin(2 * math.pi * (t * 2 + i * 0.3))
        p[f"thigh{i}"] = (w, 0, 0)
    p["eyes"] = (6 * math.sin(2 * math.pi * t * 2), 0, 0)
    p["loc"] = (0, 0, 0.0004 * math.sin(2 * math.pi * t * 2))
    return p


FPS = 30
clips = {
    "idle": (FPS, True, cycle(feeding, 2.4, 48)),
    "walk": (FPS, True, cycle(lambda u: gait_pose(u, 16, 22, -10, 0.0008), 0.42, 14)),
    "run": (FPS, True, cycle(lambda u: gait_pose(u, 22, 28, -25, 0.0012), 0.26, 10)),
    "threat": (FPS, True, cycle(lambda u: {
        "arm": (35, 0, 15), "fore": (0, 0, 10), "hand": (20, 0, 0), "finger": (-30 + 10 * math.sin(2 * math.pi * u), 0, 0),
        "thigh0": (-6, 0, 0), "thigh1": (-6, 0, 0), "thigh2": (4, 0, 0), "thigh3": (8, 0, 0),
        "loc": (0, 0, 0.004 + 0.0005 * math.sin(2 * math.pi * u)), "eyes": (8, 0, 0)}, 1.0, 12)),
    # Burrowing: legs scrabble while the body works down out of sight (one-shot).
    "burrow": (FPS, False, [(t / 30, {**gait_pose(t / 30 * 4, 14, 18, -30, 0.0), "loc": (0, 0, -0.03 * min(1, t / 30) ** 1.4)}) for t in range(31)]),
    "emerge": (FPS, False, [(t / 24, {**gait_pose(t / 24 * 3, 10, 14, -30, 0.0), "loc": (0, 0, -0.03 * (1 - min(1, t / 24)) ** 1.2)}) for t in range(25)]),
}

allowed = {0: None, 1: None, 2: None}
finish("crab", obs, rig, allowed, 0.002, paint, clips, {"size": CW * 2, "stride": 0.03}, size=0.08, look=(0, 0, 0.015), views=("front", "top"), wing_like=tuple(f"{b}{i}" for b in ("thigh", "shin") for i in LEGS) + ("arm", "fore", "hand", "finger"))
