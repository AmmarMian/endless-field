"""Contact sheet of Blender preview frames: python3 tools/dev/sheet.py <preview dir> <name> [cols]
Rows = <name>-<clip>-<view>-NN.png sequences (as written by model_birds.py --preview)."""
import sys, glob, re
from PIL import Image, ImageDraw
d, name = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 8
rows = {}
for f in sorted(glob.glob(f"{d}/{name}-*.png")):
    m = re.match(rf".*/{name}-(.+)-(\d\d)\.png", f)
    if m:
        rows.setdefault(m.group(1), []).append(f)
W, H = 216, 162
img = Image.new("RGB", (W * cols, H * max(1, len(rows))), "white")
dr = ImageDraw.Draw(img)
for r, (k, fs) in enumerate(sorted(rows.items())):
    for c, f in enumerate(fs[:cols]):
        img.paste(Image.open(f).convert("RGB").resize((W, H)), (c * W, r * H))
    dr.text((4, r * H + 4), k, fill="black")
out = f"{d}/sheet-{name}.png"
img.save(out)
print(out)
