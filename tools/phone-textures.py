"""
Phone-sized copies of every texture larger than 512 px (longest side 512; tree impostor
atlases 1024 so their frames stay readable), saved next to the original as <name>.m.<ext>,
and public/assets/phone-textures.json listing them. Phones load these instead: decoding full
size images on a phone is what runs it out of memory while loading.

  python3 tools/phone-textures.py
"""
import glob, json, os
from PIL import Image

root = "public/assets"
listed = []
for f in sorted(glob.glob(f"{root}/**/*.png", recursive=True) + glob.glob(f"{root}/**/*.jpg", recursive=True)):
    if ".m." in f:
        continue
    im = Image.open(f)
    cap = 1024 if "impostor" in f else 512
    if max(im.size) <= cap:
        continue
    k = cap / max(im.size)
    size = (round(im.width * k), round(im.height * k))
    # Each channel on its own: resizing RGBA together premultiplies, which blackens the colour
    # bled into transparent texels that alpha-tested foliage relies on at its edges.
    small = Image.merge(im.mode, [b.resize(size, Image.LANCZOS) for b in im.split()]) if im.mode in ("RGBA", "RGB") else im.resize(size, Image.LANCZOS)
    base, ext = os.path.splitext(f)
    out = f"{base}.m{ext}"
    if ext == ".jpg":
        small.convert("RGB").save(out, quality=88)
    else:
        small.save(out, optimize=True)
    listed.append(os.path.relpath(f, "public"))
json.dump(sorted(listed), open(f"{root}/phone-textures.json", "w"), indent=0)
print(len(listed), "phone textures")
