"""
Web delivery copies of the assets, next to the originals (which stay the editable sources):
  <image>.webp   for every .png / .jpg (and the phone copies <name>.m.webp): about a third
                 of the bytes; colour under transparent texels is kept exact (foliage edges)
  <mesh>.bin.gz  every .bin, gzip-compressed (decompressed in the browser)
and public/assets/web.json listing them, which the loaders consult.

  python3 tools/phone-textures.py && python3 tools/web-assets.py
"""
import glob, gzip, json, os
from PIL import Image

root = "public/assets"
webp, gz = [], []
for f in sorted(glob.glob(f"{root}/**/*.png", recursive=True) + glob.glob(f"{root}/**/*.jpg", recursive=True)):
    out = os.path.splitext(f)[0] + ".webp"
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(f):
        im = Image.open(f)
        normal = "nor" in os.path.basename(f) or "normal" in os.path.basename(f)
        im.save(out, "WEBP", quality=92 if normal else 88, method=6, exact=True, alpha_quality=100)
    webp.append(os.path.relpath(f, "public"))
for f in sorted(glob.glob(f"{root}/**/*.bin", recursive=True)):
    out = f + ".gz"
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(f):
        with open(f, "rb") as src, gzip.open(out, "wb", compresslevel=9) as dst:
            dst.write(src.read())
    gz.append(os.path.relpath(f, "public"))
json.dump({"webp": webp, "gz": gz}, open(f"{root}/web.json", "w"), indent=0)
print(len(webp), "webp,", len(gz), "gz")
