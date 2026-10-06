# Render a quick preview of a glTF in Blender. Usage:
# blender -b --factory-startup --python tools/blender/preview.py -- <in.gltf|glb> <out.png> [azimuth_deg]
import bpy, sys, math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
az = math.radians(float(argv[2])) if len(argv) > 2 else math.radians(35)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
mins = Vector((1e9, 1e9, 1e9)); maxs = Vector((-1e9, -1e9, -1e9))
tris = 0
for o in objs:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        mins = Vector(map(min, mins, w)); maxs = Vector(map(max, maxs, w))
    tris += sum(len(p.vertices) - 2 for p in o.data.polygons)
center = (mins + maxs) / 2
size = (maxs - mins).length
print(f"PREVIEW bounds min={tuple(round(v,2) for v in mins)} max={tuple(round(v,2) for v in maxs)} tris={tris}")

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 900
scene.render.resolution_y = 900
scene.render.film_transparent = False
world = bpy.data.worlds.new("w"); scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.75, 0.82, 0.95, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.5
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
scene.collection.objects.link(sun)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.lens = 50
scene.collection.objects.link(cam); scene.camera = cam
d = size * 1.25
cam.location = center + Vector((math.sin(az) * d, -math.cos(az) * d, size * 0.15))
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
