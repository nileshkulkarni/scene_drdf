import math
import os
import pdb
import pickle
import random
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

print("Starting")

print(sys.argv)

index = 5
for i in range(2, len(sys.argv)):
    if sys.argv[i] == "--":
        index = i + 1
        break

# pdb.set_trace()
modelpath = sys.argv[index]
index += 1
pngpath = sys.argv[index]
index += 1
matfile = sys.argv[index]

w = 800
h = 800
# modelpath = sys.argv[5]
# pngpath = sys.argv[6]
# az = sys.argv[7]
# el = sys.argv[8]
# distScale = sys.argv[9]
# upsampFactor = sys.argv[10]
# theta = sys.argv[11]
# if len(sys.argv) > 12:
#     focal_length = int(sys.argv[12])
# else:
#     focal_length = 37

print("Read Args")


def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)


def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx * cx + ty * cy, -1), 1)
    # roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    print(f"{yaw:f} {pitch:f} {roll:f}")
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)


def camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)


def quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return (q1, q2, q3, q4)


def add_camera(
    xyz=(0, 0, 0),
    rot_vec_rad=(0, 0, 0),
    name=None,
    proj_model="PERSP",
    f=35,
    sensor_fit="HORIZONTAL",
    sensor_width=32,
    sensor_height=18,
):
    # bpy.ops.object.camera_add()
    # cam = bpy.context.active_object
    cam = bpy.data.objects["Camera"]
    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height

    return cam


import bpy

# def enable_gpus():
#     import bpy
#     scene = bpy.context.scene
#     scene.cycles.device = 'GPU'

#     prefs = bpy.context.preferences
#     cprefs = prefs.addons['cycles'].preferences

#     # Attempt to set GPU device types if available
#     for compute_device_type in ('CUDA',):
#         try:
#             cprefs.compute_device_type = compute_device_type
#             break
#         except TypeError:
#             pass

#     # Enable all CPU and GPU devices
#     for device in cprefs.devices:
#         device.use = True

# enable_gpus()


def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = dist * math.cos(theta) * math.cos(phi)
    y = dist * math.sin(theta) * math.cos(phi)
    z = dist * math.sin(phi)
    return (x, y, z)


print("Importing")
print(modelpath)
# for obj in bpy.data.objects:
#     obj.select = True
#     bpy.ops.object.delete()

if modelpath.endswith(".obj"):
    bpy.ops.import_scene.obj(filepath=modelpath, axis_forward="Y", axis_up="Z")
    # bpy.ops.import_scene.obj(filepath=modelpath, )
    # bpy.ops.import_scene.obj(filepath=modelpath,)
elif modelpath.endswith(".ply"):
    bpy.ops.import_mesh.ply(filepath=modelpath)

    pcl_key = None
    for obj_key in bpy.data.objects.keys():
        if (
            ("Camera" not in obj_key)
            and ("Lamp" not in obj_key)
            and ("Point" not in obj_key)
        ):
            pcl_key = obj_key
    material = bpy.data.materials.new(name="Material_pcl")
    material.use_vertex_color_paint = True
    bpy.context.scene.objects.active = bpy.data.objects[pcl_key]
    object_pR_cv2bcamcl = bpy.context.active_object
    if object_pcl.data.materials:
        object_pcl.data.materials[0] = material
    else:
        object_pcl.data.materials.append(material)
    object_pcl.select = False
    # obj = bpy.data.objects[pcl_key]
else:
    modelsAll = [x for x in os.listdir(modelpath) if x.endswith(".obj")]
    for model in modelsAll:
        bpy.ops.import_scene.obj(filepath=os.path.join(modelpath, model))

###### Camera settings ######
# import pdb; pdb.set_trace()
# print(bpy.data.objects['points_mesh'].bound_box)
# print(bpy.data.objects.keys())
# bbox = np.array(bpy.data.objects['points_mesh'].bound_box)

# bbox_center = np.mean(bbox, 0)

# distance = bbox_center * 2

import scipy.io as sio

# enable_gpus(device_type='CUDA')
data = sio.loadmat(matfile)
RT = data["RT"]
intrinsic = data["intrinsic"]
img_w = data["img_size"][0][1]
img_h = data["img_size"][0][0]
focal_length_px = intrinsic[0, 0]
focal_mm = focal_length_px * 32 / img_w

print(img_w)
print(RT)
camera = add_camera(
    (0, 0, 0),
    (0, np.pi, 0),
    "camera",
    "PERSP",
    f=focal_mm,
    sensor_fit="HORIZONTAL",
    sensor_width=32,
)

# camera = add_camera((distance[0], distance[1], distance[2]), (0, np.pi, 0), 'camera', 'PERSP',
#                         focal_length, 'HORIZONTAL', 32)

R_cv2bcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
camera.data.clip_end = 10

# R = RT[0:3, 0:3]
# T = RT[0:3, 3]

# R = np.matmul(R_cv2bcam, R)
# T = np.matmul(R_cv2bcam, T)
# RT = np.matmul(R_cv2bcam, RT)
# RT = np.eye(4)
RT = Matrix(RT)
camera.matrix_world = RT

# camera.data.lens = (intrinsic[0, 0] + intrinsic[1, 1]) / 2

# camObj = bpy.data.objects['Camera']
# rho = np.linalg.norm(camObj.location)
# rho *= float(distScale)
# #print('Rho = ' + str(rho) + '\n')
# cx, cy, cz = obj_centened_camera_pos(rho, az, el)
# q1 = camPosToQuaternion(cx, cy, cz)
# q2 = camRotQuaternion(cx, cy, cz, float(theta))
# q = quaternionProduct(q2, q1)
# camObj.location[0] = cx
# camObj.location[1] = cy
# camObj.location[2] = cz
# # pdb.set_trace()
# camObj.location[0] -= bbox_center[0]
# camObj.location[1] -= bbox_center[1]
# camObj.location[2] -= bbox_center[2]
# camObj.rotation_mode = 'QUATERNION'
# camObj.rotation_quaternion[0] = q[0]
# camObj.rotation_quaternion[1] = q[1]
# camObj.rotation_quaternion[2] = q[2]
# camObj.rotation_quaternion[3] = q[3]

bpy.data.scenes["Scene"].render.filepath = pngpath
scene = bpy.context.scene
bpy.context.scene.cycles.device = "GPU"
bpy.context.scene.cycles.feature_set = "SUPPORTED"
# scene.cycles.device='GPU'
## Lighting ##
# clear default lights
# bpy.ops.object.select_by_type(type='LAMP')
# bpy.ops.object.delete(use_global=False)

# set environment lighting
# bpy.context.space_data.context = 'WORLD'
bpy.context.scene.world.light_settings.use_environment_light = True
bpy.context.scene.world.light_settings.environment_energy = 0.5
bpy.context.scene.world.light_settings.environment_color = "PLAIN"

# print(scene.render.resolution_x)
# print(scene.render.resolution_y)
scene.render.resolution_percentage = 100

scene.render.resolution_x = img_w
scene.render.resolution_y = img_h

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.render.image_settings.color_depth = "8"

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new("CompositorNodeRLayers")

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = "Depth Output"

if True:
    depth_file_output.format.file_format = "OPEN_EXR"
    links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
else:
    # Remap as other types can not represent the full range of depth.
    map = tree.nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [-0.7]
    map.size = [0.05]
    map.use_min = True
    map.min = [0]
    map.max = [255]
    map.use_max = True
    links.new(render_layers.outputs["Depth"], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = "MULTIPLY"
# scale_normal.use_alpha = True
scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs["Normal"], scale_normal.inputs[1])

bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = "ADD"
# bias_normal.use_alpha = True
bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])

import os.path as osp

basepath = osp.dirname(pngpath)
basename = osp.basename(pngpath)

normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = "Normal Output"
links.new(bias_normal.outputs[0], normal_file_output.inputs[0])
depth_file_output.base_path = basepath
normal_file_output.base_path = basepath
depth_file_output.file_slots[0].path = "depth"
normal_file_output.file_slots[0].path = "normal"
# albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
# albedo_file_output.label = 'Albedo Output'
# links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])

bpy.context.scene.use_nodes = True
bpy.ops.render.render(write_still=True)
