import os
import os.path as osp
import pdb
import pickle as pkl
import shutil
import socket
import tempfile
from posixpath import dirname

import cv2
import imageio
import numpy as np
import skimage
import torch
from skimage.io import imread

from ..utils import transformations

curr_path = os.path.dirname(os.path.abspath(__file__))

cube_v = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)
cube_v = cube_v - 0.5

cube_f = np.array(
    [
        [1, 7, 5],
        [1, 3, 7],
        [1, 4, 3],
        [1, 2, 4],
        [3, 8, 7],
        [3, 4, 8],
        [5, 7, 8],
        [5, 8, 6],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 8],
        [2, 8, 4],
    ]
).astype(np.int)

if "arc-ts" in socket.gethostname():
    # if 'lh' in socket.gethostbyname():
    #     blender_exec = '/home/nileshk/SceneTSDF/libs/blender/blender-2.79/blender'
    # else:
    blender_exec = "/home/nileshk/SceneTSDF/libs/blender/blender-2.79/blender"
    renderFilePix3d = (
        "/home/nileshk/Research/mv_drdf/rgbd_drdf/renderer/renderPosePix3d.py"
    )
    renderFileMatterport = (
        "/home/nileshk/Research/mv_drdf/rgbd_drdf/renderer/renderPoseMatterport2.py"
    )
    renderFileShapenet = (
        "/home/nileshk/Research/mv_drdf/rgbd_drdf/renderer/renderPoseShapenet.py"
    )
    renderFileThreeDf = (
        "/home/nileshk/Research/mv_drdf/rgbd_drdf/renderer/renderPoseThreeDF.py"
    )
    renderNovelView = (
        "/home/nileshk/Research/mv_drdf/rgbd_drdf/renderer/render_novel_views.py"
    )
    renderNovelView = (
        "/home/nileshk/Research/mv_drdf/rgbd_drdf/renderer/render_novel_views2.py"
    )
    renderFileScannet = (
        "/home/nileshk/Research/mv_drdf/rgbd_drdf/renderer/renderPoseScannet.py"
    )
    renderFile = "/home/nileshk/Research/mv_drdf/rgbd_drdf/renderer/renderPose.py"
else:
    blender_exec = "/Pool3/users/nileshk/libs/blender/blender-2.79/blender"
    renderFilePix3d = "/Pool3/users/nileshk/SceneTSDF/csdf/renderer/renderPosePix3d.py"
    renderFileMatterport = (
        "/Pool3/users/nileshk/SceneTSDF/csdf/renderer/renderPoseMatterport2.py"
    )
    renderFileShapenet = (
        "/Pool3/users/nileshk/SceneTSDF/csdf/renderer/renderPoseShapenet.py"
    )
    renderFileThreeDf = (
        "/Pool3/users/nileshk/SceneTSDF/csdf/renderer/renderPoseThreeDF.py"
    )
    renderNovelView = (
        "/Pool3/users/nileshk/SceneTSDF/csdf/renderer/render_novel_views.py"
    )
    renderNovelView = (
        "/Pool3/users/nileshk/SceneTSDF/csdf/renderer/render_novel_views2.py"
    )
    renderFileScannet = (
        "/Pool3/users/nileshk/SceneTSDF/csdf/renderer/renderPoseScannet.py"
    )
    # renderFileScannet = '/Pool3/users/nileshk/SceneTSDF/csdf/renderer/renderPoseMatterport2.py'
    renderFile = "/Pool3/users/nileshk/SceneTSDF/csdf/renderer/renderPose.py"


def read_im(f):
    im = skimage.img_as_float(imread(f))
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=2)

    if im.shape[-1] == 4:
        alpha = np.expand_dims(im[..., 3], 2)
        im = im[..., :3] * alpha + (1 - alpha)
    return im[..., :3]


def voxels_to_mesh(pred_vol, thresh=0.5):
    v_counter = 0
    tot_points = np.greater(pred_vol, thresh).sum()
    v_all = np.tile(cube_v, [tot_points, 1])
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = (
        np.tile(np.linspace(0, 12 * tot_points - 1, 12 * tot_points), 3)
        .reshape(3, 12 * tot_points)
        .transpose()
    )
    f_offset = (f_offset // 12 * 8).astype(np.int)
    f_all += f_offset
    for x in range(pred_vol.shape[0]):
        for y in range(pred_vol.shape[1]):
            for z in range(pred_vol.shape[2]):
                if pred_vol[x, y, z] > thresh:
                    radius = pred_vol[x, y, z]
                    v_all[v_counter : v_counter + 8, :] *= radius
                    v_all[v_counter : v_counter + 8, :] += np.array([[x, y, z]]) + 0.5
                    v_counter += 8

    return v_all, f_all


def append_obj(mf_handle, vertices, faces):
    for vx in range(vertices.shape[0]):
        mf_handle.write(
            "v {:f} {:f} {:f}\n".format(
                vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]
            )
        )
    for fx in range(faces.shape[0]):
        mf_handle.write(f"f {faces[fx, 0]:d} {faces[fx, 1]:d} {faces[fx, 2]:d}\n")
    return


def render_vol(vol, thresh=0.5, az=60, el=20):
    render_dir = os.path.join(curr_path, "..", "cachedir", "rendering")
    mesh_file = os.path.join(render_dir, "vol_mesh.obj")
    png_file = os.path.join(render_dir, "vol_mesh.png")
    with open(mesh_file, "w") as mf:
        vs, fs = voxels_to_mesh(vol, thresh=thresh)
        vs = vs[:, [0, 2, 1]]
        vs = vs / vol.shape[0] - 0.5
        vs[:, 2] *= -1
        append_obj(mf, vs, fs)
    return render_mesh(mesh_file, png_file, az=az, el=el)


def render_points_pix3d(
    pts, edge_size=0.002, az=60, el=20, render_dir=None, file_name=None
):
    # render_dir = os.path.join(curr_path, '..', 'cachedir', 'rendering', exp_name)
    # if not osp.exists(render_dir):
    #     os.makedirs(render_dir)
    # mesh_file = os.path.join(render_dir, 'points_mesh.obj')
    # png_file = os.path.join(render_dir, 'points_mesh.png')
    # pdb.set_trace()
    render_dir = render_dir
    if render_dir is None:
        render_dir = os.path.join(curr_path, "..", "cachedir", "rendering", exp_name)
    if file_name is None:
        filename = "points_mesh"
    mesh_file = os.path.join(render_dir, f"{file_name}.obj")
    png_file = os.path.join(render_dir, f"{file_name}.png")

    with open(mesh_file, "w") as mf:
        vs, fs = points_to_cubes(pts, edge_size=edge_size)
        vs = vs[:, [0, 2, 1]]
        vs[:, 2] *= -1
        append_obj(mf, vs, fs)
    # pdb.set_trace()
    return render_mesh_pix3d(mesh_file, png_file, az=az, el=el)


def render_points(
    pts,
    edge_size=0.002,
    az=60,
    el=20,
    render_dir=None,
    filename=None,
):
    render_dir = render_dir
    if render_dir is None:
        render_dir = os.path.join(curr_path, "..", "cachedir", "rendering", exp_name)
    if file_name is None:
        filename = "points_mesh"

    mesh_file = os.path.join(render_dir, f"{filename}.obj")
    png_file = os.path.join(render_dir, f"{filename}.png")
    # pdb.set_trace()

    with open(mesh_file, "w") as mf:
        vs, fs = points_to_cubes(pts, edge_size=edge_size)
        vs = vs[:, [0, 2, 1]]
        vs[:, 2] *= -1
        append_obj(mf, vs, fs)
    # pdb.set_trace()
    return render_mesh(mesh_file, png_file, az=az, el=el)


def points_to_rects(points, normals, edge_size=0.05, thickness=0.01, use_colors=False):
    tot_points = len(points)
    # rect = cube_v * 1
    # rect[:, 2] = rect[:, 2] * thickness
    # rect[:, 0:2] = rect[:, 0:2] * edge_size
    z_axis = np.array([0, 0, 1])
    v_counter = 0
    tot_points = points.shape[0]
    v_all = np.tile(cube_v, [tot_points, 1])
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = (
        np.tile(np.linspace(0, 12 * tot_points - 1, 12 * tot_points), 3)
        .reshape(3, 12 * tot_points)
        .transpose()
    )
    f_offset = (f_offset // 12 * 8).astype(np.int)
    f_all += f_offset

    if type(edge_size) == float:
        edge_size = np.ones((1, len(points))) * edge_size
    for px in range(points.shape[0]):
        v_all[v_counter : v_counter + 8, 0:2] *= edge_size[0, px]
        v_all[v_counter : v_counter + 8, 2] *= thickness
        angle = transformations.angle_between_vectors(z_axis, normals[px], axis=0)
        vector = transformations.vector_product(z_axis, normals[px], axis=0)
        R = transformations.rotation_matrix(angle, vector)[0:3, 0:3]
        v_all[v_counter : v_counter + 8, :] = np.matmul(
            v_all[v_counter : v_counter + 8, :], R.T
        )
        v_all[v_counter : v_counter + 8, :] += points[px, :]
        v_counter += 8

    return v_all, f_all


def points_to_cubes(points, edge_size=0.05):
    """
    Converts an input point cloud to a set of cubes.

    Args:
        points: N X 3 array
        edge_size: cube edge size
    Returns:
        vs: vertices
        fs: faces
    """
    v_counter = 0
    tot_points = points.shape[0]
    v_all = np.tile(cube_v, [tot_points, 1])
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = (
        np.tile(np.linspace(0, 12 * tot_points - 1, 12 * tot_points), 3)
        .reshape(3, 12 * tot_points)
        .transpose()
    )
    f_offset = (f_offset // 12 * 8).astype(np.int)
    f_all += f_offset
    for px in range(points.shape[0]):
        v_all[v_counter : v_counter + 8, :] *= edge_size
        v_all[v_counter : v_counter + 8, :] += points[px, :]
        v_counter += 8

    return v_all, f_all


def render_mesh_pix3d(
    mesh_file,
    png_file,
    az=60,
    el=20,
    dist_scale=1.0,
    upsamp=1.0,
    theta=0,
    focal_length=37,
    img_width=400,
    img_height=800,
):
    blend_file = os.path.join(curr_path, "model.blend")
    # blender_exec = '/Pool3/users/nileshk/libs/blender/blender-2.71/blender'
    command = "bash {}/render_pix3d.sh {} {} {} {} {} {} {} {} {} {}".format(
        curr_path,
        blender_exec,
        blend_file,
        renderFilePix3d,
        mesh_file,
        png_file,
        az,
        el,
        dist_scale,
        upsamp,
        theta,
        focal_length,
        img_height,
        img_width,
    )
    # print(command)
    # pdb.set_trace()
    os.system(command)
    try:
        img = read_im(png_file)
        return img
        # return img[100:500, 265:665 :]
    except:
        return np.ones((400, 400, 3))


def normalize_depth(depth, md=12):
    max_depth = depth > md
    depth[max_depth] = md
    return depth


def render_novel_view2(
    mesh_file,
    intrinsic,
    RT=np.eye(4),
    img_width=400,
    img_height=800,
    max_depth=20,
    frustrum_path="",
    offset=np.array([0.0, 0.0, 0.0]),
    dataset="matterport",
    shading=False,
    depth_out=True,
    frame_step=5,
    start_frame=1,
    end_frame=100,
):

    blender_exec = "/Pool3/users/nileshk/libs/blender/blender-2.93/blender"

    dirpath = tempfile.mkdtemp()
    # dirpath = '.'
    mat_file = osp.join(dirpath, "pose.mat")
    import scipy.io as sio

    if intrinsic is None:
        intrinsic = np.eye(3) * 35

    novel_view_file = osp.join(dirpath, "novel_view.pkl")
    data = {
        "intrinsic": intrinsic,
        "RT": RT,
        "img_size": np.array([img_height, img_width]),
        "save_dir": str(dirpath),
        "frustrum_path": frustrum_path,
        "offset": offset,
        "dataset": dataset,
        "output_file": str(novel_view_file),
        "shading": shading,
        "depth_out": depth_out,
        "frame_step": frame_step,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    sio.savemat(mat_file, data)

    # blend_file = os.path.join(curr_path, 'model.blend')
    if dataset == "threedf":
        blend_file = os.path.join(curr_path, "novel_view_close_tdf.blend")
    elif dataset == "shapenet":
        blend_file = os.path.join(curr_path, "novel_view_shapenet.blend")
    else:
        blend_file = os.path.join(curr_path, "novel_view_close.blend")
    png_file = osp.join(dirpath, "image.png")

    # blender_exec = '/Pool3/users/nileshk/libs/blender/blender-2.71/blender'
    command = f"bash {curr_path}/render_matterport.sh {blender_exec} {blend_file} {renderNovelView} {mesh_file} {png_file} {mat_file}"

    os.system(command)

    if osp.exists(novel_view_file):
        with open(novel_view_file, "rb") as f:
            novel_views = pkl.load(f)
    else:
        novel_views = []
    shutil.rmtree(dirpath)
    return novel_views


def render_novel_view(
    mesh_file,
    intrinsic,
    RT=np.eye(4),
    img_width=400,
    img_height=800,
    max_depth=20,
    frustrum_path="",
    offset=np.array([0.0, 0.0, 0.0]),
    dataset="matterport",
):

    blender_exec = "/Pool3/users/nileshk/libs/blender/blender-2.93/blender"

    dirpath = tempfile.mkdtemp()
    # dirpath = '.'
    mat_file = osp.join(dirpath, "pose.mat")
    import scipy.io as sio

    if intrinsic is None:
        intrinsic = np.eye(3) * 35

    data = {
        "intrinsic": intrinsic,
        "RT": RT,
        "img_size": np.array([img_height, img_width]),
        "save_dir": dirpath,
        "frustrum_path": frustrum_path,
        "offset": offset,
        "dataset": dataset,
    }
    sio.savemat(mat_file, data)
    # blend_file = os.path.join(curr_path, 'model.blend')
    blend_file = os.path.join(curr_path, "novel_view.blend")
    png_file = osp.join(dirpath, "image.png")

    # blender_exec = '/Pool3/users/nileshk/libs/blender/blender-2.71/blender'
    command = f"bash {curr_path}/render_matterport.sh {blender_exec} {blend_file} {renderNovelView} {mesh_file} {png_file} {mat_file}"
    pdb.set_trace()
    os.system(command)
    pdb.set_trace()
    base_name = osp.dirname(png_file)
    depth_file = osp.join(base_name, "depth0001.exr")
    normal_file = osp.join(base_name, "normal0001.png")
    novel_dirs = [dirname for dirname in os.listdir(dirpath) if "novel" in dirname]
    novel_dirs.sort()
    novel_views = []
    for novel_dir in novel_dirs:
        full_novel_path = osp.join(dirpath, novel_dir)
        depth_file = osp.join(full_novel_path, "depth0001.exr")
        normal_file = osp.join(full_novel_path, "normal0001.png")
        try:
            img = read_im(osp.join(full_novel_path, "image.png"))
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth = normalize_depth(depth, md=max_depth)
            normal = imageio.imread(normal_file)[..., 0:3]
        except Exception as e:
            img = np.ones((400, 400, 3))
            depth = img[:, :, 0:1]
            normal = img * 1
        novel_view = {}
        novel_view["image"] = img
        novel_view["depth"] = depth
        novel_view["normal"] = normal
        novel_views.append(novel_view)
    shutil.rmtree(dirpath)
    return novel_views


def render_mesh_matterport(
    mesh_file, png_file, RT, intrinsic=None, img_width=400, img_height=800, max_depth=12
):
    # pdb.set_trace()
    dirpath = tempfile.mkdtemp()
    # dirpath = '.'
    mat_file = osp.join(dirpath, "pose.mat")
    import scipy.io as sio

    if intrinsic is None:
        intrinsic = np.eye(3) * 35

    data = {
        "RT": RT,
        "intrinsic": intrinsic,
        "img_size": np.array([img_height, img_width]),
    }
    sio.savemat(mat_file, data)
    # pose_path = osp.join()

    blend_file = os.path.join(curr_path, "model.blend")

    # blender_exec = '/Pool3/users/nileshk/libs/blender/blender-2.71/blender'
    command = f"bash {curr_path}/render_matterport.sh {blender_exec} {blend_file} {renderFileMatterport} {mesh_file} {png_file} {mat_file}"
    base_name = osp.dirname(png_file)
    depth_file = osp.join(base_name, "depth0001.exr")
    normal_file = osp.join(base_name, "normal0001.png")

    os.system(command)

    try:
        img = read_im(png_file)
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = normalize_depth(depth, md=max_depth)
        normal = imageio.imread(normal_file)[..., 0:3]
        return img, depth[:, :, 0], normal
        # return img[100:500, 265:665 :]
    except:
        return np.ones((400, 400, 3)), np.ones((400, 400, 1)), np.ones((400, 400, 3))


def render_mesh_shapenet(
    mesh_file, png_file, RT, intrinsic=None, img_width=400, img_height=800, max_depth=12
):
    # pdb.set_trace()
    dirpath = tempfile.mkdtemp()
    # dirpath = '.'
    mat_file = osp.join(dirpath, "pose.mat")
    import scipy.io as sio

    if intrinsic is None:
        intrinsic = np.eye(3) * 35

    data = {
        "RT": RT,
        "intrinsic": intrinsic,
        "img_size": np.array([img_height, img_width]),
    }
    sio.savemat(mat_file, data)
    # pose_path = osp.join()

    blend_file = os.path.join(curr_path, "model.blend")
    # blender_exec = '/Pool3/users/nileshk/libs/blender/blender-2.71/blender'
    command = f"bash {curr_path}/render_matterport.sh {blender_exec} {blend_file} {renderFileShapenet} {mesh_file} {png_file} {mat_file}"
    base_name = osp.dirname(png_file)
    depth_file = osp.join(base_name, "depth0001.exr")
    normal_file = osp.join(base_name, "normal0001.png")

    os.system(command)

    try:
        img = read_im(png_file)
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = normalize_depth(depth, md=max_depth)
        normal = imageio.imread(normal_file)[..., 0:3]
        return img, depth[:, :, 0], normal
        # return img[100:500, 265:665 :]
    except:
        return np.ones((400, 400, 3)), np.ones((400, 400, 1)), np.ones((400, 400, 3))


def render_mesh_threedf(
    mesh_file, png_file, RT, intrinsic=None, img_width=400, img_height=800, max_depth=12
):
    # pdb.set_trace()
    dirpath = tempfile.mkdtemp()
    # dirpath = '.'
    mat_file = osp.join(dirpath, "pose.mat")
    import scipy.io as sio

    if intrinsic is None:
        intrinsic = np.eye(3) * 35
    data = {
        "RT": RT,
        "intrinsic": intrinsic,
        "img_size": np.array([img_height, img_width]),
    }
    sio.savemat(mat_file, data)
    # pose_path = osp.join()

    blend_file = os.path.join(curr_path, "model.blend")

    # blender_exec = '/Pool3/users/nileshk/libs/blender/blender-2.71/blender'
    command = f"bash {curr_path}/render_matterport.sh {blender_exec} {blend_file} {renderFileThreeDf} {mesh_file} {png_file} {mat_file}"
    base_name = osp.dirname(png_file)
    depth_file = osp.join(base_name, "depth0001.exr")
    normal_file = osp.join(base_name, "normal0001.png")

    os.system(command)

    try:
        img = read_im(png_file)
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = normalize_depth(depth, md=max_depth)
        normal = imageio.imread(normal_file)[..., 0:3]
        return img, depth[:, :, 0], normal
        # return img[100:500, 265:665 :]
    except:
        return np.ones((400, 400, 3)), np.ones((400, 400, 1)), np.ones((400, 400, 3))


def render_mesh_scannet(
    mesh_file, png_file, RT, intrinsic=None, img_width=400, img_height=800, max_depth=12
):
    # pdb.set_trace()
    dirpath = tempfile.mkdtemp()
    # dirpath = '.'
    mat_file = osp.join(dirpath, "pose.mat")
    import scipy.io as sio

    if intrinsic is None:
        intrinsic = np.eye(3) * 35

    data = {
        "RT": RT,
        "intrinsic": intrinsic,
        "img_size": np.array([img_height, img_width]),
    }
    sio.savemat(mat_file, data)
    # pose_path = osp.join()
    blend_file = os.path.join(curr_path, "model.blend")

    # pdb.set_trace()
    # blender_exec = '/Pool3/users/nileshk/libs/blender/blender-2.71/blender'
    command = f"bash {curr_path}/render_matterport.sh {blender_exec} {blend_file} {renderFileScannet} {mesh_file} {png_file} {mat_file}"
    base_name = osp.dirname(png_file)
    depth_file = osp.join(base_name, "depth0001.exr")
    normal_file = osp.join(base_name, "normal0001.png")
    # print(command)
    os.system(command)

    try:
        img = read_im(png_file)
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = normalize_depth(depth, md=max_depth)
        normal = imageio.imread(normal_file)[..., 0:3]
        return img, depth[:, :, 0], normal
        # return img[100:500, 265:665 :]
    except:
        return np.ones((400, 400, 3)), np.ones((400, 400)), np.ones((400, 400, 3))


# def render_mesh(mesh_file, png_file, az=60, el=20, dist_scale=1.0, upsamp=1.0, theta=0, focal_length=37):
#     blend_file = os.path.join(curr_path, 'model.blend')

#     command = 'bash {}/render.sh {} {} {} {} {} {} {} {} {} {}'.format(curr_path, blender_exec, blend_file, renderFilePix3d ,  mesh_file, png_file, az, el, dist_scale, upsamp, theta, focal_length)
#     #print(command)
#     os.system(command)

#     try:
#         img = read_im(png_file)
#         # return img[100:500, 265:665 :]
#         return img
#     except:
#         # return np.ones((400, 400, 3))
#         return np.ones((224, 224, 3))


def render_mesh(
    mesh_file,
    png_file,
    az=60,
    el=20,
    dist_scale=1.0,
    upsamp=1.0,
    theta=0,
    focal_length=37,
):
    blend_file = os.path.join(curr_path, "model.blend")

    command = "bash {}/render.sh {} {} {} {} {} {} {} {} {} {}".format(
        curr_path,
        blender_exec,
        blend_file,
        renderFile,
        mesh_file,
        png_file,
        az,
        el,
        dist_scale,
        upsamp,
        theta,
        focal_length,
    )
    # print(command)
    # pdb.set_trace()
    os.system(command)

    try:
        img = read_im(png_file)
        # return img[100:500, 265:665 :]
        return img
    except:
        # return np.ones((400, 400, 3))
        return np.ones((224, 224, 3))
