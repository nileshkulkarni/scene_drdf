"""
a simple wrapper for pytorch3d rendering
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
from copy import deepcopy

import numpy as np
import pytorch3d
import torch

# Data structures and functions for rendering
from pytorch3d.renderer import (
    AlphaCompositor,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    PointRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene

SMPL_OBJ_COLOR_LIST = [
    [0.65098039, 0.74117647, 0.85882353],  # SMPL
    [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
]

import imageio
import trimesh


class MeshRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"

    def __init__(
        self,
        image_size=1200,
        faces_per_pixel=1,
        device="cuda:0",
        blur_radius=0,
        lights=None,
        materials=None,
        max_faces_per_bin=50000,
    ):
        self.image_size = image_size
        self.faces_per_pixel = faces_per_pixel
        self.max_faces_per_bin = max_faces_per_bin  # prevent overflow, see https://github.com/facebookresearch/pytorch3d/issues/348
        self.blur_radius = blur_radius
        self.device = device
        self.lights = (
            lights
            if lights is not None
            else PointLights(
                ((0.5, 0.5, 0.5),),
                ((0.5, 0.5, 0.5),),
                ((0.05, 0.05, 0.05),),
                ((0, -2, 0),),
                device,
            )
        )
        self.materials = materials
        self.renderer = self.setup_renderer()

    def setup_renderer(self):
        # for sillhouette rendering
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=self.blur_radius,
            # blur_radius=np.log(1. / 1e-4 - 1.) * sigma, # this will create large sphere for each face
            faces_per_pixel=self.faces_per_pixel,
            clip_barycentric_coords=False,
            max_faces_per_bin=self.max_faces_per_bin,
        )
        shader = SoftPhongShader(
            device=self.device, lights=self.lights, materials=self.materials
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings), shader=shader
        )
        return renderer

    def render(self, meshes, cameras, ret_mask=False):
        images = self.renderer(meshes, cameras=cameras)
        # print(images.shape)
        if ret_mask:
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :3].cpu().detach().numpy(), mask > 0
        return images[0, ..., :3].cpu().detach().numpy()


class Pyt3DWrapper:
    def __init__(self, image_size, device="cuda:0", colors=SMPL_OBJ_COLOR_LIST):
        self.renderer = MeshRendererWrapper(image_size, device=device)
        self.front_camera = self.get_kinect_camera(device)
        self.colors = deepcopy(colors)
        self.device = device

    @staticmethod
    def get_kinect_camera(device="cuda:0"):
        R, T = torch.eye(3), torch.zeros(3)
        R[0, 0] = R[
            1, 1
        ] = -1  # pytorch3d y-axis up, need to rotate to kinect coordinate
        R = R.unsqueeze(0)
        T = T.unsqueeze(0)
        # fx, fy = 979.7844, 979.840  # focal length
        # cx, cy = 1018.952, 779.486  # camera centers

        fx, fy = 1075.1, 1075.8
        cx, cy = 629.7, 522.3
        color_w, color_h = 1280, 1024  # kinect color image size
        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)

        pyt3d_version = pytorch3d.__version__
        if pyt3d_version >= "0.6.0":
            cam = PerspectiveCameras(
                focal_length=focal_length,
                principal_point=cam_center,
                image_size=((color_w, color_h),),
                device=device,
                R=R,
                T=T,
                in_ndc=False,
            )
        else:
            cam = PerspectiveCameras(
                focal_length=focal_length,
                principal_point=cam_center,
                image_size=((color_w, color_h),),
                device=device,
                R=R,
                T=T,
            )
        return cam

    """
        input is a trimesh
    """

    def render_meshes(
        self,
        meshes,
    ):
        """
        render a list of meshes
        :param meshes: a list of psbody meshes
        :return: rendered image
        """
        colors = deepcopy(self.colors)
        pyt3d_mesh = self.prepare_render(meshes, colors)
        rend = self.renderer.render(pyt3d_mesh, self.front_camera)
        return rend

    def prepare_render(self, meshes, colors):
        py3d_meshes = []
        for mesh, color in zip(meshes, colors):
            vc = np.zeros_like(mesh.vertices)
            vc[:, :] = color
            text = TexturesVertex([torch.from_numpy(vc).float().to(self.device)])
            py3d_mesh = Meshes(
                [torch.from_numpy(mesh.vertices).float().to(self.device)],
                [torch.from_numpy(mesh.faces.astype(int)).long().to(self.device)],
                text,
            )
            py3d_meshes.append(py3d_mesh)
        joined = join_meshes_as_scene(py3d_meshes)
        return joined


from ..utils import geometry_utils


def render_world_mesh(meshes, RT, py3d_renderer):
    ## Transform meshes to camera coordiante frame.
    mesh_verts = np.array(meshes.vertices)
    breakpoint()
    mesh_verts = geometry_utils.transform_points(mesh_verts.transpose(), RT).transpose()
    mesh = trimesh.Trimesh(vertices=mesh_verts, faces=meshes.faces)
    rendered = py3d_renderer.render_meshes(
        [mesh],
    )
    return rendered
