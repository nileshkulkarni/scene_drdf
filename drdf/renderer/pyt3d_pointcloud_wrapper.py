from copy import deepcopy

import numpy as np
import pytorch3d
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    PerspectiveCameras,
    PointLights,
    PointRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)


class PointRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"

    def __init__(
        self,
        cameras,
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
        self.renderer = self.setup_renderer(cameras)

    def setup_renderer(self, cameras):
        # for sillhouette rendering
        sigma = 1e-4
        raster_settings = PointRasterizationSettings(
            image_size=self.image_size,
            blur_radius=self.blur_radius,
            # blur_radius=np.log(1. / 1e-4 - 1.) * sigma, # this will create large sphere for each face
            faces_per_pixel=self.faces_per_pixel,
            clip_barycentric_coords=False,
            max_faces_per_bin=self.max_faces_per_bin,
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
        return renderer

    def render(self, meshes, ret_mask=False):
        images = self.renderer(meshes)
        # print(images.shape)
        if ret_mask:
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :3].cpu().detach().numpy(), mask > 0
        return images[0, ..., :3].cpu().detach().numpy()


class Pyt3DWrapperPointCloud:
    def __init__(self, image_size, device="cuda:0", colors=SMPL_OBJ_COLOR_LIST):
        self.colors = deepcopy(colors)
        self.device = device
        self.front_camera = self.get_matterport_camera(device)
        self.renderer = PointRendererWrapper(
            cameras=self.front_camera, image_size=image_size, device=device
        )

    @staticmethod
    def get_matterport_camera(device="cuda:0"):
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

    def render_point_clouds(self, point_clouds):
        images = self.renderer(
            point_clouds,
        )
        return images

    def prepare_point_cloud_render(self, point_cloud_lst, colors):

        out_point_clouds = []
        for pcl, color in zip(point_cloud_lst, colors):
            verts = torch.Tensor(pcl).to(self.device)
            rgb = torch.Tensor(pcl * 0 + color).to(self.device)
            point_cloud = Pointclouds(points=[verts], features=[rgb])
