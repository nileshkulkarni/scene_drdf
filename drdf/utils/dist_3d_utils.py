import copy
import pdb

import numpy as np
import torch
import trimesh

from . import geometry_utils, grid_utils, intersection_finder_utils


def gen_depth_from_net(
    opts, net, device, meta_data, intersection_finder, resolution=128
):

    image_tensor = meta_data["img"].to(device=device)
    Kndc = meta_data["Kndc"].to(device=device)
    net.filter_images(
        images=image_tensor[
            None,
        ]
    )
    if "RT" in meta_data.keys():
        RT = meta_data["RT"].to(device=device)
    else:
        RT = None
    if "transforms" in meta_data.keys():
        transforms = meta_data["transforms"]
    else:
        transforms = None

    if True:
        coords, ray_dir = create_pixel_query_grid(
            resolution=resolution, z_max=meta_data["z_max"], RT=RT, Kndc=Kndc
        )
        coords = torch.cat([coords, ray_dir * 0], dim=0)

    distances, depths = reconstruction_vol(
        net=net,
        cuda=device,
        Kndc=Kndc,
        RT=RT,
        coords=coords,
    )

    coords = coords.transpose(1, 0)
    coords = coords.reshape(resolution, resolution, resolution, 6).cpu().numpy()
    distances = distances.reshape(resolution, resolution, resolution)
    depths = depths.reshape(resolution, resolution, resolution)
    distances = distances.cpu().numpy()
    depths = depths.cpu().numpy()

    _, results = intersection_finder_utils.double_batch_compute_intersections(
        distances,
        depths,
        intersection_finder,
        add_zeros=True,
    )
    interesection_bumps_pred = []
    intersection_coords = []
    for result in results:
        non_zero = np.where(result[0] > 0)[0]
        if len(non_zero) > 0:
            interesection_bumps_pred.append(result[0][non_zero[0]])
        else:
            interesection_bumps_pred.append(0)
        if len(result[2]) > 0:
            intersection_coords.extend(
                [result[2][0]]
            )  ## this is only keeping the first one.

    interesection_bumps_pred = np.stack(interesection_bumps_pred)
    interesection_bumps_pred = interesection_bumps_pred.reshape(
        resolution, resolution, -1
    )

    intersection_coords = np.array(intersection_coords)
    # yapf: disable
    try:
        if len(intersection_coords) > 0:
            intersection_locs = coords[(intersection_coords[:,0], intersection_coords[:,1], intersection_coords[:,2])]
            point_locs =  intersection_locs[:, :3]
            depth_map = interesection_bumps_pred.transpose(1,0, 2)[:,:,0]
        else:
            depth_map = np.zeros((resolution, resolution))
    except IndexError:
        depth_map = np.zeros((resolution, resolution))
        breakpoint()

    # yapf: enable
    return depth_map


def gen_mesh_from_net(opts, net, device, meta_data, intersection_finder):

    image_tensor = meta_data["img"].to(device=device)
    Kndc = meta_data["Kndc"].to(device=device)
    save_path_pref = meta_data["save_path"]
    if "meshes" in meta_data.keys():
        meshes = meta_data["meshes"]
    else:
        meshes = None
    net.filter_images(
        images=image_tensor[
            None,
        ]
    )
    if "RT" in meta_data.keys():
        RT = meta_data["RT"].to(device=device)
    else:
        RT = None
    if "transforms" in meta_data.keys():
        transforms = meta_data["transforms"]
    else:
        transforms = None

    save_path = save_path_pref
    resolution = opts.MODEL.RESOLUTION
    breakpoint()
    if True:
        coords, ray_dir = create_pixel_query_grid(
            resolution=resolution, z_max=meta_data["z_max"], RT=RT, Kndc=Kndc
        )
        coords = torch.cat([coords, ray_dir * 0], dim=0)
    distances, depths = reconstruction_vol(
        net=net,
        cuda=device,
        Kndc=Kndc,
        RT=RT,
        coords=coords,
    )

    resolution = opts.MODEL.RESOLUTION
    coords = coords.transpose(1, 0)
    coords = coords.reshape(resolution, resolution, resolution, 6).cpu().numpy()
    distances = distances.reshape(resolution, resolution, resolution)
    depths = depths.reshape(resolution, resolution, resolution)
    distances = distances.cpu().numpy()
    depths = depths.cpu().numpy()

    _, results = intersection_finder_utils.double_batch_compute_intersections(
        distances,
        depths,
        intersection_finder,
        add_zeros=True,
    )
    interesection_bumps_pred = []
    intersection_coords = []
    for result in results:
        interesection_bumps_pred.append(result[0])
        if len(result[2]) > 0:
            intersection_coords.extend(result[2])

    interesection_bumps_pred = np.stack(interesection_bumps_pred)
    interesection_bumps_pred = interesection_bumps_pred.reshape(
        resolution, resolution, -1
    )
    intersection_coords = np.array(intersection_coords)
    # yapf: disable
    intersection_locs = coords[(intersection_coords[:,0], intersection_coords[:,1], intersection_coords[:,2])]

    point_locs =  intersection_locs[:, :3]
    mesh = convert_points_to_mesh(point_locs, save_path_pref, radius=0.05)
    # yapf: enable
    return mesh


def convert_points_to_mesh(points, mesh_file, radius=0.1):

    mesh = trimesh.points.PointCloud(points)
    trimesh.exchange.export.export_mesh(mesh, mesh_file)
    # pdb.set_trace()
    icosphere = trimesh.creation.icosahedron()
    icosphere.vertices *= radius
    faces = icosphere.faces
    vertices = icosphere.vertices
    faces_offset = np.arange(0, len(points), dtype=np.int32)
    faces_offset = len(vertices) * faces_offset[:, None] * np.ones((1, len(faces)))

    new_vertices = (
        vertices[
            None,
        ]
        + points[:, None, :]
    )
    new_vertices = new_vertices.reshape(-1, 3)
    new_faces = (
        faces_offset[:, :, None]
        + faces[
            None,
        ]
    )
    new_faces = new_faces.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    # trimesh.exchange.export.export_mesh(mesh, mesh_file)
    # for p in points:
    #     sp = copy.deepcopy(icosphere)
    #     sp.vertices += p[None, :]
    #     mesh = mesh + sp
    # breakpoint()
    # pdb.set_trace()
    return mesh


def create_pixel_query_grid(resolution, z_max, RT, Kndc):
    b_min = np.array([-1.0, -1.0, 0.1])
    b_max = np.array([1.0, 1.0, z_max])
    coords = grid_utils.create_pixel_aligned_grid(
        resolution, resolution, resolution, b_min, b_max, transform=None
    )
    coords = coords.reshape(3, -1)
    device = RT.device
    coords = coords.to(device=device)
    coords = geometry_utils.convert_pixel_to_world_points(coords, RT, Kndc)
    ray_dir = coords - RT[0:3, 3][:, None]
    ## TODO: Convert ray directions to camera frame.
    ## these directions are not relative to the camera. But in the global frame of reference.
    ## So we need to transform them to the camera frame. Eventually at some point.
    ray_dir = torch.nn.functional.normalize(ray_dir, dim=0)
    return coords, ray_dir


def reconstruction_vol(
    net,
    cuda,
    Kndc,
    RT,
    coords,
):
    """
    Create distance volume from network:
    Args:
        net: network
        cuda: cuda device

    """

    # coords, mat = grid_utils.create_pixel_aligned_grid(
    #     resolution, resolution, resolution, b_min, b_max, transform=transform
    # )
    # pdb.set_trace()
    # coords = geometry_utils.convert_pixel_to_world_points(
    #     coords, RT[0], calibndc[0]
    # )
    """
        points is torch.FloatTensor

    """

    def eval_func(points):
        # points = np.repeat(points, net.num_views, axis=0)
        # samples = points.to(device=cuda).float() ## this is B x 6 x N. Where first 3 are xyz, last 3 are ray dir
        # transform points here.
        coords = points[:, :3, :]
        ray_dir = points[:, 3:, :]
        net.query(
            points=coords,
            ray_dir=ray_dir,
            kNDC=Kndc[
                None,
            ],
            RT=RT[
                None,
            ],
        )
        pred = net.get_preds()
        return pred

    def get_depth(points):
        coords = points[:, :3, :]
        ray_dir = points[:, 3:, :]

        # samples = points.to(device=cuda).float()
        depths, xyz_ndc = net.get_depth_points(
            points=coords,
            ray_dir=ray_dir,
            RT=RT[
                None,
            ],
            kNDC=Kndc[
                None,
            ],
        )
        return depths, xyz_ndc

    net.eval()
    coords = coords.view(6, -1)
    with torch.no_grad():
        distances = eval_batch(coords, eval_func)
        depths, _ = eval_batch(coords, get_depth)
    return distances, depths


def convert_intersections_to_mesh(intersections, mesh_file):
    mesh = trimesh.base.Trimesh(
        vertices=intersections[:, :3], faces=intersections[:, 3:]
    )
    mesh.export(mesh_file)
    return mesh_file


def recursive_collate(outputs):
    if len(outputs) > 0:
        if type(outputs[0]) == tuple:
            collected_outputs = ()
            for i in range(len(outputs[0])):
                collected_outputs += (
                    torch.cat([output[i] for output in outputs], dim=-1),
                )
        else:
            collected_outputs = torch.cat(outputs, dim=-1)

        return collected_outputs
    else:
        return []


def eval_batch(points, eval_func, num_samples=10000):
    num_pts = points.shape[1]
    num_batches = (num_pts // num_samples) + 1
    outputs = []
    for i in range(num_batches):
        pts = points[:, i * num_samples : i * num_samples + num_samples]
        outputs.append(eval_func(pts[None]))

    outputs = recursive_collate(outputs)
    if type(outputs) == tuple:
        return_outs = ()
        for ix in range(len(outputs)):
            return_outs = return_outs + (outputs[ix][0],)
        outputs = return_outs
        # outputs = (output[0] for output in outputs)
    else:
        outputs = outputs[0]
    return outputs
