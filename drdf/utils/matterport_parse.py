import time
from typing import IO, Any, DefaultDict, Dict, List

import numpy as np
import trimesh

from .default_imports import *


def read_camera_pose(f: IO) -> np.array:
    camera = []
    for line in f.readlines():
        line = line.strip()
        camera.append([float(l) for l in line.split()])

    camera = np.array(camera)
    return camera


def load_face_metadata(metadata, face_cachedir, low_res=False):
    house_name = metadata["house_id"]

    house_name = metadata["house_id"]
    image_name = metadata["image_names"]
    mat_filename = metadata["image_names"].replace(".jpg", ".mat")
    data_mat = osp.join(face_cachedir, house_name, mat_filename)
    data = sio.loadmat(data_mat, squeeze_me=True, struct_as_record=True)
    if not low_res:
        face_labels, fids_c, fids_w = None, None, None
        face_instances = None
        if "fclass" in data.keys():
            face_labels = data["fclass"]
        if "finstance" in data.keys():
            face_instances = data["finstance"]

        if "fids_c" in data.keys():
            fids_c = data["fids_c"]

        if "fids_w" in data.keys():
            fids_w = data["fids_w"]

        # print(data['fids'])
        if isinstance(data["fids"], np.ndarray) and len(data["fids"]) > 0:
            return data["fids"], face_labels, face_instances, fids_c, fids_w
        else:
            return None, None, None, None, None
    else:
        fids = data["low_res"]["fids"].item()
        return fids


def delete_unused_verts(mesh):
    verts = mesh["vertices"]
    faces = mesh["faces"]
    unique_verts = np.unique(faces.reshape(-1))
    oldVertId2newVertId = {x: i for (i, x) in enumerate(unique_verts)}
    verts = verts[unique_verts]
    new_faces = np.array([oldVertId2newVertId[k] for k in faces.reshape(-1)])
    new_faces = new_faces.reshape(-1, 3)
    new_mesh = {
        "vertices": verts,
        "faces": new_faces,
    }
    # new_mesh = trimesh.Trimesh(vertices=verts, faces=new_faces)
    # trimesh.exchange.obj.export_obj(new_mesh, 'temp.obj')
    return new_mesh


def preload_meshes(
    indexed_items: Dict[str, Any], matterport_dir: str
) -> Dict[str, trimesh.Trimesh]:
    houses2mesh = {}
    for index in range(len(indexed_items)):
        house_id = indexed_items[index]["house_id"]
        if house_id in houses2mesh.keys():
            continue
        else:
            mesh_dir = osp.join(
                matterport_dir, "poisson_meshes", house_id, "poisson_meshes"
            )
            mesh_path = osp.join(mesh_dir, f"{house_id}.ply")
            # print(mesh_path)

            # from plyfile import PlyData, PlyElement
            # mesh_ply = PlyData.read(mesh_path)
            sttime = time.time()
            with open(mesh_path, "rb") as f:
                mesh = trimesh.exchange.ply.load_ply(f)
            houses2mesh[house_id] = mesh
    return houses2mesh


def preload_conf_files(
    house_ids: List[str], matterport_dir, img_split: str = "all", group_by_region=False
) -> Dict[int, Any]:
    indexed_items = []
    for house_id in house_ids:

        pano2region_dir = osp.join(
            matterport_dir, "house_segmentations", house_id, "house_segmentations"
        )
        pano2region_file_path = osp.join(pano2region_dir, "panorama_to_region.txt")
        pano2region = {}
        with open(pano2region_file_path) as f:
            for line in f.readlines():
                line = line.strip().split()
                pano2region[line[1]] = (line[2], line[3])

        conf_file_dir = osp.join(
            matterport_dir,
            "undistorted_camera_parameters",
            house_id,
            "undistorted_camera_parameters",
        )

        conf_file_path = osp.join(conf_file_dir, f"{house_id}.conf")

        with open(conf_file_path) as f:
            conf_data = parse_conf_file(f)

        inds = np.array([i for i in range(len(conf_data["image_names"]))])
        rng = np.random.RandomState([ord(c) for c in house_id])
        rng.shuffle(inds)
        n_imgs = len(inds)
        house_items = []
        house_regions = DefaultDict(list)
        for ix in inds:
            ix_struct = {}
            ix_struct["house_id"] = house_id
            for key in conf_data.keys():
                ix_struct[key] = conf_data[key][ix]
            pano_name = ix_struct["image_names"].split("_")[0]
            if pano_name in pano2region.keys():
                ix_struct["region_id"], ix_struct["region_type"] = pano2region[
                    pano_name
                ]

            ix_struct["uuid"] = "{}_{}".format(
                house_id, ix_struct["image_names"].split(".")[0]
            )
            house_items.append(ix_struct)
            if group_by_region:
                house_regions[ix_struct["region_id"]].append(ix_struct)

        if group_by_region:
            house_items = [k for k in house_regions.values()]
        indexed_items.extend(house_items)
    return indexed_items


def parse_conf_file(f: IO) -> Dict[str, Any]:
    n_images = None
    image_names, depth_image_names, intrinsics, cam2world = [], [], [], []
    current_intrinsics = None
    for lx, line in enumerate(f.readlines()):
        line = line.strip()
        if "intrinsics" in line:
            current_intrinsics = np.array([float(f) for f in line.split()[1:]])
            current_intrinsics = current_intrinsics.reshape(3, 3)
            continue
        if "scan" in line:
            line = line.split()
            cam2world_matrix = np.array([float(l) for l in line[3:]])
            cam2world_matrix = cam2world_matrix.reshape(4, 4)

            depth_image_names.append(line[1])
            image_names.append(line[2])
            cam2world.append(cam2world_matrix)
            intrinsics.append(current_intrinsics * 1)

    ## parse region files too.

    data = {}
    data["image_names"] = image_names
    data["depth_image_name"] = depth_image_names
    data["intrinsics"] = intrinsics
    data["cam2world"] = cam2world
    return data


def filter_indices_regions(indexed_items, filter_keys, **kwargs):
    filtered_items = []
    for index_item in indexed_items:
        filtered_list = filter_indices(index_item, filter_keys, kwargs)
        filtered_items.append(filtered_list)
    return filtered_items


def filter_indices(indexed_items, filter_keys, keep_valid_regions=False):
    filtered_items = []
    for index_item in indexed_items:
        for filter_key in filter_keys:
            if filter_key in index_item["image_names"]:
                if keep_valid_regions:
                    if int(index_item["region_id"]) > 0:
                        filtered_items.append(index_item)
                else:
                    filtered_items.append(index_item)
    return filtered_items
