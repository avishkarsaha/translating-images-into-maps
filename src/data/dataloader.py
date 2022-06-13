import os
import torch
import csv

from pathlib import Path
import io
import lmdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import (
    view_points,
    box_in_image,
    BoxVisibility,
    transform_matrix,
)
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.data_classes import LidarPointCloud

from src import utils


class nuScenesMaps(Dataset):

    def __init__(
            self,
            root="temp",
            split="train_mini",
            grid_size=(50.0, 50.0),
            grid_res=1.0,
            classes=[
                "bus",
                "bicycle",
                "car",
                "construction_vehicle",
                "motorcycle",
                "trailer",
                "truck",
                "pedestrian",
            ],
            dataset_size=1.0,
            mini=False,
            desired_image_size=(1280, 720),
            gt_out_size=(100, 100)
    ):
        self.dataset_size = dataset_size
        self.desired_image_size = desired_image_size
        self.gt_out_size = gt_out_size

        # paths for data files
        self.root = os.path.join(root)
        self.gtmaps_db_path = os.path.join(
            root, "lmdb",
            "semantic_maps_new_200x200"
        )
        self.images_db_path = os.path.join(
            root, "lmdb",
            "samples", "CAM_FRONT"
        )

        # databases
        if mini:
            self.nusc = NuScenes(version="v1.0-mini",
                                 dataroot=self.root,
                                 verbose=False)
        else:
            self.nusc = NuScenes(version="v1.0-trainval", dataroot=self.root, verbose=False)

        self.tokens = read_split(
            os.path.join(root, "splits", "{}.txt".format(split))
        )
        self.gtmaps_db = lmdb.open(
            path=self.gtmaps_db_path,
            readonly=True,
            readahead=False,
            max_spare_txns=128,
            lock=False,
        )
        self.images_db = lmdb.open(
            path=self.images_db_path,
            readonly=True,
            readahead=False,
            max_spare_txns=128,
            lock=False,
        )

        # Set classes
        self.classes = list(classes)
        self.classes.append("lidar_ray_mask_dense")
        self.class2idx = {
            name: idx for idx, name in enumerate(self.classes)
        }
        self.nusc_classes = [
            "vehicle.bus",
            "vehicle.bicycle",
            "vehicle.car",
            "vehicle.construction",
            "vehicle.motorcycle",
            "vehicle.trailer",
            "vehicle.truck",
            "human.pedestrian",
        ]
        self.nuscclass2idx = {
            name: idx for idx, name in enumerate(self.nusc_classes)
        }
        # load FOV mask
        self.fov_mask = Image.open(
            os.path.join(root, "lmdb", "semantic_maps_new_200x200", "fov_mask.png")
        )
        # Make grid
        self.grid2d = utils.make_grid2d(grid_size, (-grid_size[0] / 2.0, 0.0), grid_res)

    def __len__(self):
        return int(len(self.tokens) * self.dataset_size - 1)

    def __getitem__(self, index):


        # Load sample ID
        sample_token = self.tokens[index]
        sample_record = self.nusc.get("sample", sample_token)
        cam_token = sample_record["data"]["CAM_FRONT"]
        cam_record = self.nusc.get("sample_data", cam_token)
        cam_path = self.nusc.get_sample_data_path(cam_token)
        id = Path(cam_path).stem

        # Load intrinsincs
        calib = self.nusc.get(
            "calibrated_sensor", cam_record["calibrated_sensor_token"]
        )["camera_intrinsic"]
        calib = np.array(calib)

        # Load input images
        image_input_key = pickle.dumps(id)
        with self.images_db.begin() as txn:
            value = txn.get(key=image_input_key)
            image = Image.open(io.BytesIO(value)).convert(mode='RGB')

        # resize/augment images
        image, calib = self.image_calib_pad_and_crop(image, calib)
        image = to_tensor(image)
        calib = to_tensor(calib).reshape(3, 3)

        # Load ground truth maps
        gtmaps_key = [pickle.dumps("{}___{}".format(id, cls)) for cls in self.classes]
        with self.gtmaps_db.begin() as txn:
            value = [txn.get(key=key) for key in gtmaps_key]
            gtmaps = [Image.open(io.BytesIO(im)) for im in value]

        # each map is of shape [1, 200, 200]
        mapsdict = {cls: to_tensor(map) for cls, map in zip(self.classes, gtmaps)}
        mapsdict["fov_mask"] = to_tensor(self.fov_mask)
        mapsdict = self.merge_map_classes(mapsdict)

        # Create visbility mask from lidar and fov masks
        lidar_ray_mask = mapsdict['lidar_ray_mask_dense']
        fov_mask = mapsdict['fov_mask']
        vis_mask = lidar_ray_mask * fov_mask
        mapsdict['vis_mask'] = vis_mask

        del mapsdict['lidar_ray_mask_dense'], mapsdict['fov_mask']

        # downsample maps to required output resolution
        mapsdict = {
            cls: F.interpolate(cls_map.unsqueeze(0), size=self.gt_out_size).squeeze(0)
            for cls, cls_map in mapsdict.items()
        }

        # apply vis mask to maps
        mapsdict = {
            cls: cls_map * mapsdict['vis_mask'] for cls, cls_map in mapsdict.items()
        }

        cls_maps = torch.cat(
            [cls_map for cls, cls_map in mapsdict.items() if 'mask' not in cls], dim=0
        )
        vis_mask = mapsdict['vis_mask']

        return (
            image, cls_maps, vis_mask, calib, self.grid2d
        )

    def merge_map_classes(self, mapsdict):
        classes_to_merge = ["drivable_area", "road_segment", "lane"]
        merged_class = 'drivable_area'
        maps2merge = torch.stack([mapsdict[k] for k in classes_to_merge])  # [n, 1, 200, 200]
        maps2merge = maps2merge.sum(dim=0)
        maps2merge = (maps2merge > 0).float()
        mapsdict[merged_class] = maps2merge
        del mapsdict['road_segment'], mapsdict['lane']
        return mapsdict

    def image_calib_pad_and_crop(self, image, calib):

        og_w, og_h = 1600, 900
        desired_w, desired_h = self.desired_image_size
        scale_w, scale_h = desired_w / og_w, desired_h / og_h
        # Scale image
        image = image.resize((int(image.size[0] * scale_w), int(image.size[1] * scale_h)))
        # Pad images to the same dimensions
        w = image.size[0]
        h = image.size[1]
        delta_w = desired_w - w
        delta_h = desired_h - h
        pad_left = int(delta_w / 2)
        pad_right = delta_w - pad_left
        pad_top = int(delta_h / 2)
        pad_bottom = delta_h - pad_top
        left = 0 - pad_left
        right = pad_right + w
        top = 0 - pad_top
        bottom = pad_bottom + h
        image = image.crop((left, top, right, bottom))

        # Modify calibration matrices
        # Scale first two rows of calibration matrix
        calib[:2, :] *= scale_w
        # cx' = cx - du
        calib[0, 2] = calib[0, 2] + pad_left
        # cy' = cy - dv
        calib[1, 2] = calib[1, 2] + pad_top

        return image, calib


def read_split(filename):
    """
    Read a list of NuScenes sample tokens
    """
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        return [val for val in lines if val != ""]


def create_batch_indices_old(split, batch_size, seq_len, n_pred_frames):
    nuscenes_root = "/vol/research/sceneEvolution/data/nuscenes"
    scene_len_file = os.path.join(
        nuscenes_root, "splits", (split + "_with_seq_len.txt")
    )
    scene_len = np.array(read_split(scene_len_file), dtype=np.int)
    cumsum_scene_len = np.cumsum(scene_len)
    zeros = np.zeros(len(cumsum_scene_len) + 1, dtype=np.int)
    zeros[1:] = cumsum_scene_len
    cumsum_scene_len = zeros

    idxs_batch_start = []
    idxs_batch_end = []
    idxs_batch_pred = []

    for idx_scene, scene in enumerate(scene_len):
        # Split scene length into chunks of size seq_len
        nbatches_in_scene = (scene - 1) // seq_len
        local_batch_num = (np.arange(nbatches_in_scene, dtype=np.int) + 1) * seq_len
        z = np.zeros(len(local_batch_num) + 1, dtype=np.int)
        z[1:] = local_batch_num
        local_batch_idx = z

        # Add cumsum scene_lengths to get global idx
        global_batch_idx = local_batch_idx + cumsum_scene_len[idx_scene]

        start_batch_idx = global_batch_idx[:-1]
        end_batch_idx = global_batch_idx[1:] - 1
        pred_batch_idx = end_batch_idx + n_pred_frames
        pred_batch_idx = np.clip(
            pred_batch_idx, a_min=0, a_max=scene - 1 + cumsum_scene_len[idx_scene]
        )

        idxs_batch_start.extend(list(start_batch_idx))
        idxs_batch_end.extend(list(end_batch_idx))
        idxs_batch_pred.extend(list(pred_batch_idx))

    return idxs_batch_start, idxs_batch_end, idxs_batch_pred


def create_batch_indices(split, batch_size, seq_len, n_pred_frames):
    nuscenes_root = "/vol/research/sceneEvolution/data/nuscenes"
    scene_len_file = os.path.join(
        nuscenes_root, "splits", (split + "_with_seq_len.txt")
    )
    scene_len = np.array(read_split(scene_len_file), dtype=np.int)
    cumsum_scene_len = np.cumsum(scene_len)
    zeros = np.zeros(len(cumsum_scene_len) + 1, dtype=np.int)
    zeros[1:] = cumsum_scene_len
    cumsum_scene_len = zeros

    # Offset for sliding window through sequences
    offset = seq_len // 2

    idxs_batch_start = []
    idxs_batch_end = []
    idxs_batch_pred = []

    for idx_scene, scene in enumerate(scene_len):
        # Split scene length into chunks of size seq_len
        nbatches_in_scene = (scene - 1) // seq_len
        local_batch_num = (np.arange(nbatches_in_scene, dtype=np.int) + 1) * seq_len
        z = np.zeros(len(local_batch_num) + 1, dtype=np.int)
        z[1:] = local_batch_num
        local_batch_idx = z

        # Add cumsum scene_lengths to get global idx
        global_batch_idx = local_batch_idx + cumsum_scene_len[idx_scene]

        start_batch_idx = global_batch_idx[:-1]
        end_batch_idx = global_batch_idx[1:] - 1
        pred_batch_idx = end_batch_idx + n_pred_frames
        pred_batch_idx = np.clip(
            pred_batch_idx, a_min=0, a_max=scene - 1 + cumsum_scene_len[idx_scene]
        )

        idxs_batch_start.extend(list(start_batch_idx))
        idxs_batch_end.extend(list(end_batch_idx))
        idxs_batch_pred.extend(list(pred_batch_idx))

        # Create intermediate sequences (first trim either end)
        start_batch_idx = start_batch_idx[1:-1] - offset
        end_batch_idx = end_batch_idx[1:-1] - offset
        pred_batch_idx = pred_batch_idx[1:-1] - offset

        idxs_batch_start.extend(list(start_batch_idx))
        idxs_batch_end.extend(list(end_batch_idx))
        idxs_batch_pred.extend(list(pred_batch_idx))

    return idxs_batch_start, idxs_batch_end, idxs_batch_pred


def create_batch_indices_wo_int(split, batch_size, seq_len, n_pred_frames):
    nuscenes_root = "/vol/research/sceneEvolution/data/nuscenes"
    scene_len_file = os.path.join(
        nuscenes_root, "splits", (split + "_with_seq_len.txt")
    )
    scene_len = np.array(read_split(scene_len_file), dtype=np.int)
    cumsum_scene_len = np.cumsum(scene_len)
    zeros = np.zeros(len(cumsum_scene_len) + 1, dtype=np.int)
    zeros[1:] = cumsum_scene_len
    cumsum_scene_len = zeros

    # Offset for sliding window through sequences
    offset = seq_len // 2

    idxs_batch_start = []
    idxs_batch_end = []
    idxs_batch_pred = []

    for idx_scene, scene in enumerate(scene_len):
        # Split scene length into chunks of size seq_len
        nbatches_in_scene = (scene - 1) // seq_len
        local_batch_num = (np.arange(nbatches_in_scene, dtype=np.int) + 1) * seq_len
        z = np.zeros(len(local_batch_num) + 1, dtype=np.int)
        z[1:] = local_batch_num
        local_batch_idx = z

        # Add cumsum scene_lengths to get global idx
        global_batch_idx = local_batch_idx + cumsum_scene_len[idx_scene]

        start_batch_idx = global_batch_idx[:-1]
        end_batch_idx = global_batch_idx[1:] - 1
        pred_batch_idx = end_batch_idx + n_pred_frames
        pred_batch_idx = np.clip(
            pred_batch_idx, a_min=0, a_max=scene - 1 + cumsum_scene_len[idx_scene]
        )

        idxs_batch_start.extend(list(start_batch_idx))
        idxs_batch_end.extend(list(end_batch_idx))
        idxs_batch_pred.extend(list(pred_batch_idx))

        # # Create intermediate sequences (first trim either end)
        # start_batch_idx = start_batch_idx[1:-1] - offset
        # end_batch_idx = end_batch_idx[1:-1] - offset
        # pred_batch_idx = pred_batch_idx[1:-1] - offset
        #
        # idxs_batch_start.extend(list(start_batch_idx))
        # idxs_batch_end.extend(list(end_batch_idx))
        # idxs_batch_pred.extend(list(pred_batch_idx))

    return idxs_batch_start, idxs_batch_end, idxs_batch_pred


def create_batch_indices2(split, seq_len):
    nuscenes_root = "/vol/research/sceneEvolution/data/nuscenes"
    scene_len_file = os.path.join(
        nuscenes_root, "splits", (split + "_with_seq_len.txt")
    )
    scene_len = np.array(read_split(scene_len_file), dtype=np.int)
    cumsum_scene_len = np.cumsum(scene_len)
    zeros = np.zeros(len(cumsum_scene_len) + 1, dtype=np.int)
    zeros[1:] = cumsum_scene_len
    cumsum_scene_len = zeros

    idxs_batch_start = []
    for idx_scene, scene in enumerate(scene_len):
        # Split scene length into chunks of size seq_len
        nbatches_in_scene = (scene - 1) // seq_len
        local_batch_num = (np.arange(nbatches_in_scene, dtype=np.int) + 1) * seq_len
        z = np.zeros(len(local_batch_num) + 1, dtype=np.int)
        z[1:] = local_batch_num
        local_batch_idx = z

        # Add cumsum scene_lengths to get global idx
        global_batch_idx = local_batch_idx + cumsum_scene_len[idx_scene]
        idxs_batch_start.extend(list(global_batch_idx))

    idxs_batch_end = list(np.array(idxs_batch_start, dtype=np.int) - 1)

    return idxs_batch_start, idxs_batch_end


def create_batch_indices_mini(split, batch_size, seq_len, n_pred_frames):
    nuscenes_root = "/vol/research/sceneEvolution/data/nuscenes"
    scene_len_file = os.path.join(
        nuscenes_root, "splits", (split + "_mini_with_seq_len.txt")
    )
    scene_len = np.array(read_split(scene_len_file), dtype=np.int)
    cumsum_scene_len = np.cumsum(scene_len)
    zeros = np.zeros(len(cumsum_scene_len) + 1, dtype=np.int)
    zeros[1:] = cumsum_scene_len
    cumsum_scene_len = zeros

    idxs_batch_start = []
    idxs_batch_end = []
    idxs_batch_pred = []

    for idx_scene, scene in enumerate(scene_len):
        # Split scene length into chunks of size seq_len
        nbatches_in_scene = (scene - 1) // seq_len
        local_batch_num = (np.arange(nbatches_in_scene, dtype=np.int) + 1) * seq_len
        z = np.zeros(len(local_batch_num) + 1, dtype=np.int)
        z[1:] = local_batch_num
        local_batch_idx = z

        # Add cumsum scene_lengths to get global idx
        global_batch_idx = local_batch_idx + cumsum_scene_len[idx_scene]

        start_batch_idx = global_batch_idx[:-1]
        end_batch_idx = global_batch_idx[1:] - 1
        pred_batch_idx = end_batch_idx + n_pred_frames
        pred_batch_idx = np.clip(
            pred_batch_idx, a_min=0, a_max=scene - 1 + cumsum_scene_len[idx_scene]
        )

        idxs_batch_start.extend(list(start_batch_idx))
        idxs_batch_end.extend(list(end_batch_idx))
        idxs_batch_pred.extend(list(pred_batch_idx))

    return idxs_batch_start, idxs_batch_end, idxs_batch_pred

