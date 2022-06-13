import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from src.utils import merge_classes2
from src.utils import make_uvcoords, merge_nusc_static_classes, downsample_gt, merge_classes2, merge_classes_lyft


def collate_nusc_s_occ_200down100up(batch):
    """
    Collate fuction for:
        - NuScenes
        - Singe image input models
        - Ground truths with occluded regions masked out

    Merges input classes "road_segment" + "lane" into "drivable_area"

    Down Up scheme: downsample 200 to 100, then upsample 100 to required resolution
    Output resolution: 100
    """

    idxs, images, calibs, gt_maps, grids2d, class_dict, des_image_size = zip(*batch)

    # All class dicts are same, so only select first
    class_dict = class_dict[0]

    og_w, og_h = 1600, 900
    desired_w, desired_h = des_image_size[0]
    scale_w, scale_h = desired_w / og_w, desired_h / og_h

    # Scale image
    images = [
        image.resize((int(image.size[0] * scale_w), int(image.size[1] * scale_h)))
        for image in images
    ]

    # Pad images to the same dimensions
    w = [img.size[0] for img in images]
    h = [img.size[1] for img in images]

    delta_w = [desired_w - img.size[0] for img in images]
    delta_h = [desired_h - img.size[1] for img in images]

    pad_left = [int(d / 2) for d in delta_w]
    pad_right = [delta_w[i] - pad_left[i] for i in range(len(images))]
    pad_top = [int(d / 2) for d in delta_h]
    pad_bottom = [delta_h[i] - pad_top[i] for i in range(len(images))]

    left = [0 - pad_left[i] for i in range(len(images))]
    right = [pad_right[i] + w[i] for i in range(len(images))]
    top = [0 - pad_top[i] for i in range(len(images))]
    bottom = [pad_bottom[i] + h[i] for i in range(len(images))]

    images = [
        images[i].crop((left[i], top[i], right[i], bottom[i]))
        for i in range(len(images))
    ]

    w = [img.size[0] for img in images]
    h = [img.size[1] for img in images]

    # Stack images and calibration matrices along the batch dimension
    images = torch.stack([to_tensor(img) for img in images])
    gt_maps = torch.stack(
        [
            torch.stack([to_tensor(class_map) for class_map in gt_map])
            for gt_map in gt_maps
        ]
    ).squeeze(2)
    calibs = torch.stack(calibs)
    grids2d = torch.stack(grids2d)

    # Create visbility mask from lidar and fov masks
    lidar_ray_mask = gt_maps[:, -2:-1]
    fov_mask = gt_maps[:, -1:]
    vis_mask = lidar_ray_mask * fov_mask
    vis_mask = downsample_gt(vis_mask, [[100, 100]])[0]

    # Downsample all classes from 200x200 to 100x100
    gt_maps_all_classes = gt_maps[:, :-2]
    gt_maps_200down100 = downsample_gt(gt_maps_all_classes, [[100, 100]])[0]

    # Apply vis mask
    gt_maps = gt_maps_200down100 * vis_mask

    # Modify calibration matrices
    # Scale first two rows of calibration matrix
    calibs[:, :2, :] *= scale_w
    # cx' = cx - du
    calibs[:, 0, 2] = calibs[:, 0, 2] + torch.tensor(pad_left)
    # cy' = cy - dv
    calibs[:, 1, 2] = calibs[:, 1, 2] + torch.tensor(pad_top)

    # Create a vector of indices
    idxs = torch.LongTensor(idxs)

    # Add drivable area to first cell in FOV
    # gt_maps = gt_maps_masked.squeeze(2)
    gt_maps[:, 0, 0, int(gt_maps.shape[-1] / 2 - 1)] = 1
    gt_maps[:, 0, 0, int(gt_maps.shape[-1] / 2)] = 1

    # Merge ground truth for road_segment + lane into one and create new gt
    classes_to_merge = ["drivable_area", "road_segment", "lane"]
    class_idx_to_merge = [class_dict[name] for name in classes_to_merge]
    merged_class_idx = class_dict["drivable_area"]
    gt_maps_new = torch.stack(
        [
            merge_nusc_static_classes(gt, class_idx_to_merge, merged_class_idx)
            for gt in gt_maps
        ]
    )

    return idxs, images, calibs, gt_maps_new, grids2d, vis_mask


def collate_nusc_s(batch):
    """
    Collate fuction for:
        - NuScenes
        - Singe image input models
        - Ground truths with occluded regions masked out
    """

    images, cls_maps, vis_masks, calibs, grids2d = zip(*batch)

    # Stack images and calibration matrices along the batch dimension
    images = torch.stack(images)
    cls_maps = torch.stack(cls_maps)
    vis_masks = torch.stack(vis_masks)
    calibs = torch.stack(calibs)
    grids2d = torch.stack(grids2d)

    return (images, calibs, grids2d), (cls_maps, vis_masks)