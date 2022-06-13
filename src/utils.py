import time
import os
import torch
import numpy as np
import cv2

from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from pyquaternion import Quaternion

from collections import namedtuple, defaultdict

import src
from src.model.loss import dice_loss_per_class, dice_loss_per_class_infer
# from src.util.box_ops import box_cxcywh_to_xyxy

ObjectDataBEV = namedtuple(
    "ObjectData",
    ["classname", "x_pos", "z_pos", "x_width", "z_height", "visibility", "not_in_grid"],
)
ObjectData3D = namedtuple(
    "ObjectData",
    ["classname",
     "x_pos", "y_pos", "z_pos",  # object ceter
     "width", "height", "length",  # object dims
     "x1", "x2", "x3", "x4",  # object bottom corner x
     "z1", "z2", "z3", "z4",  # object bottom corner z
     "visibility", "not_in_grid"],
    # image-plane visibility and within 50m x 50m BEV grid check
)
Object3D = namedtuple(
    "ObjectData",
    ["classname",
     "x_pos", "y_pos", "z_pos",  # object ceter
     "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
     "z1", "z2", "z3", "z4",

     "visibility", "not_in_grid"],
)


class MetricDict(defaultdict):
    def __init__(self):
        super().__init__(float)
        self.count = defaultdict(int)

    def __add__(self, other):
        for key, value in other.items():
            self[key] += value
            self.count[key] += 1
        return self

    @property
    def mean(self):
        return {key: self[key] / self.count[key] for key in self.keys()}


class Timer(object):
    def __init__(self):
        self.total = 0
        self.runs = 0
        self.t = 0

    def reset(self):
        self.total = 0
        self.runs = 0
        self.t = 0

    def start(self):
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        self.t = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        self.total += time.perf_counter() - self.t
        self.runs += 1

    @property
    def mean(self):
        val = self.total / self.runs
        self.reset()
        return val


def merge_classes_lyft(input):
    """
    [B, C, H, W] ----> [B, c, H, W]
    """

    driv = input[:, 0:1]
    vehicles = input[:, 1:]
    vehicles_merged = (vehicles.sum(dim=1) > 0).float()

    return torch.cat([driv, vehicles_merged.unsqueeze(1)], dim=1)


def mask_to_objpos(mask, skip_classes=4):
    """

    :param mask:
    :return:
    """

    for scale_idx, scale in enumerate(mask):
        print("Scale: {}, {}".format(scale_idx, scale.shape))
        for batch_idx, batch in enumerate(scale):
            print("    Batch: {}, {}".format(batch_idx, batch.shape))
            for cls_idx, cls in enumerate(batch):
                print("        Class: {}, {}".format(cls_idx, cls.shape))
                # read image
                # img = cv2.imread('two_blobs.jpg')

                # convert to grayscale
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # threshold
                cls = (cls * 255).cpu().numpy().astype(np.uint8)
                print("        {}".format(cls.mean()))
                thresh = cv2.threshold(cls, 128, 255, cv2.THRESH_BINARY)[1]

                # get contours
                # result = img.copy()
                contours = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = contours[0] if len(contours) == 2 else contours[1]
                pos = []
                for cntr in contours:
                    x, y, w, h = cv2.boundingRect(cntr)
                    # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # print(x, y, w, h)
                    pos.append(np.array([x, y, h, w]))
                    plt.imshow(cls)
                    plt.title("{},{},{},{}".format(x, y, w, h))
                    plt.show()

                pos = np.array(pos)
                print(pos)


def downsample_gt(gt_s1, map_sizes, threshold=0, batch_list=False):
    gt_ms = []
    gt = gt_s1

    if batch_list:

        for batch in gt_s1:
            batch_ms = []
            for size in map_sizes:
                gt = F.interpolate(batch.unsqueeze(0), size=size, mode="bilinear")
                batch_ms.append((gt.squeeze(0) > threshold).float())
            gt_ms.append(batch_ms)

        return gt_ms
    else:
        ndims_gt = len(gt.shape)
        for size in map_sizes:
            if ndims_gt == 5:
                batch_size, t, c, _, _ = gt.shape

                # flatten to 4D
                gt = gt.flatten(start_dim=0, end_dim=1)

                gt = F.interpolate(gt, size=size, mode="bilinear")

                # partition back to original dims if originally 5D
                gt = gt.unsqueeze(1).view(batch_size, t, c, size[0], size[1])

            else:
                gt = F.interpolate(gt, size=size, mode="bilinear")

            gt_ms.append(gt)

        return [(gt > threshold).float() for gt in gt_ms]


def downsample(gt_s1, map_sizes):
    gt_ms = []
    gt = gt_s1
    for size in map_sizes:
        gt = F.interpolate(gt, size=size, mode="bilinear")
        gt_ms.append(gt)

    return gt_ms


def datasets_comparisons(dataloaders):
    """
    compare stats of dataloaders
    """

    stats_dict = MetricDict()
    counter = 0
    for batches in zip(*dataloaders):
        gt = []
        for idx, batch in enumerate(batches):
            _, _, _, gt_maps, _, _ = batch

            gt_maps = gt_maps.cuda()
            gt_s1 = (gt_maps > 0).float()
            gt.append(gt_s1)
            # gt_ms = downsample_gt(gt_s1, map_sizes)

            # Visualise
            # gt_ms = downsample_gt(gt_maps, map_sizes)
            # vis_ms = downsample_gt(vis_mask, map_sizes)
            # gt_map = gt_ms[0][0] * vis_ms[0][0]
            #
            # class_idx = torch.arange(len(gt_map)) + 1
            # gt_vis = gt_map * class_idx.view(-1, 1, 1)
            # gt_vis, _ = gt_vis.max(dim=0)
            # plt.imshow(gt_vis, vmin=0, vmax=12, cmap="magma")
            # plt.title(titles[idx])
            # plt.show()
            #
            # print("wait")

        gt0_v_gt1 = (gt[0] != gt[1]).sum(dim=-1).sum(dim=-1).sum(dim=0)
        counter += 1
        print(counter)
        stats = gt0_v_gt1
        comp_stats = {"stats": stats}
        stats_dict += comp_stats

    print("comparison stats:", stats_dict["stats"])


def project_bev2img(grid2d, calib):
    g = make_grid3d([50, 50], [-25, 1, 1], 1.0).cuda()

    # Convert grid from 2d to 3d
    # grid = grid2d.reshape(-1, 2).T   # reshape from [X, Z, 2] to [2, X * Z]

    print(g[0].max(), g[0].min(), g[2].max(), g[2].min())

    # Project grid coords into image
    print(calib.shape, g.shape)
    uv = torch.matmul(calib, g.double())
    print('before', uv[0].max())
    uv = uv[:-1] / uv[-1]
    print('after', uv[0].max())
    return uv


def bev2img(
        bev, calib, grid_res, image_h, image_w, grid_x=50, grid_z=50,
        grid_x_off=-25, grid_y_off=1, grid_z_off=1
):
    """
    project features from BEV to the image-plane
    Calibration matrices must be scaled accordingly before passing in

    Takes in BEV features [B, C, H, W] and returns them in the image-plane
    in a tensor of size [B, C, image_h, image_w]
    """

    assert len(bev.shape) == 4, "BEV needs to be of shape [B, C, H, W]"
    assert len(calib.shape) == 3, "calib needs to be of shape [B, 3, 3]"

    batch_size = bev.shape[0]

    # Get mapping indices from BEV to Image
    # Create 3D coordinate grid of BEV features
    grid = make_grid3d(
        grid_size=[grid_x, grid_z],
        grid_offset=[grid_x_off, grid_y_off, grid_z_off],
        grid_res=grid_res,
    )  # [3, grid_size]
    grid_as_idxs = make_grid3d(
        grid_size=[bev.shape[-2], bev.shape[-1]],
        grid_offset=[0, 0, 0],
        grid_res=1.0,
    )  # [3, grid_size]
    assert grid.shape == grid_as_idxs.shape, "grid shapes need to match"

    grid = grid.unsqueeze(0).to(bev.device)  # [1, 3, grid_size]
    grid_as_idxs = grid_as_idxs.unsqueeze(0).to(bev.device)  # [1, 3, grid_size]

    # Project grid coordinates to obtain image pixel coordinates
    # [B, 3, 3] * [1, 3, grid_size]
    UV = torch.matmul(calib.float(), grid.float())
    UV = UV[:, :-1] / UV[:, -1].unsqueeze(1)  # [B, 2, grid_size]

    # Mask out everything outside image
    conditions = torch.stack([
        UV[:, 0] >= 0, UV[:, 0] < image_w, UV[:, 1] >= 0, UV[:, 1] < image_h
    ], dim=1)  # [B, 4, grid_size]

    assert conditions.shape[-1] == grid.shape[-1]
    assert len(conditions.shape) == 3

    masks = torch.all(conditions, dim=1)

    # cycle through every set of UV coords in batch, using same grid to get indices
    bev_feats_in_image = []
    pos_z_in_image = []
    for idx in range(batch_size):
        # get uv and BEV coords by appling mask
        uv = UV[idx, :, masks[idx]].long()
        bev_xyz = grid_as_idxs[0, :, masks[idx]].long()

        # Fill in image with BEV features at projected locations
        bev_masked = bev[idx, :, bev_xyz[2], bev_xyz[0]]

        bev_in_im = torch.zeros(
            [bev.shape[1], image_h, image_w], device=bev.device
        )
        pos_z_in_im = torch.zeros(
            [bev.shape[1], image_h, image_w], device=bev.device
        )

        bev_in_im[..., uv[1], uv[0]] = bev_masked
        pos_z_in_im[..., uv[1], uv[0]] = bev_xyz[2].float()

        # similarity = (image_w_bev[..., uv[1], uv[0]] == bev_masked).sum() / \
        #              torch.prod(torch.tensor(bev_masked.shape))

        # assert similarity > 0.5, "similarity = {}".format(similarity)
        bev_feats_in_image.append(bev_in_im)
        pos_z_in_image.append(pos_z_in_im)

    # [B, C, image_h, image_w]
    return torch.stack(bev_feats_in_image), torch.stack(pos_z_in_image)


def dataloader_vis_mask_stats(dataloader, map_sizes, scales=[1, 2, 4, 8, 16]):
    """
    calculate class frequency and class relative size at multiple scales
    """

    stats_dict = MetricDict()
    print("len dataloader:", len(dataloader))
    for i, (_, _, _, _, _, vis_mask) in enumerate(dataloader):
        vis_mask = vis_mask.cuda()
        gt_s1 = (vis_mask > 0.5).float()
        gt_ms = downsample_gt(gt_s1, map_sizes)

        # Class size
        batch_class_size_ms = [gt.sum(dim=-1).sum(dim=-1).sum(dim=0) for gt in gt_ms]
        class_size_ms_keys = ["class_size_s{}".format(scale) for scale in scales]

        batch_length = len(vis_mask)

        keys = [*class_size_ms_keys, "length"]
        vals = [*batch_class_size_ms, batch_length]

        batch_stats = {k: v for k, v in zip(keys, vals)}

        stats_dict += batch_stats
        print(i, batch_length)

    n_cells_ms = torch.tensor(
        [torch.prod(torch.tensor(map_size)) for map_size in map_sizes]
    )

    class_size = torch.stack(
        [stats_dict["class_size_s{}".format(scale)] for scale in scales]
    )

    class_count = stats_dict["length"]
    relative_class_size = class_size / (class_count * n_cells_ms[:, None].cuda())

    print(relative_class_size)
    print("done")


def dataloader_class_stats(dataloader, map_sizes, scales=[1, 2, 4, 8, 16]):
    """
    calculate class frequency and class relative size at multiple scales
    """

    stats_dict = MetricDict()
    print("len dataloader:", len(dataloader))

    for i, (_, _, _, gt_maps, _, _) in enumerate(dataloader):
        gt_maps = gt_maps.cuda()

        gt_s1 = (gt_maps > 0).float()
        gt_ms = downsample_gt(gt_s1, map_sizes)

        # Class frequency
        batch_class_count_ms = [count_classes(gt) for gt in gt_ms]
        class_count_ms_keys = ["class_count_s{}".format(scale) for scale in scales]

        # Class size
        batch_class_size_ms = [gt.sum(dim=-1).sum(dim=-1).sum(dim=0) for gt in gt_ms]
        class_size_ms_keys = ["class_size_s{}".format(scale) for scale in scales]

        batch_length = len(gt_maps)

        keys = [*class_count_ms_keys, *class_size_ms_keys, "length"]
        vals = [*batch_class_count_ms, *batch_class_size_ms, batch_length]

        batch_stats = {k: v for k, v in zip(keys, vals)}

        stats_dict += batch_stats

        if i % 100 == 0:
            print(i)

    n_cells_ms = torch.tensor(
        [torch.prod(torch.tensor(map_size)) for map_size in map_sizes]
    )

    class_count = torch.stack(
        [stats_dict["class_count_s{}".format(scale)] for scale in scales]
    )
    class_size = torch.stack(
        [stats_dict["class_size_s{}".format(scale)] for scale in scales]
    )

    class_frequency = class_count / stats_dict["length"]
    relative_class_size = class_size / (class_count * n_cells_ms[:, None].cuda())

    print(class_frequency)
    print(relative_class_size)
    print("done")


def dataloader_class_stats_segdet(dataloader, map_sizes=[[200, 200]], n_dyn_classes=8):
    """
    calculate class frequency and class relative size at multiple scales
    for the maps and objects dataset
    """
    stats_dict = src.MetricDict()

    dynobj_count = 0
    dynobj_pixcount = 0
    roadobj_count = 0
    roadobj_pixcount = 0
    n_samples = 0

    for (
            i,
            (image, calib, roadmaps, grid2d, vis_masks, objlabels, objboxes, objmaps),
    ) in enumerate(dataloader):

        # Move tensors to GPU
        roadmaps, vis_masks, objlabels, objmaps = (
            roadmaps.cuda(),
            vis_masks.cuda(),
            [labels.cuda() for labels in objlabels],
            [maps.cuda() for maps in objmaps],
        )

        objmaps = [(maps > 0.5).float() for maps in objmaps]
        roadmaps = [(maps > 0.5).float() for maps in roadmaps]
        vis_masks = [(mask > 0).float() for mask in vis_masks]

        # Up/downsample to set size
        objmap_sizes = map_sizes
        objmaps = src.utils.downsample_gt(objmaps, objmap_sizes, batch_list=True)
        roadmaps = src.utils.downsample_gt(roadmaps, objmap_sizes, batch_list=True)
        vis_masks = src.utils.downsample_gt(vis_masks, objmap_sizes, batch_list=True)

        # Mask out occluded areas
        objmaps = [obj[0] * mask[0] for obj, mask in
                   zip(objmaps, vis_masks)]  # only use 200x200
        roadmaps = [road[0] * mask[0] for road, mask in
                    zip(roadmaps, vis_masks)]  # only use 200x200
        roadmaps = torch.stack(roadmaps)

        # Count classes
        obj_onehot = torch.cat(
            [F.one_hot(labels, n_dyn_classes) for labels in objlabels],
            dim=0
        )
        obj_count = torch.stack(
            [(F.one_hot(labels, n_dyn_classes).sum(0) > 0).float() for labels in
             objlabels],
        ).sum(dim=0)
        road_count = (roadmaps.sum(-1).sum(-1) > 0).float().sum(0)

        # Count pixels
        obj_pixcount = torch.cat([maps.sum(-1).sum(-1) for maps in objmaps], dim=0)
        obj_pixcount = obj_onehot * obj_pixcount[:, None]
        obj_pixcount = obj_pixcount.sum(dim=0)
        road_pixcount = roadmaps.sum(-1).sum(-1).sum(0)

        # accumulate
        dynobj_count += obj_count.detach()
        dynobj_pixcount += obj_pixcount.detach()
        roadobj_count += road_count.detach()
        roadobj_pixcount += road_pixcount.detach()
        n_samples += len(image)

        if i % 50 == 0:
            print(i, n_samples)

    # class frequency
    dyn_cf = dynobj_count / n_samples
    road_cf = roadobj_count / n_samples

    # object size as percentage of map size, only 200x200 in this case
    dyn_objsize = dynobj_pixcount / (torch.tensor(map_sizes[0]).prod() * dynobj_count)
    road_objsize = roadobj_pixcount / (
                torch.tensor(map_sizes[0]).prod() * roadobj_count)

    print("Class frequency")
    print("    dynamic objects:", dyn_cf.cpu())
    print("    road objects:", road_cf.cpu())

    print("object size")
    print("    dynamic objects:", dyn_objsize.cpu())
    print("    road objects:", road_objsize.cpu())


def dataloader_token_indices(dataloader, class_idxs):
    """
    Get token indices of images which match conditions for certain classes
    """
    stats_dict = MetricDict()
    print("len dataloader:", len(dataloader))

    tokens_file = os.path.join(
        "/vol/research/sceneEvolution/data/nuscenes/splits", "val_roddick.txt"
    )
    tokens = src.data.nuscenes_obj_dataset.read_split(tokens_file)

    # Make file for selected tokens
    # Clear file if anything exists
    file = os.path.join(
        "/vol/research/sceneEvolution/data/nuscenes/splits",
        "val_pedestrians_beyond_35m.txt",
    )
    open(file, "w").close()

    counta = 0
    for i, (idxs, image, _, gt_maps, _, _, token) in enumerate(dataloader):

        # Move tensors to GPU
        gt_maps = gt_maps.cuda()

        # Check if small distant objects are beyond 35m only
        gt = gt_maps[:, 11].unsqueeze(1)
        pixel_dist = int(35 * 4)

        # First make sure there are objects over 35m
        if gt[:, :, pixel_dist:].sum() > 0:
            # Make sure nothing under 35m
            if gt[:, :, :pixel_dist].sum() == 0:
                # plt.imshow(image[0].permute(1,2,0))
                # plt.grid()
                # plt.show()
                #
                # plt.imshow(gt[0,0].cpu())
                # plt.show()

                # Add token at index
                with open(file, "a") as f:
                    f.write("\n")
                    f.write(token[0])

        counta += 1
        if counta % 200 == 0:
            print(counta, len(tokens))


def compute_iou(pred, labels):
    pred = pred.detach().clone() > 0.5
    labels = labels.detach().clone().bool()

    intersection = pred * labels
    union = pred + labels

    # Calc IoU per class
    iou_per_class = (intersection.float().sum(dim=-1).sum(dim=-1)) / (
            union.float().sum(dim=-1).sum(dim=-1) + 1e-5
    )
    iou_per_class = iou_per_class.sum(dim=0)
    class_count = count_classes(labels)

    # Mean IoU
    iou_mean = iou_per_class.mean()

    iou_dict = {
        "iou_per_class": iou_per_class,
        "class_count": class_count,
    }

    return iou_mean, iou_dict


def get_multiscale_iou(epoch_iou, num_classes):
    s1_iou_seq = (
        (epoch_iou["s1_iou_per_sample"] / (epoch_iou["s1_sample_count"])).cpu().numpy()
    )
    s2_iou_seq = (
        (epoch_iou["s2_iou_per_sample"] / (epoch_iou["s2_sample_count"])).cpu().numpy()
    )
    s4_iou_seq = (
        (epoch_iou["s4_iou_per_sample"] / (epoch_iou["s4_sample_count"])).cpu().numpy()
    )

    s1_new_iou_seq = s1_iou_seq
    s1_new_iou_seq[s1_new_iou_seq != s1_new_iou_seq] = 0
    s1_iou_per_sample = s1_new_iou_seq.sum(axis=1) / num_classes

    s2_new_iou_seq = s2_iou_seq
    s2_new_iou_seq[s2_new_iou_seq != s2_new_iou_seq] = 0
    s2_iou_per_sample = s2_new_iou_seq.sum(axis=1) / num_classes

    s4_new_iou_seq = s4_iou_seq
    s4_new_iou_seq[s4_new_iou_seq != s4_new_iou_seq] = 0
    s4_iou_per_sample = s4_new_iou_seq.sum(axis=1) / num_classes

    return s1_iou_per_sample, s2_iou_per_sample, s4_iou_per_sample


def compute_iou_per_sample(pred, labels, num_classes):
    pred = pred.detach().clone() > 0.5
    labels = labels.detach().clone().bool()

    intersection = pred * labels
    union = pred + labels

    # Calc IoU per class
    iou_per_class = (intersection.float().sum(dim=-1).sum(dim=-1)) / (
            union.float().sum(dim=-1).sum(dim=-1) + 1e-5
    )
    iou_per_class = iou_per_class
    class_count = count_classes_per_sample(labels)

    iou_dict = {
        "iou_per_sample": iou_per_class,
        "sample_count": class_count,
        "iou_per_class": iou_per_class.sum(dim=0),
        "class_count": class_count.sum(dim=0),
    }

    iou_per_sample = iou_per_class / class_count
    iou_per_sample = torch.sum(iou_per_sample, dim=1) / num_classes

    return iou_per_sample, iou_dict


def compute_multiscale_iou(preds, labels, visible_masks, num_classes, threshold=0.5):
    multiscale_iou_per_class = []
    multiscale_class_count = []

    for pred, label, mask in zip(preds, labels, visible_masks):
        assert pred.shape == label.shape
        assert len(pred.shape) == len(mask.shape)

        # Only evaluate in visible areas
        pred = torch.sigmoid(pred.detach().clone()) * mask.float()
        pred = pred > threshold

        label = label.detach().clone() * mask.float()
        label = label > 0.5

        intersection = pred * label
        union = pred + label

        # Calc IoU per class
        iou_per_class = (intersection.float().sum(dim=-1).sum(dim=-1)) / (
                union.float().sum(dim=-1).sum(dim=-1) + 1e-5
        )
        iou_per_class = iou_per_class
        class_count = count_classes_per_sample(label)
        # pred_count = count_classes_per_sample(pred)

        multiscale_iou_per_class.append(iou_per_class)
        multiscale_class_count.append(class_count)

    # Create dict
    scales = [pred.shape[-1] for pred in preds]
    keys_ss = ["iou_per_sample", "sample_count", "iou_per_class", "class_count"]
    keys_ms = ["s{}_".format(scale) + key for scale in scales for key in keys_ss]
    vals_ms = [
        [
            ms_iou_per_class,
            ms_class_count,
            ms_iou_per_class.sum(dim=0),
            ms_class_count.sum(dim=0),
        ]
        for ms_iou_per_class, ms_class_count, ms_iou_per_class, ms_class_count in zip(
            multiscale_iou_per_class,
            multiscale_class_count,
            multiscale_iou_per_class,
            multiscale_class_count,
        )
    ]
    vals_ms = sum(vals_ms, [])
    iou_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    iou_per_sample = multiscale_iou_per_class[0] / (multiscale_class_count[0] + 1e-5)
    iou_per_sample = torch.sum(iou_per_sample, dim=1) / num_classes

    return iou_per_sample, iou_dict


def compute_multiscale_iou_4dims(preds, labels, visible_masks, threshold=0.5):
    multiscale_iou_per_class = []
    multiscale_class_count = []

    for pred, label, mask in zip(preds, labels, visible_masks):
        # Only evaluate in visible areas
        pred = torch.sigmoid(pred.detach()) * mask.float()
        pred = pred > threshold

        label = label.detach() * mask.float()
        label = label > 0.5

        intersection = pred * label
        union = pred + label

        # Calc IoU per class
        iou_per_class = (intersection.float().sum(dim=-1).sum(dim=-1)) / (
                union.float().sum(dim=-1).sum(dim=-1) + 1e-5
        )
        iou_per_class = iou_per_class
        class_count = count_classes_per_sample(label)
        # pred_count = count_classes_per_sample(pred)

        multiscale_iou_per_class.append(iou_per_class)
        multiscale_class_count.append(class_count)

    # Create dict
    scales = [pred.shape[-1] for pred in preds]
    keys_ss = ["iou_static_classes", "count_static_classes"]
    keys_ms = ["s{}_".format(scale) + key for scale in scales for key in keys_ss]
    vals_ms = [
        [ms_iou_per_class.sum(dim=0), ms_class_count.sum(dim=0), ]
        for ms_iou_per_class, ms_class_count in zip(
            multiscale_iou_per_class, multiscale_class_count,
        )
    ]
    vals_ms = sum(vals_ms, [])
    iou_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    return iou_dict


def compute_multiscale_iou_3dims(preds, labels, visible_masks,
                                 metric="iou_dyn_classes"):
    multiscale_iou_per_class = []
    print(metric)
    for pred, label, mask in zip(preds, labels, visible_masks):
        assert (
                len(pred.shape) == len(label.shape) == len(mask.shape)
        ), "IOU masks diff shapes"

        # Only evaluate in visible areas
        pred = torch.sigmoid(pred.detach()) * mask.float()
        pred = pred > 0.5

        label = label.detach() * mask.float()
        label = label > 0.5

        intersection = pred * label
        union = pred + label

        # Calc IoU per class
        iou_per_class = (intersection.float().sum(dim=-1).sum(dim=-1)) / (
                union.float().sum(dim=-1).sum(dim=-1) + 1e-5
        )
        iou_per_class = iou_per_class

        multiscale_iou_per_class.append(iou_per_class)

    # # Create dict
    scales = [pred.shape[-1] for pred in preds]
    keys_ss = [metric]
    keys_ms = ["s{}_".format(scale) + key for scale in scales for key in keys_ss]
    vals_ms = [ms_iou_per_class for ms_iou_per_class in multiscale_iou_per_class]
    # vals_ms = sum(vals_ms, [])
    iou_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    return iou_dict


def compute_multiband_iou(preds, labels, visible_masks, num_classes, threshold=0.5):
    "Compute IoU across the partitioned output"

    multiband_iou_per_class = []
    multiband_class_count = []
    multiband_pred_count = []
    multiscale_false_neg = []
    multiscale_false_pos = []
    multiscale_true_pos = []

    # Partition preds, labels and visible masks
    n_sections = 50
    chunk_size = preds.shape[-2] // n_sections
    part_preds = torch.split(preds, split_size_or_sections=chunk_size, dim=-2)
    part_labels = torch.split(labels, split_size_or_sections=chunk_size, dim=-2)
    part_vis_masks = torch.split(
        visible_masks, split_size_or_sections=chunk_size, dim=-2
    )

    for pred, label, mask in zip(part_preds, part_labels, part_vis_masks):
        # Only evaluate in visible areas
        pred = torch.sigmoid(pred.detach().clone()) * mask.float()
        pred = pred > threshold

        label = label.detach().clone() * mask.float()
        label = label > 0.5

        intersection = pred * label
        union = pred + label

        # Calc IoU per class
        iou_per_class = (intersection.float().sum(dim=-1).sum(dim=-1)) / (
                union.float().sum(dim=-1).sum(dim=-1) + 1e-5
        )
        iou_per_class = iou_per_class
        class_count = count_classes_per_sample(label)
        pred_count = count_classes_per_sample(pred)
        false_neg = (class_count > pred_count).sum(dim=0)
        false_pos = (class_count < pred_count).sum(dim=0)
        true_pos = (class_count * pred_count).sum(dim=0)

        multiband_iou_per_class.append(iou_per_class)
        multiband_class_count.append(class_count)
        multiband_pred_count.append(pred_count)
        multiscale_false_neg.append(false_neg)
        multiscale_false_pos.append(false_pos)
        multiscale_true_pos.append(true_pos)

    # Create dict
    bands = torch.arange(n_sections).int() + 1
    keys_ss = [
        "iou_per_sample",
        "sample_count",
        "iou_per_class",
        "class_count",
        "pred_count",
        "false_neg",
        "false_pos",
        "true_pos",
    ]
    keys_ms = ["d{}_".format(scale) + key for scale in bands for key in keys_ss]
    vals_ms = [
        [
            ms_iou_per_class,
            ms_class_count,
            ms_iou_per_class.sum(dim=0),
            ms_class_count.sum(dim=0),
            ms_pred_count.sum(dim=0),
            ms_false_neg,
            ms_false_pos,
            ms_true_pos,
        ]
        for
        ms_iou_per_class, ms_class_count, ms_iou_per_class, ms_class_count, ms_pred_count, ms_false_neg, ms_false_pos, ms_true_pos
        in zip(
            multiband_iou_per_class,
            multiband_class_count,
            multiband_iou_per_class,
            multiband_class_count,
            multiband_pred_count,
            multiscale_false_neg,
            multiscale_false_pos,
            multiscale_true_pos,
        )
    ]
    vals_ms = sum(vals_ms, [])
    iou_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    iou_per_sample = multiband_iou_per_class[0] / (multiband_class_count[0] + 1e-5)
    iou_per_sample = torch.sum(iou_per_sample, dim=1) / num_classes

    return iou_per_sample, iou_dict


def cart2polar_grid(grid2d):
    # Convert cartesian grid to polar

    r = torch.sqrt(grid2d[..., 0] ** 2 + grid2d[..., 1] ** 2)
    theta = torch.atan(grid2d[..., 1] / (grid2d[..., 0] + 1e-9))

    # _, r_edges, theta_edges = np.histogram2d(
    #     r.numpy().reshape(-1), theta.numpy().reshape(-1), bins=[5, 5]
    # )

    return r, theta


def create_partitioned_polar_grid(grid_h, grid_w, res, nr_bins, ntheta_bins):
    grid2d = make_grid2d([grid_h, grid_w], (-grid_w / 2.0, 0.0), res)

    r, theta = cart2polar_grid(grid2d)

    r_range = [r.min(), r.max()]
    theta_range = [theta.min(), theta.max()]

    r_step = (r_range[1] - r_range[0]) / nr_bins
    r_bins = torch.arange(start=r_range[0], end=r_range[1], step=r_step)
    r_bins = torch.cat(
        [r_bins[None, :], r_range[1].unsqueeze(dim=0)[None, :]], dim=1
    ).reshape(-1)

    theta_step = (theta_range[1] - theta_range[0]) / ntheta_bins
    theta_bins = torch.arange(start=theta_range[0], end=theta_range[1], step=theta_step)
    theta_bins = torch.cat(
        [theta_bins[None, :], theta_range[1].unsqueeze(dim=0)[None, :]], dim=1
    ).reshape(-1)

    # Partition into polar grid and get indices of each partition
    left_r = r_bins[:-1]
    right_r = r_bins[1:]
    left_theta = theta_bins[:-1]
    right_theta = theta_bins[1:]

    lb_theta_r = torch.meshgrid(left_r, left_theta)
    ub_theta_r = torch.meshgrid(right_r, right_theta)

    # Get indices partitioned by polar grid
    polar_idxs = [
        torch.where(
            (
                    (r >= r_lb).float()
                    * (r < r_ub).float()
                    * (theta >= t_lb).float()
                    * (theta < t_ub).float()
            )
            == 1
        )
        for r_lb, r_ub, t_lb, t_ub in zip(
            lb_theta_r[0].flatten(),
            ub_theta_r[0].flatten(),
            lb_theta_r[1].flatten(),
            ub_theta_r[1].flatten(),
        )
    ]
    #
    # for idx_set, set in enumerate(polar_idxs):
    #     plt.scatter(set[1], set[0])
    # plt.show()

    return polar_idxs


def compute_binned_polar_iou(preds, labels, vis_masks, num_classes, polar_idxs):
    "Compute IoU across a partitioned polar spatial grid"

    multiband_iou_per_class = []
    multiband_class_count = []
    multiband_pred_count = []
    multiscale_false_neg = []
    multiscale_true_pos = []

    threshold = 0.5

    # Partition preds, labels and visible masks
    n_sections = len(polar_idxs)
    # chunk_size = preds.shape[-2] // n_sections
    part_preds = [preds[:, :, i[0], i[1]] for i in polar_idxs]
    part_labels = [labels[..., i[0], i[1]] for i in polar_idxs]
    part_vis_masks = [vis_masks[..., i[0], i[1]] for i in polar_idxs]

    for pred, label, mask in zip(part_preds, part_labels, part_vis_masks):
        # Only evaluate in visible areas
        pred = torch.sigmoid(pred.detach().clone()) * mask.float()
        pred = pred > threshold

        label = label.detach().clone() * mask.float()
        label = label > 0.5

        intersection = pred * label
        union = pred + label

        # Calc IoU per class: [N, n_classes, n_points] ----> [N, n_classes]
        iou_per_class = (intersection.float().sum(dim=-1)) / (
                union.float().sum(dim=-1) + 1e-5
        )
        iou_per_class = iou_per_class
        class_count = count_classes_per_sample_1D(label)
        pred_count = count_classes_per_sample_1D(pred)
        false_neg = (class_count > pred_count).sum(dim=0)
        true_pos = (class_count * pred_count).sum(dim=0)

        multiband_iou_per_class.append(iou_per_class)
        multiband_class_count.append(class_count)
        multiband_pred_count.append(pred_count)
        multiscale_false_neg.append(false_neg)
        multiscale_true_pos.append(true_pos)

    # Create dict
    bands = torch.arange(n_sections).int() + 1
    keys_ss = [
        "iou_per_sample",
        "sample_count",
        "iou_per_class",
        "class_count",
        "pred_count",
        "false_neg",
        "true_pos",
    ]
    keys_ms = ["d{}_".format(scale) + key for scale in bands for key in keys_ss]

    vals_ms = [
        [
            ms_iou_per_class,
            ms_class_count,
            ms_iou_per_class.sum(dim=0),
            ms_class_count.sum(dim=0),
            ms_pred_count.sum(dim=0),
            ms_false_neg,
            ms_true_pos,
        ]
        for
        ms_iou_per_class, ms_class_count, ms_iou_per_class, ms_class_count, ms_pred_count, ms_false_neg, ms_true_pos
        in zip(
            multiband_iou_per_class,
            multiband_class_count,
            multiband_iou_per_class,
            multiband_class_count,
            multiband_pred_count,
            multiscale_false_neg,
            multiscale_true_pos,
        )
    ]
    vals_ms = sum(vals_ms, [])
    iou_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    iou_per_sample = multiband_iou_per_class[0] / (multiband_class_count[0] + 1e-5)
    iou_per_sample = torch.sum(iou_per_sample, dim=1) / num_classes

    return iou_per_sample, iou_dict


def compute_multiscale_iou_infer(
        preds, labels, visible_masks, num_classes, threshold=0.5
):
    multiscale_iou_per_class = []
    multiscale_class_count = []
    multiscale_pred_count = []
    multiscale_false_neg = []
    multiscale_true_pos = []

    for pred, label, mask in zip(preds, labels, visible_masks):
        # Only evaluate in visible areas
        pred = torch.sigmoid(pred.detach().clone()) * mask.float()
        pred = pred > threshold

        label = label.detach().clone() * mask.float()
        label = label > 0.5

        intersection = pred * label
        union = pred + label

        # Calc IoU per class
        iou_per_class = (intersection.float().sum(dim=-1).sum(dim=-1)) / (
                union.float().sum(dim=-1).sum(dim=-1) + 1e-5
        )
        iou_per_class = iou_per_class
        class_count = count_classes_per_sample(label)
        pred_count = count_classes_per_sample(pred)
        false_neg = class_count[pred_count == 0].sum()
        true_pos = class_count[pred_count == 1].sum()

        multiscale_iou_per_class.append(iou_per_class)
        multiscale_class_count.append(class_count)
        multiscale_pred_count.append(pred_count)
        multiscale_false_neg.append(false_neg)
        multiscale_true_pos.append(true_pos)

    # Create dict
    scales = [pred.shape[-1] for pred in preds]
    keys_ss = [
        "iou_per_sample",
        "sample_count",
        "iou_per_class",
        "class_count",
        "pred_count",
        "false_neg",
        "true_pos",
    ]
    keys_ms = ["s{}_".format(scale) + key for scale in scales for key in keys_ss]
    vals_ms = [
        [
            ms_iou_per_class,
            ms_class_count,
            ms_iou_per_class.sum(dim=0),
            ms_class_count.sum(dim=0),
            ms_pred_count.sum(dim=0),
            ms_false_neg,
            ms_true_pos,
        ]
        for
        ms_iou_per_class, ms_class_count, ms_iou_per_class, ms_class_count, ms_pred_count, ms_false_neg, ms_true_pos
        in zip(
            multiscale_iou_per_class,
            multiscale_class_count,
            multiscale_iou_per_class,
            multiscale_class_count,
            multiscale_pred_count,
            multiscale_false_neg,
            multiscale_true_pos,
        )
    ]
    vals_ms = sum(vals_ms, [])
    iou_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    iou_per_sample = multiscale_iou_per_class[0] / (multiscale_class_count[0] + 1e-5)
    iou_per_sample = torch.sum(iou_per_sample, dim=1) / num_classes

    return iou_per_sample, iou_dict


def compute_multiscale_iou_wo_mask(preds, labels, num_classes):
    multiscale_iou_per_class = []
    multiscale_class_count = []

    for pred, label in zip(preds, labels):
        # Only evaluate in visible areas
        pred = pred.detach().clone() > 0.5
        label = label.detach().clone().bool()

        intersection = pred * label
        union = pred + label

        # Calc IoU per class
        iou_per_class = (intersection.float().sum(dim=-1).sum(dim=-1)) / (
                union.float().sum(dim=-1).sum(dim=-1) + 1e-5
        )
        iou_per_class = iou_per_class
        class_count = count_classes_per_sample(label)

        multiscale_iou_per_class.append(iou_per_class)
        multiscale_class_count.append(class_count)

    # Create dict
    scales = [2 ** i for i in range(len(preds))]
    keys_ss = ["iou_per_sample", "sample_count", "iou_per_class", "class_count"]
    keys_ms = ["s{}_".format(scale) + key for scale in scales for key in keys_ss]
    vals_ms = [
        [
            ms_iou_per_class,
            ms_class_count,
            ms_iou_per_class.sum(dim=0),
            ms_class_count.sum(dim=0),
        ]
        for ms_iou_per_class, ms_class_count, ms_iou_per_class, ms_class_count in zip(
            multiscale_iou_per_class,
            multiscale_class_count,
            multiscale_iou_per_class,
            multiscale_class_count,
        )
    ]
    vals_ms = sum(vals_ms, [])
    iou_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    iou_per_sample = multiscale_iou_per_class[0] / (multiscale_class_count[0] + 1e-5)
    iou_per_sample = torch.sum(iou_per_sample, dim=1) / num_classes

    return iou_per_sample, iou_dict


def compute_multiscale_loss_per_class(preds, labels):
    multiscale_loss_per_class = []
    multiscale_class_count = []

    for pred, label in zip(preds, labels):
        loss_per_class = dice_loss_per_class(
            pred.detach().clone(), label.detach().clone()
        )
        class_count = count_classes_per_sample(label)

        multiscale_loss_per_class.append(loss_per_class)
        multiscale_class_count.append(class_count)

    # Create dict
    scales = [pred.shape[-1] for pred in preds]
    keys_ss = ["loss_per_class", "class_count"]
    keys_ms = ["s{}_".format(scale) + key for scale in scales for key in keys_ss]
    vals_ms = [
        [ms_loss_per_class, ms_class_count.sum(dim=0)]
        for ms_loss_per_class, ms_class_count in zip(
            multiscale_loss_per_class, multiscale_class_count
        )
    ]
    vals_ms = sum(vals_ms, [])

    loss_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    return loss_dict


def compute_multiscale_bceloss_per_class(preds, labels, loss_per_class):
    multiscale_loss_per_class = []
    multiscale_class_count = []

    for pred, label in zip(preds, labels):
        class_count = count_classes_per_sample(label)

        multiscale_loss_per_class.append(loss_per_class)
        multiscale_class_count.append(class_count)

    # Create dict
    scales = [pred.shape[-1] for pred in preds]
    keys_ss = ["loss_per_class", "class_count"]
    keys_ms = ["s{}_".format(scale) + key for scale in scales for key in keys_ss]
    vals_ms = [
        [ms_loss_per_class, ms_class_count.sum(dim=0)]
        for ms_loss_per_class, ms_class_count in zip(
            multiscale_loss_per_class, multiscale_class_count
        )
    ]
    vals_ms = sum(vals_ms, [])

    loss_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    return loss_dict


def compute_multiscale_loss_per_class_infer(preds, labels):
    multiscale_loss_per_class = []
    multiscale_inter_per_class = []
    multiscale_union_per_class = []
    multiscale_class_count = []

    for pred, label in zip(preds, labels):
        loss_per_class, inter, union = dice_loss_per_class_infer(
            pred.detach().clone(), label.detach().clone()
        )
        class_count = count_classes_per_sample(label)

        multiscale_loss_per_class.append(loss_per_class)
        multiscale_inter_per_class.append(inter)
        multiscale_union_per_class.append(union)
        multiscale_class_count.append(class_count)

    # Create dict
    scales = [pred.shape[-1] for pred in preds]
    keys_ss = ["loss_per_class", "inter_per_class", "union_per_class", "class_count"]
    keys_ms = ["s{}_".format(scale) + key for scale in scales for key in keys_ss]
    vals_ms = [
        [
            ms_loss_per_class,
            ms_inter_per_class,
            ms_union_per_class,
            ms_class_count.sum(dim=0),
        ]
        for ms_loss_per_class, ms_inter_per_class, ms_union_per_class, ms_class_count in
        zip(
            multiscale_loss_per_class,
            multiscale_inter_per_class,
            multiscale_union_per_class,
            multiscale_class_count,
        )
    ]
    vals_ms = sum(vals_ms, [])

    loss_dict = {k: v for k, v in zip(keys_ms, vals_ms)}

    return loss_dict


def count_classes(labels):
    class_count = labels.sum(dim=-1).sum(dim=-1)
    class_count[class_count > 0] = 1
    class_count = class_count.sum(dim=0)
    return class_count


def count_classes_per_sample(labels):
    class_count = labels.sum(dim=-1).sum(dim=-1)
    class_count[class_count > 0] = 1
    return class_count


def count_classes_per_sample_1D(labels):
    class_count = labels.sum(dim=-1)
    class_count[class_count > 0] = 1
    return class_count


def rotate(vector, angle):
    """
    Rotate a vector around the y-axis
    """
    sinA, cosA = torch.sin(angle), torch.cos(angle)
    xvals = cosA * vector[..., 0] + sinA * vector[..., 2]
    yvals = vector[..., 1]
    zvals = -sinA * vector[..., 0] + cosA * vector[..., 2]
    return torch.stack([xvals, yvals, zvals], dim=-1)


def perspective(matrix, vector):
    """
    Applies perspective projection to a vector using projection matrix
    """
    vector = vector.unsqueeze(-1)
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


def perspective2(matrix, vector):
    """
    Applies perspective projection to a vector using projection matrix
    """
    vector = torch.cat([vector, torch.ones_like(vector[:, 0:1])], dim=-1)
    print(vector.shape)
    homogenous = torch.matmul(vector, matrix.T)
    # homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


def make_grid2d(grid_size, grid_offset, grid_res):
    """
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, zoff = grid_offset
    xcoords = torch.arange(0.0, width, grid_res) + xoff
    zcoords = torch.arange(0.0, depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, zz], dim=-1)


def make_grid3d(grid_size, grid_offset, grid_res):
    """
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0.0, width, grid_res) + xoff
    zcoords = torch.arange(0.0, depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack(
        [xx.reshape(-1), torch.full_like(xx, yoff).reshape(-1), zz.reshape(-1)]
    )


def make_uvcoords(width, height):
    """
    Construct an array representing the UV coordinates of the image
    """
    xcoords = torch.arange(0.0, width)
    ycoords = torch.arange(0.0, height)

    yy, xx = torch.meshgrid(ycoords, xcoords)
    return torch.stack([xx, yy, torch.ones_like(xx)])


def gaussian_kernel(sigma=1.0, trunc=2.0):
    width = round(trunc * sigma)
    x = torch.arange(-width, width + 1).float() / sigma
    kernel1d = torch.exp(-0.5 * x ** 2)
    kernel2d = kernel1d.view(1, -1) * kernel1d.view(-1, 1)

    return kernel2d / kernel2d.sum()


def find_rows(a, b):
    """ Find matching rows between two arrays"""
    dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))

    a_view = np.ascontiguousarray(a).view(dt).ravel()
    b_view = np.ascontiguousarray(b).view(dt).ravel()

    sort_b = np.argsort(b_view)
    where_in_b = np.searchsorted(b_view, a_view, sorter=sort_b)
    where_in_b = np.take(sort_b, where_in_b)
    which_in_a = np.take(b_view, where_in_b) == a_view
    where_in_b = where_in_b[which_in_a]
    which_in_a = np.nonzero(which_in_a)[0]
    return np.column_stack((which_in_a, where_in_b))


def bbox_corners(obj):
    """
    Return the 2D
    """

    # Get corners of bounding box in object space
    offsets = torch.tensor(
        [
            [-0.5, 0.0, -0.5],  # Back-left lower
            [0.5, 0.0, -0.5],  # Front-left lower
            [-0.5, 0.0, 0.5],  # Back-right lower
            [0.5, 0.0, 0.5],  # Front-right lower
            [-0.5, -1.0, -0.5],  # Back-left upper
            [0.5, -1.0, -0.5],  # Front-left upper
            [-0.5, -1.0, 0.5],  # Back-right upper
            [0.5, -1.0, 0.5],  # Front-right upper
        ]
    )
    corners = offsets * torch.tensor(obj.dimensions)
    # corners = corners[:, [2, 0, 1]]

    # Apply y-axis rotation
    corners = rotate(corners, torch.tensor(obj.angle))

    # Apply translation
    corners = corners + torch.tensor(obj.position)
    return corners


def quaternion_yaw2(q: Quaternion, up_vector: np.array) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xz plane.
    v = np.dot(q.rotation_matrix, np.array(up_vector))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[0], v[2])

    return yaw


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xz plane.
    v = np.dot(q.rotation_matrix, np.array([0, 1, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[0], v[2])

    return yaw


def build_z_intervals(f, grid_res, scales=[8.0, 16.0, 32.0, 64.0]):
    z = [float(int(f * grid_res / s)) for s in scales]
    z.append(1.0)
    return z


def calc_cropped_heights(f, y_total, z_intervals, scales):
    # Reverse scales so theyre in same order as z_intervals
    scales = np.array(scales)[::-1]
    u1 = f * y_total / np.array(z_intervals)[1:]
    u_scaled = u1 / (scales * 0.5)
    # print(f, y_total, z_intervals, scales)

    return np.flip(np.rint(u_scaled))


def collate(batch):
    """ The augmentation results in images of different sizes, so
    this collate function pads them all to [1242, 375]"""

    idxs, images, calibs, objects, grids2d, grids3d = zip(*batch)

    # Pad images to the same dimensions
    w = [img.size[0] for img in images]
    h = [img.size[1] for img in images]

    desired_w, desired_h = 1242, 375
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
    calibs = torch.stack(calibs)
    grids2d = torch.stack(grids2d)
    grids3d = torch.stack(grids3d)

    # Modify calibration matrices
    # cx' = cx - du
    calibs[:, 0, 2] = calibs[:, 0, 2] + torch.tensor(pad_left)
    # cy' = cy - dv
    calibs[:, 1, 2] = calibs[:, 1, 2] + torch.tensor(pad_top)
    # tx' = tx - du * tz
    calibs[:, 0, 3] = calibs[:, 0, 3] + torch.tensor(pad_left) * calibs[:, 2, 3]
    # ty' = ty - dv * tz
    calibs[:, 1, 3] = calibs[:, 1, 3] + torch.tensor(pad_top) * calibs[:, 2, 3]

    # plt.imshow(images[0])
    # plt.scatter(calibs[0,0,2], calibs[0,1,2], s=20, c='b')
    # plt.show()

    # Create a vector of indices
    idxs = torch.LongTensor(idxs)

    return idxs, images, calibs, objects, grids2d, grids3d


def assign_points_voxels(points):
    """ Assign points to voxels on a grid by rounding the
    coordinates and then using them as indices to populate
    the points of the grid """

    batch_size = len(points)

    # Create 3D voxel grid
    grid_y_max, grid_y_min, grid_y_res = 2.0, 0, 0.4
    grid_x_max, grid_x_min, grid_x_res = 50.0, 0, 1.0
    grid_z_max, grid_z_min, grid_z_res = 50.0, 0, 1.0
    num_vox_y = int((grid_y_max - grid_y_min) / grid_y_res)
    num_vox_x = int((grid_x_max - grid_x_min) / grid_x_res)
    num_vox_z = int((grid_z_max - grid_z_min) / grid_z_res)
    voxels = torch.zeros((batch_size, num_vox_z, num_vox_x))

    # Shift points along X dimension to fit on grid
    for i in range(len(points)):
        points[i][0] += grid_x_max / 2

    # Round coordinates to integers
    points_int = [(points[i] // grid_x_res).int().long() for i in range(batch_size)]

    for i in range(len(points)):
        voxels[i, points_int[i][2], points_int[i][0]] += 1.0

    # plt.imshow(voxels[0].cpu())
    # plt.show()

    return voxels


def assign_points_voxels2(points):
    """ Assign points to voxels on a grid by rounding the
    coordinates and then using them as indices to populate
    the points of the grid """

    batch_size = len(points)

    crop_xmax = points[0] < 25
    crop_xmin = points[0] > -25
    crop_zmax = points[2] < 50
    crop_zmin = points[2] > 0
    crop_pts_mask = crop_xmax & crop_xmin & crop_zmax & crop_zmin
    points = points[:, crop_pts_mask]

    # Create 3D voxel grid
    grid_x_max, grid_x_min, grid_x_res = 50.0, 0, 1.0
    grid_z_max, grid_z_min, grid_z_res = 50.0, 0, 1.0
    num_vox_x = int((grid_x_max - grid_x_min) / grid_x_res)
    num_vox_z = int((grid_z_max - grid_z_min) / grid_z_res)
    vox = torch.zeros((num_vox_z, num_vox_x))

    # Shift points along X dimension to fit on grid
    points[0] += grid_x_max / 2

    # Round coordinates to integers
    points_int = (points // grid_x_res).int().long()

    # print(points_int[2].max(), points_int[0].max())

    vox[points_int[2], points_int[0]] += 1.0

    # plt.imshow(voxels[0].cpu())
    # plt.show()

    return vox


def merge_nusc_static_classes(gt, idxs2merge, idx_merged):
    """
    Takes ground truth maps of shape [N, H, W]
    and merges the indices given to produce
    a resultant tensor of shape [N - len(idx_merge), H, W].

    The indices provided must be consecutive
    """
    gt2merge = torch.stack([gt[i] for i in idxs2merge])
    gt_merged = gt2merge.sum(dim=0)
    gt_merged = (gt_merged > 0).float()

    gt_new = torch.cat(
        [
            gt_merged.unsqueeze(0),
            gt[idx_merged + 1: idxs2merge[1]],
            gt[idxs2merge[-1] + 1:],
        ],
        dim=0,
    )
    return gt_new


def merge_classes2(gt, idxs2merge, idx_merged):
    gt2merge = torch.stack([gt[i] for i in idxs2merge])
    gt_merged = gt2merge.sum(dim=0)
    gt_merged = (gt_merged > 0).float()

    # exclude idx
    idx_exclude = (
            torch.tensor(idxs2merge)[:, None]
            != torch.tensor(idx_merged).unsqueeze(0)[None, :]
    )
    idx_exclude = torch.tensor(idxs2merge)[torch.any(idx_exclude, dim=1)]

    idxs_all = torch.arange(len(gt))
    gt_new = gt[idxs_all != idx_exclude]
    gt_new[idx_merged] = gt_merged

    return gt_new


def convert_figure(fig, tight=True):
    # Converts a matplotlib figure into a numpy array

    # Draw plot
    if tight:
        fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Convert figure to numpy via a string array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def gaussian_smoothing(input, sigma, channels, kernel_size=13, padding=6, dim=2):
    """

    :param input: Input tensor to be convolved with Gaussian kernel
    :param channels: Num of channels of input tensor, output will have
    same number of channels
    :param kernel_size (int, sequence): Size of Gaussian kernel
    :param sigma (int, sequence): std of gaussian kernel
    :param dim: Number of dimensions of data
    :return: Convolved output tensor with same dim as input tensor
    """

    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
        )

    # Make sure sure kernel sums to 1
    kernel = kernel / torch.sum(kernel)

    kernel = kernel.view(1, 1, *kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    # filter = nn.Conv2d(in_channels=channels, out_channels=channels,
    #                    kernel_size=kernel_size, groups=channels)
    # filter.weight.data = kernel
    # filter.weight.requires_grad = False

    return F.conv2d(input, kernel, padding=padding, groups=channels)


def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)

    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)


def bresenhamline(start, end, max_iter=3):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.
        """

    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


def connect2(ends):
    # # Modify array if any column has only zeros (for arange calculation later)
    if np.any(ends.sum(axis=0) == 0):
        ends = np.diag(ends.shape)

    d0, d1 = np.diff(ends, axis=0)[0]
    if np.abs(d0) > np.abs(d1):
        return np.c_[
            np.arange(
                ends[0, 0], ends[1, 0] + np.sign(d0), np.sign(d0), dtype=np.int32
            ),
            np.arange(
                ends[0, 1] * np.abs(d0) + np.abs(d0) // 2,
                ends[0, 1] * np.abs(d0) + np.abs(d0) // 2 + (np.abs(d0) + 1) * d1,
                d1,
                dtype=np.int32,
            )
            // np.abs(d0),
        ]
    else:
        return np.c_[
            np.arange(
                ends[0, 0] * np.abs(d1) + np.abs(d1) // 2,
                ends[0, 0] * np.abs(d1) + np.abs(d1) // 2 + (np.abs(d1) + 1) * d0,
                d0,
                dtype=np.int32,
            )
            // np.abs(d1),
            np.arange(
                ends[0, 1], ends[1, 1] + np.sign(d1), np.sign(d1), dtype=np.int32
            ),
        ]


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        raise ValueError
    return (x / z, y / z)


def ccw(A, B, C):
    """
    A: [x,y]
    B: [x,y]
    C: [x,y]
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def check_intersect(A, B, C, D):
    """
    # Return true if line segments AB and CD intersect

    A: [x,y]
    B: [x,y]
    C: [x,y]
    D: [x,y]
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def crude_depth_estimator(bbox_2d, classes, corners_3d, calibs):
    assert len(classes[0].shape) == 1
    assert len(calibs.shape) == 3
    print("classes", classes)

    f_x = calibs[:, 0, 0]
    obj_average_heights = torch.tensor(
        src.data.nuscenes_obj_dataset.average_obj_heights, device=classes[0].device
    )
    objheight_w = [
        torch.ones_like(cls) * obj_average_heights[cls] for cls in classes
    ]
    objheights_i = [torch.abs(box[:, 3] - box[:, 1]) for box in bbox_2d]

    scale = [height_i / height_w for height_i, height_w in
             zip(objheights_i, objheight_w)]

    objdepth = [f * s * 1e-4 for f, s in zip(f_x, scale)]
    return objdepth


def xyxy_to_cxcy(bbox_2d, mode="center"):
    if mode == "center":
        image_cx = [
            torch.clip(torch.stack([box[:, 2], box[:, 0]]).mean(dim=0), min=0, max=1600)
            for box in bbox_2d
        ]
        image_cy = [
            torch.clip(torch.stack([box[:, 1], box[:, 3]]).mean(dim=0), min=0, max=900)
            for box in bbox_2d
        ]
    if mode == "bottom":
        image_cx = [
            torch.clip(torch.stack([box[:, 2], box[:, 0]]).mean(dim=0), min=0, max=1600)
            for box in bbox_2d
        ]
        image_cy = [
            torch.clip(torch.stack([box[:, 1], box[:, 3]]).min(dim=0)[0], min=0,
                       max=900)
            for box in bbox_2d
        ]
    graph_maker_pos = [torch.stack([cx, cy]).T for cx, cy in zip(image_cx, image_cy)]
    return graph_maker_pos


def xyxy_to_pts_along_vanishing_line(args, bbox_2d, calibs, images):
    pos = xyxy_to_cxcy(bbox_2d, mode=args.vanishing_line_mode)
    # vanishing point
    v_x, v_y = calibs[:, 0, 2], calibs[:, 1, 2]
    v_pt = torch.stack([v_x, v_y]).T
    # center vanishing line
    i_center_bottom_pt = torch.stack(
        [
            torch.ones_like(v_pt[:, 0]) * images.shape[-1] * 0.5,
            # get mid width of image
            torch.zeros_like(v_pt[:, 0])
        ]
    ).T
    v_cl = v_pt - i_center_bottom_pt
    # standardize to [0,1], origin bottom left of image
    y_max = images.shape[-2]
    x_max = images.shape[-1]
    scaler = torch.tensor([x_max, y_max], device=v_cl.device)
    pos = [p / scaler.unsqueeze(0) for p in pos]
    v_pt = v_pt / scaler.unsqueeze(0)
    v_cl = v_cl / scaler.unsqueeze(0)
    # object vanishing line (from object to optical center)
    obj_v_l = [v.unsqueeze(0) - p for v, p in zip(v_pt, pos)]
    # project object vanishing line onto center vanishing line
    proj_obj_v_l = [
        o_v_l.matmul(v_l.unsqueeze(-1)) / torch.linalg.norm(v_l) for o_v_l, v_l in
        zip(obj_v_l, v_cl)
    ]
    # subtract from 0 so objects closer to the vanishing point are larger
    proj_obj_v_l = [0 - p for p in proj_obj_v_l]
    pos = [torch.stack([p[:, 0], proj.squeeze(-1)]).T for p, proj in
           zip(pos, proj_obj_v_l)]
    return pos


def evaluate_iou_with_gt_obj(
        args, classes, gt_depths, gt_x_pos, gt_corners_3d,
        mapsdict, num_classes, obj_per_batch, preds
):
    # Calculate IoU using actual object dimensions
    # convert object centers to BEV maps with box at object center
    pred_centers = torch.cat([gt_x_pos, preds.detach()], dim=1)  # [n_obj, 2]
    gt_centers = torch.cat([gt_x_pos, gt_depths], dim=1)  # [n_obj, 2]

    # create objects at their respective centers
    assert gt_corners_3d[0].shape[-1] == 8
    assert gt_corners_3d[0][:, :, [0, 1, 4, 5]].shape[-1] == 4

    corners_xz = torch.cat(
        [torch.stack(
            [(corners[:, 0, [2, 3, 7, 6]] + 25) / 50, corners[:, 2, [2, 3, 7, 6]] / 50]
        ).permute(1, 0, 2)
         for corners in gt_corners_3d],
        dim=0
    )  # [n_obj, 2, 4]

    o_corners_xz = corners_xz - gt_centers.unsqueeze(-1)
    pred_corners = o_corners_xz + pred_centers.unsqueeze(-1)

    pred_maps = objcorners_to_classmaps(args, classes, num_classes, obj_per_batch,
                                        pred_corners)
    gt_maps = objcorners_to_classmaps(args, classes, num_classes, obj_per_batch,
                                      corners_xz)

    # plt.imshow(gt_maps[0].sum(0).cpu())
    # plt.show()
    #
    # plt.imshow(pred_maps[0].sum(0).cpu())
    # plt.show()
    #
    # maps = torch.stack([map[0] for name,map in mapsdict[0].items() if "fov_mask" not in name])
    # plt.imshow(maps.sum(0).cpu())
    # plt.show()

    # Calculate IoU between predicted and gt maps
    vis_masks = torch.stack([maps["lidar_ray_mask_dense"] for maps in mapsdict])
    assert pred_maps.shape == gt_maps.shape
    assert vis_masks.dim() == pred_maps.dim()
    _, iou_dict = src.utils.compute_multiscale_iou(
        [pred_maps.detach()], [gt_maps.detach()], [vis_masks.detach()], num_classes
    )
    return iou_dict


def objcorners_to_classmaps(args, classes, num_classes, obj_per_batch, pred_corners):
    # Create mask for object
    obj_masks = []
    for pts in pred_corners:
        pts = pts.cpu().numpy()
        pts = pts * 200
        poly_mask = np.zeros([200, 200], np.uint8)
        pts_list = [
            (p0, p1) for (p0, p1) in zip(pts[0], pts[1])
        ]
        poly = Polygon(pts_list)
        exteriors = [int_coords(poly.exterior.coords)]
        interiors = [int_coords(pi.coords) for pi in poly.interiors]
        offset = (int(0), int(0))
        cv2.fillPoly(poly_mask, exteriors, 1, offset=offset)
        cv2.fillPoly(poly_mask, interiors, 0, offset=offset)
        obj_masks.append(poly_mask)

    # partition tensors by batch
    obj_masks = torch.split(torch.tensor(obj_masks).cuda(), obj_per_batch)
    objclasses = torch.split(classes, obj_per_batch)
    # Accumulate objects onto maps by class
    class_maps = torch.zeros((args.batch_size, num_classes, 200, 200)).cuda()
    for idx_batch, (masks, cls) in enumerate(zip(obj_masks, objclasses)):
        for idx_obj, mask in enumerate(masks):
            class_maps[idx_batch, cls[idx_obj]] += mask
    return class_maps


def objcenters_to_classmaps(args, classes, mapsdict, num_classes, obj_per_batch,
                            pred_center):
    """
    Turns object (x,z) centers in BEV to class maps (per class), using a basic object shape prior
    :param args:
    :param classes:
    :param mapsdict:
    :param num_classes:
    :param obj_per_batch:
    :param pred_center:
    :return:
    """

    obj_w, obj_l = 2 / 50, 3 / 50
    obj_cxcywh = torch.cat([
        pred_center,
        torch.full_like(pred_center[:, 0:1], obj_w),
        torch.full_like(pred_center[:, 0:1], obj_l)
    ], dim=1)
    objboxes_xyxy = torch.stack([box_cxcywh_to_xyxy(x) for x in obj_cxcywh])

    # partition tensors by batch
    objboxes_xyxy = torch.split(objboxes_xyxy, obj_per_batch)
    objclasses = torch.split(classes, obj_per_batch)

    # Accumulate objects onto maps by class
    class_maps = torch.zeros((args.batch_size, num_classes, 200, 200)).cuda()
    for idx_batch, (boxes, cls) in enumerate(zip(objboxes_xyxy, objclasses)):
        for idx_obj, box in enumerate(boxes):
            box_scaled = box * 200
            x0, z0, x1, z1 = box_scaled.int().unbind(-1)
            class_maps[idx_batch, cls[idx_obj], z0:z1, x0:x1] = 1

    # Check visually
    # pred_centers_bev = torch.split(pred_center, obj_per_batch)
    # for map, centers, gts in zip(class_maps, pred_centers_bev, mapsdict):
    #     plt.imshow(map.sum(0).cpu() + gts["fov_mask"][0])
    #     plt.scatter(centers[:, 0].cpu() * 200, centers[:, 1].cpu() * 200, s=20)
    #     plt.show()
    #
    #     gtmaps = torch.cat([map for map in gts.values()], dim=0).sum(0)
    #     plt.imshow(gtmaps)
    #     plt.show()

    return class_maps


def int_coords(x):
    # function to round and convert to int
    return np.array(x).round().astype(np.int32)


def rotation_matrix(yaws, clockwise=True):
    if clockwise:
        rot_matrix = torch.stack([torch.tensor(
            [[torch.cos(yaw), torch.sin(yaw)],
             [-torch.sin(yaw), torch.cos(yaw)]], device=yaws.device)
            for yaw in yaws
        ])
    else:
        rot_matrix = torch.stack([torch.tensor(
            [[torch.cos(yaw), -torch.sin(yaw)],
             [torch.sin(yaw), torch.cos(yaw)]], device=yaws.device)
            for yaw in yaws
        ])
    return rot_matrix