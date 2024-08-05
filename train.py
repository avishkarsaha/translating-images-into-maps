import os
import time

import json
import math
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as font_manager

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

from datetime import datetime, timedelta
from argparse import ArgumentParser

from collections import defaultdict
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from yacs.config import CfgNode

import src
import src.data.collate_funcs
from src import utils
from src.utils import MetricDict
from src.data.dataloader import nuScenesMaps
import src.model.network as networks
from src.data.dataloader_new import build_dataloaders

mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf")
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True

def train(args, dataloader, model, optimizer, epoch):
    print("\n==> Training on {} minibatches".format(len(dataloader)))
    model.train()
    epoch_loss = MetricDict()
    epoch_loss_per_class = MetricDict()
    batch_acc_loss = MetricDict()
    epoch_iou = MetricDict()
    t = time.time()
    num_classes = len(args.pred_classes_nusc)

    for i, (image, calib, cls_map, vis_mask) in enumerate(dataloader):

        grid2d = utils.make_grid2d(args.grid_size, (-args.grid_size[0] / 2.0, 0.0), args.grid_res)
        batch_size = len(calib)
        grid2d = grid2d.repeat((batch_size,)+(1,)*len(grid2d.shape))
        vis_mask = vis_mask.repeat((batch_size,)+(1,)*len(vis_mask.shape))
        # Move tensors to GPU
        image, calib, cls_map, vis_mask, grid2d = (
            image.cuda(),
            calib.cuda(),
            cls_map.cuda(),
            vis_mask.cuda(),
            grid2d.cuda(),
        )

        # Run network forwards
        pred_ms = model(image, calib, grid2d)

        # Convert ground truths to binary mask
        gt_s1 = (cls_map > 0).float()
        visibility_mask_s1 = (vis_mask > 0).float()

        # Downsample to match model outputs
        map_sizes = [pred.shape[-2:] for pred in pred_ms]
        gt_ms = src.utils.downsample_gt(gt_s1, map_sizes)
        vis_ms = src.utils.downsample_gt(visibility_mask_s1, map_sizes)

        # Compute losses for backprop
        loss, loss_dict = compute_loss(pred_ms, gt_ms, args.loss, args)

        # Calculate gradients
        loss.backward()

        # Compute IoU
        iou_per_sample, iou_dict = src.utils.compute_multiscale_iou(
            pred_ms, gt_ms, vis_ms, num_classes
        )
        # Compute per class loss for eval
        per_class_loss_dict = src.utils.compute_multiscale_loss_per_class(
            pred_ms, gt_ms,
        )

        if float(loss) != float(loss):
            raise RuntimeError("Loss diverged :(")

        epoch_loss += loss_dict
        epoch_loss_per_class += per_class_loss_dict
        batch_acc_loss += loss_dict
        epoch_iou += iou_dict

        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Print summary
            batch_time = (time.time() - t) / (1 if i == 0 else args.print_iter)
            eta = ((args.epochs - epoch + 1) * len(dataloader) - i) * batch_time

            s = "[Epoch: {} {:4d}/{:4d}] batch_time: {:.2f}s eta: {:s} loss: ".format(
                epoch, i, len(dataloader), batch_time, str(timedelta(seconds=int(eta)))
            )
            for k, v in batch_acc_loss.mean.items():
                s += "{}: {:.2e} ".format(k, v)

            with open(os.path.join(args.savedir, args.name, "output.txt"), "a") as fp:
                fp.write(s)
            print(s)
            t = time.time()

            batch_acc_loss = MetricDict()

    # Calculate per class IoUs over set
    scales = [pred.shape[-1] for pred in pred_ms]
    ms_cumsum_iou_per_class = torch.stack(
        [epoch_iou["s{}_iou_per_class".format(scale)] for scale in scales]
    )
    ms_count_per_class = torch.stack(
        [epoch_iou["s{}_class_count".format(scale)] for scale in scales]
    )
    ms_ious_per_class = (
        (ms_cumsum_iou_per_class / (ms_count_per_class + 1)).cpu().numpy()
    )
    ms_mean_iou = ms_ious_per_class.mean(axis=1)

    # Calculate per class loss over set
    ms_cumsum_loss_per_class = torch.stack(
        [epoch_loss_per_class["s{}_loss_per_class".format(scale)] for scale in scales]
    )
    ms_loss_per_class = (
        (ms_cumsum_loss_per_class / (ms_count_per_class + 1)).cpu().numpy()
    )
    total_loss = ms_loss_per_class.mean(axis=1).sum()

    # Print epoch summary and save results
    print("==> Training epoch complete")
    for key, value in epoch_loss.mean.items():
        print("{:8s}: {:.4e}".format(key, value))

    with open(os.path.join(args.savedir, args.name, "train_loss.txt"), "a") as f:
        f.write("\n")
        f.write(
            "{},".format(epoch)
            + "{},".format(float(total_loss))
            + "".join("{},".format(v) for v in ms_mean_iou)
        )
    with open(os.path.join(args.savedir, args.name, "train_ious.txt"), "a") as f:
        f.write("\n")
        f.write(
            "Epoch: {}, \n".format(epoch)
            + "Total Loss: {}, \n".format(float(total_loss))
            + "".join(
                "s{}_ious_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_ious_per_class)
            )
            + "".join(
                "s{}_loss_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_loss_per_class)
            )
        )


def validate(args, dataloader, model, epoch):
    print("\n==> Validating on {} minibatches\n".format(len(dataloader)))
    model.eval()
    epoch_loss = MetricDict()
    epoch_iou = MetricDict()
    epoch_loss_per_class = MetricDict()
    num_classes = len(args.pred_classes_nusc)
    times = []

    grid2d = utils.make_grid2d(args.grid_size, (-args.grid_size[0] / 2.0, 0.0), args.grid_res)

    for i, (image, calib, cls_map, vis_mask) in enumerate(dataloader):
        batch_size = len(calib)
        grid2d = grid2d.repeat((batch_size,)+(1,)*len(grid2d.shape))
        vis_mask = vis_mask.repeat((batch_size,)+(1,)*len(vis_mask.shape))

        # Move tensors to GPU
        image, calib, cls_map, vis_mask, grid2d = (
            image.cuda(),
            calib.cuda(),
            cls_map.cuda(),
            vis_mask.cuda(),
            grid2d.cuda(),
        )

        with torch.no_grad():

            # Run network forwards
            pred_ms = model(image, calib, grid2d)

            # Upsample largest prediction to 200x200
            pred_200x200 = F.interpolate(
                pred_ms[0], size=(200, 200), mode="bilinear"
            )
            # pred_200x200 = (pred_200x200 > 0).float()
            pred_ms = [pred_200x200, *pred_ms]

            # Get required gt output sizes
            map_sizes = [pred.shape[-2:] for pred in pred_ms]

            # Convert ground truth to binary mask
            gt_s1 = (cls_map > 0).float()
            vis_mask_s1 = (vis_mask > 0.5).float()

            # Downsample to match model outputs
            gt_ms = src.utils.downsample_gt(gt_s1, map_sizes)
            vis_ms = src.utils.downsample_gt(vis_mask_s1, map_sizes)

            # Compute IoU
            iou_per_sample, iou_dict = src.utils.compute_multiscale_iou(
                pred_ms, gt_ms, vis_ms, num_classes
            )
            # Compute per class loss for eval
            per_class_loss_dict = src.utils.compute_multiscale_loss_per_class(
                pred_ms, gt_ms,
            )

            epoch_iou += iou_dict
            epoch_loss_per_class += per_class_loss_dict

            # # Visualize predictions
            # if epoch % args.val_interval * 4 == 0 and i % 50 == 0:
            #     vis_img = ToPILImage()(image[0].detach().cpu())
            #     pred_vis = pred_ms[1].detach().cpu()
            #     label_vis = gt_ms[1]
            #
            #     # Visualize scores
            #     vis_fig = visualize_score(
            #         pred_vis[0],
            #         label_vis[0],
            #         grid2d[0],
            #         vis_img,
            #         iou_per_sample[0],
            #         num_classes,
            #     )
            #     plt.savefig(
            #         os.path.join(
            #             args.savedir,
            #             args.name,
            #             "val_output_epoch{}_iter{}.png".format(epoch, i),
            #         )
            #     )

    print("\n==> Validation epoch complete")

    # Calculate per class IoUs over set
    scales = [pred.shape[-1] for pred in pred_ms]

    ms_cumsum_iou_per_class = torch.stack(
        [epoch_iou["s{}_iou_per_class".format(scale)] for scale in scales]
    )
    ms_count_per_class = torch.stack(
        [epoch_iou["s{}_class_count".format(scale)] for scale in scales]
    )

    ms_ious_per_class = (
        (ms_cumsum_iou_per_class / (ms_count_per_class + 1e-6)).cpu().numpy()
    )
    ms_mean_iou = ms_ious_per_class.mean(axis=1)

    # Calculate per class loss over set
    ms_cumsum_loss_per_class = torch.stack(
        [epoch_loss_per_class["s{}_loss_per_class".format(scale)] for scale in scales]
    )
    ms_loss_per_class = (
        (ms_cumsum_loss_per_class / (ms_count_per_class + 1)).cpu().numpy()
    )
    total_loss = ms_loss_per_class.mean(axis=1).sum()

    with open(os.path.join(args.savedir, args.name, "val_loss.txt"), "a") as f:
        f.write("\n")
        f.write(
            "{},".format(epoch)
            + "{},".format(float(total_loss))
            + "".join("{},".format(v) for v in ms_mean_iou)
        )

    with open(os.path.join(args.savedir, args.name, "val_ious.txt"), "a") as f:
        f.write("\n")
        f.write(
            "Epoch: {},\n".format(epoch)
            + "Total Loss: {},\n".format(float(total_loss))
            + "".join(
                "s{}_ious_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_ious_per_class)
            )
            + "".join(
                "s{}_loss_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_loss_per_class)
            )
        )


def compute_loss(preds, labels, loss_name, args):

    scale_idxs = torch.arange(len(preds)).int()

    # Dice loss across classes at multiple scales
    ms_loss = torch.stack(
        [
            src.model.loss.__dict__[loss_name](pred, label, idx_scale, args)
            for pred, label, idx_scale in zip(preds, labels, scale_idxs)
        ]
    )

    if "90" not in args.model_name:
        total_loss = torch.sum(ms_loss[3:]) + torch.mean(ms_loss[:3])
    else:
        total_loss = torch.sum(ms_loss)

    # Store losses in dict
    total_loss_dict = {
        "loss": float(total_loss),
    }

    return total_loss, total_loss_dict


def visualize_score(scores, heatmaps, grid, image, iou, num_classes):
    # Condese scores and ground truths to single map
    class_idx = torch.arange(len(scores)) + 1
    logits = scores.clone().cpu() * class_idx.view(-1, 1, 1)
    logits, _ = logits.max(dim=0)
    scores = (scores.detach().clone().cpu() > 0.5).float() * class_idx.view(-1, 1, 1)
    scores, _ = scores.max(dim=0)
    heatmaps = (heatmaps.detach().clone().cpu() > 0.5).float() * class_idx.view(
        -1, 1, 1
    )
    heatmaps, _ = heatmaps.max(dim=0)

    # Visualize score
    fig = plt.figure(num="score", figsize=(8, 6))
    fig.clear()

    gs = mpl.gridspec.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1:, 1])
    ax4 = fig.add_subplot(gs[1:, 2])

    image = ax1.imshow(image)
    ax1.grid(which="both")
    src.visualization.encoded.vis_score_raw(logits, grid, cmap="magma", ax=ax2)
    src.vis_score(scores, grid, cmap="magma", ax=ax3, num_classes=num_classes)
    src.vis_score(heatmaps, grid, cmap="magma", ax=ax4, num_classes=num_classes)

    grid = grid.cpu().detach().numpy()
    yrange = np.arange(grid[:, 0].max(), step=5)
    xrange = np.arange(start=grid[0, :].min(), stop=grid[0, :].max(), step=5)
    ymin, ymax = 0, grid[:, 0].max()
    xmin, xmax = grid[0, :].min(), grid[0, :].max()

    ax2.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    ax2.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    ax3.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    ax3.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    ax4.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    ax4.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)

    ax1.set_title("Input image", size=11)
    ax2.set_title("Model output logits", size=11)
    ax3.set_title("Model prediction = logits" + r"$ > 0.5$", size=11)
    ax4.set_title("Ground truth", size=11)

    # plt.suptitle(
    #     "IoU : {:.2f}".format(iou), size=14,
    # )

    gs.tight_layout(fig)
    gs.update(top=0.9)

    return fig


def parse_args():
    import time
    parser = ArgumentParser()

    # ----------------------------- Data options ---------------------------- #
    parser.add_argument(
        "--root",
        type=str,
        default="nuscenes_data",
        help="root directory of the dataset",
    )
    parser.add_argument(
        "--nusc-version", type=str, default="v1.0-trainval", help="nuscenes version",
    )
    parser.add_argument(
        "--occ-gt",
        type=str,
        default="200down100up",
        help="occluded (occ) or unoccluded(unocc) ground truth maps",
    )
    parser.add_argument(
        "--gt-version",
        type=str,
        default="semantic_maps_new_200x200",
        help="ground truth name",
    )
    parser.add_argument(
        "--train-split", type=str, default="train_mini", help="ground truth name",
    )
    parser.add_argument(
        "--val-split", type=str, default="val_mini", help="ground truth name",
    )
    parser.add_argument(
        "--data-size",
        type=float,
        default=0.2,
        help="percentage of dataset to train on",
    )
    parser.add_argument(
        "--load-classes-nusc",
        type=str,
        nargs=14,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "road_segment",
            "lane",
            "bus",
            "bicycle",
            "car",
            "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
        ],
        help="Classes to load for NuScenes",
    )
    parser.add_argument(
        "--pred-classes-nusc",
        type=str,
        nargs=12,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "bus",
            "bicycle",
            "car",
            "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
        ],
        help="Classes to predict for NuScenes",
    )
    parser.add_argument(
        "--lidar-ray-mask",
        type=str,
        default="dense",
        help="sparse or dense lidar ray visibility mask",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        nargs=2,
        default=(50.0, 50.0),
        help="width and depth of validation grid, in meters",
    )
    parser.add_argument(
        "--z-intervals",
        type=float,
        nargs="+",
        default=[1.0, 9.0, 21.0, 39.0, 51.0],
        help="depths at which to predict BEV maps",
    )
    parser.add_argument(
        "--grid-jitter",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="magn. of random noise applied to grid coords",
    )
    parser.add_argument(
        "--aug-image-size",
        type=int,
        nargs="+",
        default=[1280, 720],
        help="size of random image crops during training",
    )
    parser.add_argument(
        "--desired-image-size",
        type=int,
        nargs="+",
        default=[1600, 900],
        help="size images are padded to before passing to network",
    )
    parser.add_argument(
        "--yoffset",
        type=float,
        default=1.74,
        help="vertical offset of the grid from the camera axis",
    )

    # -------------------------- Model options -------------------------- #
    parser.add_argument(
        "--model-name",
        type=str,
        default="PyrOccTranDetr_S_0904_old_rep100x100_out100x100",
        help="Model to train",
    )
    parser.add_argument(
        "-r",
        "--grid-res",
        type=float,
        default=0.5,
        help="size of grid cells, in meters",
    )
    parser.add_argument(
        "--frontend",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="name of frontend ResNet architecture",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="choose pretrained frontend ResNet",
    )
    parser.add_argument(
        "--pretrained-bem",
        type=bool,
        default=False,
        help="choose pretrained BEV estimation model",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="iccv_segdet_pyrocctrandetr_s_0904_100x100_200down100up_dice_adam_lr5e5_di3_1600x900",
        help="name of pretrained model to load",
    )
    parser.add_argument(
        "--load-ckpt",
        type=str,
        default="checkpoint-0020.pth.gz",
        help="name of checkpoint to load",
    )
    parser.add_argument(
        "--ignore", type=str, default=["nothing"], help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--ignore-reload",
        type=str,
        default=["nothing"],
        help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--focal-length", type=float, default=1266.417, help="focal length",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs=4,
        default=[8.0, 16.0, 32.0, 64.0],
        help="resnet frontend scale factor",
    )
    parser.add_argument(
        "--cropped-height",
        type=float,
        nargs=4,
        default=[20.0, 20.0, 20.0, 20.0],
        help="resnet feature maps cropped height",
    )
    parser.add_argument(
        "--y-crop",
        type=float,
        nargs=4,
        default=[15, 15.0, 15.0, 15.0],
        help="Max y-dimension in world space for all depth intervals",
    )
    parser.add_argument(
        "--dla-norm",
        type=str,
        default="GroupNorm",
        help="Normalisation for inputs to topdown network",
    )
    parser.add_argument(
        "--bevt-linear-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to linear layer in BEVT",
    )
    parser.add_argument(
        "--bevt-conv-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to conv layer in BEVT",
    )
    parser.add_argument(
        "--dla-l1-nchannels",
        type=int,
        default=64,
        help="vertical offset of the grid from the camera axis",
    )
    parser.add_argument(
        "--n-enc-layers",
        type=int,
        default=2,
        help="number of transfomer encoder layers",
    )
    parser.add_argument(
        "--n-dec-layers",
        type=int,
        default=2,
        help="number of transformer decoder layers",
    )

    # ---------------------------- Loss options ---------------------------- #
    parser.add_argument(
        "--loss", type=str, default="dice_loss_mean", help="Loss function",
    )
    parser.add_argument(
        "--exp-cf",
        type=float,
        default=0.0,
        help="Exponential for class frequency in weighted dice loss",
    )
    parser.add_argument(
        "--exp-os",
        type=float,
        default=0.2,
        help="Exponential for object size in weighted dice loss",
    )

    # ------------------------ Optimization options ----------------------- #
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("-l", "--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.99,
        help="factor to decay learning rate by every epoch",
    )

    # ------------------------- Training options ------------------------- #
    parser.add_argument(
        "-e", "--epochs", type=int, default=600, help="number of epochs to train for"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1, help="mini-batch size for training"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=5,
        help="Gradient accumulation over number of batches",
    )

    # ------------------------ Experiment options ----------------------- #
    parser.add_argument(
        "--name", type=str,
        default=str(time.time()),
        help="name of experiment",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="experiments",
        help="directory to save experiments to",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        nargs="*",
        default=[0],
        help="ids of gpus to train on. Leave empty to use cpu",
    )
    parser.add_argument(
        "--num-gpu", type=int, default=1, help="number of gpus",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="number of worker threads to use for data loading",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="number of epochs between validation runs",
    )
    parser.add_argument(
        "--print-iter",
        type=int,
        default=5,
        help="print loss summary every N iterations",
    )
    parser.add_argument(
        "--vis-iter",
        type=int,
        default=20,
        help="display visualizations every N iterations",
    )
    args = parser.parse_args()

    # This is a patch to use "mono-semantic-maps" dataloader instead of lmdb's.
    # The directories content follows "mono-semantic-maps"'s requirements
    args.dataloader_config = {
        'label_root': "/home/shai/DataSets/nuscenes/map-labels-v1.0",
        'dataroot': "/home/shai/DataSets/nuscenes",
        'nuscenes_version': args.nusc_version,
        'img_size':  args.desired_image_size,
        'hold_out_calibration': False,
        'hflip': True,
        'epoch_size': None,
        'batch_size': args.batch_size,
        'num_workers': args.workers,
    }

    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _make_experiment(args):
    print("\n" + "#" * 80)
    print(datetime.now().strftime("%A %-d %B %Y %H:%M"))
    print(
        "Creating experiment '{}' in directory:\n  {}".format(args.name, args.savedir)
    )
    print("#" * 80)
    print("\nConfig:")
    for key in sorted(args.__dict__):
        print("  {:12s} {}".format(key + ":", args.__dict__[key]))
    print("#" * 80)

    # Create a new directory for the experiment
    savedir = os.path.join(args.savedir, args.name)
    os.makedirs(savedir, exist_ok=True)

    # # Create tensorboard summary writer
    summary = SummaryWriter(savedir)

    # # Save configuration to file
    with open(os.path.join(savedir, "config.txt"), "w") as fp:
        json.dump(args.__dict__, fp)

    # # Write config as a text summary
    # summary.add_text(
    #     "config",
    #     "\n".join("{:12s} {}".format(k, v) for k, v in sorted(args.__dict__.items())),
    # )
    # summary.file_writer.flush()

    return None


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    ckpt_file = os.path.join(
        args.savedir, args.name, "checkpoint-{:04d}.pth.gz".format(epoch)
    )
    print("==> Saving checkpoint '{}'".format(ckpt_file))
    torch.save(ckpt, ckpt_file)


def main():

    # Parse command line arguments
    args = parse_args()
    args.root = os.path.join(os.getcwd(), args.root)
    print(args.root)
    args.savedir = os.path.join(os.getcwd(), args.savedir)
    print(args.savedir)

    # Build depth intervals along Z axis and reverse
    z_range = args.z_intervals
    args.grid_size = (z_range[-1] - z_range[0], z_range[-1] - z_range[0])

    # Calculate cropped heights of feature maps
    h_cropped = src.utils.calc_cropped_heights(
        args.focal_length, np.array(args.y_crop), z_range, args.scales
    )
    args.cropped_height = [h for h in h_cropped]

    num_gpus = torch.cuda.device_count()
    args.num_gpu = num_gpus

    ### Create experiment ###
    summary = _make_experiment(args)

    print("loading train data - This paer replaces the original dataloader (lmdb)")
    train_loader, val_loader = build_dataloaders('nuscenes', CfgNode(args.dataloader_config))

    # Build model
    model = networks.__dict__[args.model_name](
        num_classes=len(args.pred_classes_nusc),
        frontend=args.frontend,
        grid_res=args.grid_res,
        pretrained=args.pretrained,
        img_dims=args.desired_image_size,
        z_range=z_range,
        h_cropped=args.cropped_height,
        dla_norm=args.dla_norm,
        additions_BEVT_linear=args.bevt_linear_additions,
        additions_BEVT_conv=args.bevt_conv_additions,
        dla_l1_n_channels=args.dla_l1_nchannels,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
    )

    if args.pretrained_bem:
        pretrained_model_dir = os.path.join(args.savedir, args.pretrained_model)
        # pretrained_ckpt_fn = sorted(
        #     [
        #         f
        #         for f in os.listdir(pretrained_model_dir)
        #         if os.path.isfile(os.path.join(pretrained_model_dir, f))
        #         and ".pth.gz" in f
        #     ]
        # )
        pretrained_pth = os.path.join(pretrained_model_dir, args.load_ckpt)
        pretrained_dict = torch.load(pretrained_pth)["model"]
        mod_dict = OrderedDict()

        # # Remove "module" from name
        for k, v in pretrained_dict.items():
            if any(module in k for module in args.ignore):
                continue
            else:
                name = k[7:]
                mod_dict[name] = v

        model.load_state_dict(mod_dict, strict=False)
        print("loaded pretrained model")

    device = torch.device("cuda")
    model = nn.DataParallel(model)
    model.to(device)

    # Setup optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), args.lr,)
    else:
        optimizer = optim.__dict__[args.optimizer](
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    # Check if saved model checkpoint exists
    model_dir = os.path.join(args.savedir, args.name)
    checkpt_fn = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f)) and ".pth.gz" in f
        ]
    )
    if len(checkpt_fn) != 0:
        model_pth = os.path.join(model_dir, checkpt_fn[-1])
        ckpt = torch.load(model_pth)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epoch_ckpt = ckpt["epoch"]
        print("starting training from {}".format(checkpt_fn[-1]))
    else:
        epoch_ckpt = 1
        pass

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    for epoch in range(epoch_ckpt, args.epochs + 1):

        print("\n=== Beginning epoch {} of {} ===".format(epoch, args.epochs))

        # # Train model
        train(args, train_loader, model, optimizer, epoch)

        # Run validation every N epochs
        if epoch % args.val_interval == 0:
            # Save model checkpoint
            save_checkpoint(args, epoch, model, optimizer, scheduler)
            validate(args, val_loader, model, epoch)

        # Update and log learning rate
        scheduler.step()


if __name__ == "__main__":
    main()
