import os
import csv
import cv2
import glob
import torch
import descartes
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from typing import Dict, List, Tuple, Optional, Union
from collections import namedtuple
from pyquaternion import Quaternion
from argparse import ArgumentParser
import PIL as pil
from PIL import Image
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely import affinity
from descartes import PolygonPatch

from torchvision.transforms.functional import to_tensor

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import (
    view_points,
    box_in_image,
    BoxVisibility,
    transform_matrix,
)
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.map_expansion import map_api

from src import utils


def read_split(filename):
    """
    Read a list of NuScenes sample tokens
    """
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        return [val for val in lines if val != ""]


def get_token():
    data_root = "/vol/vssp/datasets/mixedmode/nuscenes/nuscenes"
    root = "/vol/research/sceneEvolution/data/nuscenes/"
    split = "paper_vis"
    split_file = os.path.join(root, "splits", (split + ".txt"))
    data = NuScenes(version="v1.0-trainval", dataroot=data_root, verbose=False)

    fn2compare = [
        "n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535656817762404",
        "n008-2018-08-30-15-52-26-0400__CAM_FRONT__1535658978412404",
        "n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731145162404",
        "n015-2018-07-24-11-13-19+0800__CAM_FRONT__1532402419512460",
    ]

    split_tokens = read_split(split_file)
    image_tokens = []
    print(split_file)
    for fn in fn2compare:
        for token in split_tokens:
            sample_record = data.get("sample", token)

            # Load image
            cam_token = sample_record["data"]["CAM_FRONT"]
            cam_record = data.get("sample_data", cam_token)
            cam_path = data.get_sample_data_path(cam_token)
            image_fn = cam_path
            if fn in image_fn:
                image_tokens.append(token)

    print(image_tokens)


def int_coords(x):
    # function to round and convert to int
    return np.array(x).round().astype(np.int32)


def mask_for_polygons(polygons: MultiPolygon, mask: np.ndarray):
    """
    Convert a polygon or multipolygon list to an image mask ndarray.
    :param polygons: List of Shapely polygons to be converted to numpy array.
    :param mask: Canvas where mask will be generated.
    :return: Numpy ndarray polygon mask.
    """
    if not polygons:
        return mask

    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
    cv2.fillPoly(mask, exteriors, 1)
    cv2.fillPoly(mask, interiors, 0)
    return mask


def get_patch_coord(
    patch_box: Tuple[float, float, float, float], patch_angle: float = 0.0
) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(
        patch, patch_angle, origin=(patch_x, patch_y), use_radians=False
    )

    return patch


def _get_layer_polygon(self, patch_box, patch_angle, layer_name, pose, pose_rot):
    """
     Retrieve the polygons of a particular layer within the specified patch.
     :param patch_box: Patch box defined as [x_center, y_center, height, width].
     :param patch_angle: Patch orientation in degrees.
     :param layer_name: name of map layer to be extracted.
     :return: List of Polygon in a patch box.
     """
    if layer_name not in self.non_geometric_polygon_layers:
        raise ValueError("{} is not a polygonal layer".format(layer_name))

    ego_pose_x = patch_box[0]
    ego_pose_y = patch_box[1]

    ego_pose_rot_rad = np.arccos(pose_rot[0][0, 0])

    # cam_pose = pose[1]
    # cam_pose_rot = pose_rot[1]
    cam_pose_rot_rad = np.arccos(pose_rot[1][0, 0])

    x_min = ego_pose_x - 50
    x_max = ego_pose_x + 50
    y_min = ego_pose_y - 50
    y_max = ego_pose_y + 50

    # Transform patch into ego frame
    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(
        patch, ego_pose_rot_rad, origin=(ego_pose_x, ego_pose_y), use_radians=True
    )
    patch = affinity.affine_transform(
        patch, [1.0, 0.0, 0.0, 1.0, -ego_pose_x, -ego_pose_y]
    )
    # Transform patch into camera
    patch = affinity.rotate(patch, cam_pose_rot_rad, origin=(0, 0), use_radians=True)

    records = getattr(self, layer_name)

    polygon_list = []

    for record in records:
        polygons = [
            self.extract_polygon(polygon_token)
            for polygon_token in record["polygon_tokens"]
        ]
        for polygon in polygons:
            # Transform into camera frame
            new_polygon = affinity.affine_transform(
                polygon, [1.0, 0.0, 0.0, 1.0, -ego_pose_x, -ego_pose_y]
            )
            # Transform into camera
            new_polygon = affinity.rotate(
                new_polygon, cam_pose_rot_rad, origin=(0, 0), use_radians=True
            )

            # Intersect with patch
            new_polygon = new_polygon.intersection(patch)

            new_polygon = affinity.rotate(
                new_polygon, -cam_pose_rot_rad, origin=(0, 0), use_radians=True
            )
            new_polygon = affinity.rotate(
                new_polygon, -ego_pose_rot_rad, origin=(0, 0), use_radians=True
            )

            # new_polygon = affinity.affine_transform(
            #     new_polygon, [1.0, 0.0, 0.0, 1.0, -ego_pose_x, -ego_pose_y]
            # )

            if new_polygon.geom_type is "Polygon":
                new_polygon = MultiPolygon([new_polygon])
            polygon_list.append(new_polygon)

    return polygon_list


def _polygon_geom_to_mask(
    self,
    layer_geom: List[Polygon],
    local_box: Tuple[float, float, float, float],
    layer_name: str,
    canvas_size: Tuple[int, int],
):
    """
    Convert polygon inside patch to binary mask and return the map patch.
    :param layer_geom: list of polygons for each map layer
    :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
        x_center = y_center = 0.
    :param layer_name: name of map layer to be converted to binary map mask patch.
    :param canvas_size: Size of the output mask (h, w).
    :return: Binary map mask patch with the size canvas_size.
    """
    if layer_name not in self.non_geometric_polygon_layers:
        raise ValueError("{} is not a polygonal layer".format(layer_name))

    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]

    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for polygon in layer_geom:
        new_polygon = polygon.intersection(patch)
        if not new_polygon.is_empty:
            new_polygon = affinity.affine_transform(
                new_polygon, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y]
            )
            new_polygon = affinity.scale(
                new_polygon, xfact=scale_width, yfact=scale_height, origin=(0, 0)
            )

            if new_polygon.geom_type is "Polygon":
                new_polygon = MultiPolygon([new_polygon])
            map_mask = mask_for_polygons(new_polygon, map_mask)

    return map_mask


def _layer_geom_to_mask(
    self,
    layer_name,
    layer_geom,
    local_box: Tuple[float, float, float, float],
    canvas_size: Tuple[int, int],
):
    """
    Wrapper method that gets the mask for each layer's geometries.
    :param layer_name: The name of the layer for which we get the masks.
    :param layer_geom: List of the geometries of the layer specified in layer_name.
    :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
        x_center = y_center = 0.
    :param canvas_size: Size of the output mask (h, w).
    """
    if layer_name in self.non_geometric_polygon_layers:
        return _polygon_geom_to_mask(
            self, layer_geom, local_box, layer_name, canvas_size
        )
    elif layer_name in self.non_geometric_line_layers:
        return self._line_geom_to_mask(
            self, layer_geom, local_box, layer_name, canvas_size
        )
    else:
        raise ValueError("{} is not a valid layer".format(layer_name))


def _get_layer_geom(self, patch_box, patch_angle, layer_name, ego_pose, ego_pose_rot):
    """
    Wrapper method that gets the geometries for each layer.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :param layer_name: Name of map layer to be converted to binary map mask patch.
    :return: List of geometries for the given layer.
    """
    if layer_name in self.non_geometric_polygon_layers:
        return _get_layer_polygon(
            self, patch_box, patch_angle, layer_name, ego_pose, ego_pose_rot
        )
    else:
        raise ValueError("{} is not a valid layer".format(layer_name))


def get_map_geom(self, patch_box, patch_angle, layer_names, ego_pose, ego_pose_rot):
    """
    Returns a list of geometries in the specified patch_box.
    These are unscaled, but aligned with the patch angle.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
                        North-facing corresponds to 0.
    :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
    :return: List of layer names and their corresponding geometries.
    """
    # If None, return all geometric layers.
    if layer_names is None:
        layer_names = self.non_geometric_layers

    # Get each layer name and geometry and store them in a list.
    map_geom = []
    for layer_name in layer_names:
        layer_geom = _get_layer_geom(
            self, patch_box, patch_angle, layer_name, ego_pose, ego_pose_rot
        )
        if layer_geom is None:
            continue
        map_geom.append((layer_name, layer_geom))

    return map_geom


def map_geom_to_mask(
    self,
    map_geom,
    local_box: Tuple[float, float, float, float],
    canvas_size: Tuple[int, int],
):
    """
    Return list of map mask layers of the specified patch.
    :param map_geom: List of layer names and their corresponding geometries.
    :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
        x_center = y_center = 0.
    :param canvas_size: Size of the output mask (h, w).
    :return: Stacked numpy array of size [c x h x w] with c channels and the same height/width as the canvas.
    """
    # Get each layer mask and stack them into a numpy tensor.
    map_mask = []
    for layer_name, layer_geom in map_geom:
        layer_mask = _layer_geom_to_mask(
            self, layer_name, layer_geom, local_box, canvas_size
        )
        if layer_mask is not None:
            map_mask.append(layer_mask)

    return np.array(map_mask)


def get_map_mask(
    self, patch_box, patch_angle, layer_names, canvas_size, ego_pose, ego_pose_rot
):
    """
    Return list of map mask layers of the specified patch.
    :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, this plots the entire map.
    :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
    :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
    :param canvas_size: Size of the output mask (h, w). If None, we use the default resolution of 10px/m.
    :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
    """

    # If None, return all geometric layers.
    if layer_names is None:
        layer_names = self.non_geometric_layers

    # If None, return the specified patch in the original scale of 10px/m.
    if canvas_size is None:
        map_scale = 4
        canvas_size = np.array((patch_box[2], patch_box[3])) * map_scale
        canvas_size = tuple(np.round(canvas_size).astype(np.int32))

    # Get geometry of each layer.
    map_geom = get_map_geom(
        self, patch_box, patch_angle, layer_names, ego_pose, ego_pose_rot
    )

    # Convert geometry of each layer into mask and stack them into a numpy tensor.
    # Convert the patch box from global coordinates to local coordinates by setting the center to (0, 0).
    local_box = (0.0, 0.0, 100, 100)
    map_mask = map_geom_to_mask(self, map_geom, local_box, canvas_size)
    assert np.all(map_mask.shape[1:] == canvas_size)

    return map_mask


def render_map_mask(
    self,
    patch_box,
    patch_angle,
    layer_names,
    canvas_size,
    figsize,
    n_row,
    ego_pose,
    ego_pose_rot,
):
    """
    Render map mask of the patch specified by patch_box and patch_angle.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :param layer_names: A list of layer names to be extracted.
    :param canvas_size: Size of the output mask (h, w).
    :param figsize: Size of the figure.
    :param n_row: Number of rows with plots.
    :return: The matplotlib figure and a list of axes of the rendered layers.
    """
    if layer_names is None:
        layer_names = self.non_geometric_layers

    map_mask = get_map_mask(
        self, patch_box, patch_angle, layer_names, canvas_size, ego_pose, ego_pose_rot
    )

    # If no canvas_size is specified, retrieve the default from the output of get_map_mask.
    if canvas_size is None:
        canvas_size = map_mask.shape[1:]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, canvas_size[1])
    ax.set_ylim(0, canvas_size[0])

    n_col = len(map_mask) // n_row
    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.025, hspace=0.05)
    for i in range(len(map_mask)):
        r = i // n_col
        c = i - r * n_col
        subax = plt.subplot(gs[r, c])
        subax.imshow(map_mask[i], origin="lower")
        subax.text(canvas_size[0] * 0.5, canvas_size[1] * 1.1, layer_names[i])
        subax.grid(False)

    return fig, fig.axes, map_mask


def _render_layer(self, ax, layer_name, alpha: float, tokens: List[str] = None):
    """
    Wrapper method that renders individual layers on an axis.
    :param ax: The matplotlib axes where the layer will get rendered.
    :param layer_name: Name of the layer that we are interested in.
    :param alpha: The opacity of the layer to be rendered.
    :param tokens: Optional list of tokens to render. None means all tokens are rendered.
    """
    if layer_name in self.non_geometric_polygon_layers:
        polygons = _render_polygon_layer(self, ax, layer_name, alpha, tokens)
        return polygons
    # elif layer_name in self.non_geometric_line_layers:
    #     self._render_line_layer(self, ax, layer_name, alpha, tokens)


def _render_polygon_layer(
    self, ax, layer_name: str, alpha: float, tokens: List[str] = None
):
    """
    Renders an individual non-geometric polygon layer on an axis.
    :param ax: The matplotlib axes where the layer will get rendered.
    :param layer_name: Name of the layer that we are interested in.
    :param alpha: The opacity of the layer to be rendered.
    :param tokens: Optional list of tokens to render. None means all tokens are rendered.
    """
    if layer_name not in self.non_geometric_polygon_layers:
        raise ValueError("{} is not a polygonal layer".format(layer_name))

    first_time = True
    records = getattr(self, layer_name)
    polygons_drivable = []

    if tokens is not None:
        records = [r for r in records if r["token"] in tokens]
    if layer_name == "drivable_area":
        for record in records:
            polygons = [
                self.extract_polygon(polygon_token)
                for polygon_token in record["polygon_tokens"]
            ]

            for polygon in polygons:
                if first_time:
                    label = layer_name
                    first_time = False
                else:
                    label = None
                polygons_drivable.append(polygon)

                ax.add_patch(
                    descartes.PolygonPatch(polygon, fc="blue", alpha=alpha, label=label)
                )
        return polygons_drivable


def render_map_patch(
    self,
    box_coords: Tuple[float, float, float, float],
    layer_names: List[str] = None,
    alpha: float = 0.5,
    figsize: Tuple[float, float] = (15, 15),
    render_egoposes_range: bool = True,
    render_legend: bool = True,
):
    """
    Renders a rectangular patch specified by `box_coords`. By default renders all layers.
    :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
    :param layer_names: All the non geometric layers that we want to render.
    :param alpha: The opacity of each layer.
    :param figsize: Size of the whole figure.
    :param render_egoposes_range: Whether to render a rectangle around all ego poses.
    :param render_legend: Whether to render the legend of map layers.
    :return: The matplotlib figure and axes of the rendered layers.
    """
    x_min, y_min, x_max, y_max = box_coords

    if layer_names is None:
        layer_names = self.non_geometric_layers

    fig = plt.figure(figsize=figsize)

    local_width = x_max - x_min
    local_height = y_max - y_min
    assert local_height > 0, "Error: Map patch has 0 height!"
    local_aspect_ratio = local_width / local_height

    ax = fig.add_axes([0, 0, 1, 1 / local_aspect_ratio])

    polygons = []
    for layer_name in layer_names:
        polygon = _render_layer(self, ax, layer_name, alpha)
        polygons.append(polygon)

    x_margin = np.minimum(local_width / 4, 0)
    y_margin = np.minimum(local_height / 4, 0)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # if render_egoposes_range:
    #     ax.add_patch(Rectangle((x_min, y_min), local_width, local_height, fill=False, linestyle='-.', color='red',
    #                            lw=2))
    #     ax.text(x_min + local_width / 100, y_min + local_height / 2, "%g m" % local_height,
    #             fontsize=14, weight='bold')
    #     ax.text(x_min + local_width / 2, y_min + local_height / 100, "%g m" % local_width,
    #             fontsize=14, weight='bold')

    if render_legend:
        ax.legend(frameon=True, loc="upper right")

    return fig, ax, polygons


def render_map_in_image(
    rootdir=None,
    nusc: NuScenes = None,
    sample_token: str = None,
    sample_data_token: str = None,
    camera_channel: str = "CAM_FRONT",
    alpha: float = 0.3,
    patch_radius: float = 1000,
    min_polygon_area: float = 1000,
    render_behind_cam: bool = True,
    render_outside_im: bool = True,
    map_layer_names: List[str] = None,
    vehicle_layer_names: List[str] = None,
    grid_res=0.25,
):
    """
    Render a nuScenes camera image and overlay the polygons for the
    specified map layers.
    Note that the projections are not always accurate as the localization is in 2d.
    :param nusc: The NuScenes instance to load the image from.
    :param sample_token: The image's corresponding sample_token.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param alpha: The transparency value of the layers to render in [0, 1].
    :param patch_radius: The radius in meters around the ego car in which to select map records.
    :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
    :param render_behind_cam: Whether to render polygons where any point is behind the camera.
    :param render_outside_im: Whether to render polygons where any point is outside the image.
    :param layer_names: The names of the layers to render, e.g. ['lane'].
        If set to None, the recommended setting will be used.
    :param verbose: Whether to print to stdout.
    :param out_path: Optional path to save the rendered figure to disk.
    """
    near_plane = 1e-8

    # Check that NuScenesMap was loaded for the correct location.
    sample_record = nusc.get("sample", sample_token)
    scene_record = nusc.get("scene", sample_record["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"])
    log_location = log_record["location"]
    self = NuScenesMap(dataroot=rootdir, map_name=log_location)
    assert self.map_name == log_location, (
        "Error: NuScenesMap loaded for location %s, should be %s!"
        % (self.map_name, log_location)
    )

    # Check layers whether we can render them.
    for layer_name in map_layer_names:
        assert layer_name in self.non_geometric_polygon_layers, (
            "Error: Can only render non-geometry polygons: %s" % map_layer_names
        )

    # Grab the front camera image and intrinsics.
    cam_token = sample_record["data"][camera_channel]
    cam_record = nusc.get("sample_data", cam_token)
    cam_path = nusc.get_sample_data_path(cam_token)
    im = Image.open(cam_path)
    im_size = im.size
    cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
    cam_intrinsic = np.array(cs_record["camera_intrinsic"])

    # Retrieve the current map.
    poserecord = nusc.get("ego_pose", cam_record["ego_pose_token"])
    ego_pose = poserecord["translation"]
    ego_pose_rot = Quaternion(poserecord["rotation"]).rotation_matrix.T
    cam_pose_rot = Quaternion(cs_record["rotation"]).rotation_matrix.T
    box_coords = (
        ego_pose[0] - patch_radius,
        ego_pose[1] - patch_radius,
        ego_pose[0] + patch_radius,
        ego_pose[1] + patch_radius,
    )
    records_in_patch = self.get_records_in_patch(
        box_coords, map_layer_names, "intersect"
    )

    # Set dimensions of BEV map
    bev_dimensions = np.array([50, 50])
    bev_canvas_size = (bev_dimensions / grid_res).astype(np.int32)

    # Retrieve and render each record.
    bev_masks = []
    bev_id_masks = []
    final_layers = []

    # drivable_map = create_drivable_map(
    #     bev_canvas_size, ego_pose, [ego_pose_rot, cam_pose_rot], self
    # )
    # bev_masks.append(drivable_map)

    # for layer_name in map_layer_names:
    #     final_layers.append(layer_name)
    #     layer_mask = np.zeros(bev_canvas_size)
    #     for token in records_in_patch[layer_name]:
    #         record = self.get(layer_name, token)
    #         if layer_name == "drivable_area":
    #             polygon_tokens = record["polygon_tokens"]
    #         else:
    #             polygon_tokens = [record["polygon_token"]]
    #         for polygon_token in polygon_tokens:
    #             polygon = self.extract_polygon(polygon_token)
    #
    #             # Convert polygon nodes to pointcloud with 0 height.
    #             points = np.array(polygon.exterior.xy)
    #             points = np.vstack((points, np.zeros((1, points.shape[1]))))
    #
    #             # Transform into the ego vehicle frame for the timestamp of the image.
    #             points = points - np.array(poserecord["translation"]).reshape((-1, 1))
    #             points = np.dot(
    #                 Quaternion(poserecord["rotation"]).rotation_matrix.T, points
    #             )
    #
    #             # Transform into the camera.
    #             points = points - np.array(cs_record["translation"]).reshape((-1, 1))
    #             points = np.dot(
    #                 Quaternion(cs_record["rotation"]).rotation_matrix.T, points
    #             )
    #
    #             # plt.scatter(points[0], points[2])
    #             # plt.show()
    #
    #             # Remove points that are partially behind the camera.
    #             depths = points[2, :]
    #             behind = depths < near_plane
    #             if np.all(behind):
    #                 continue
    #
    #             if render_behind_cam:
    #                 # Perform clipping on polygons that are partially behind the camera.
    #                 points = NuScenesMapExplorer._clip_points_behind_camera(
    #                     points, near_plane
    #                 )
    #             elif np.any(behind):
    #                 # Otherwise ignore any polygon that is partially behind the camera.
    #                 continue
    #
    #             # Ignore polygons with less than 3 points after clipping.
    #             if len(points) == 0 or points.shape[1] < 3:
    #                 continue
    #
    #             pre_points = np.array(points)
    #
    #             # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    #             pts_unorm = view_points(pre_points, cam_intrinsic, normalize=False)
    #             points = view_points(points, cam_intrinsic, normalize=True)
    #
    #             # Skip polygons where all points are outside the image.
    #             # Leave a margin of 1 pixel for aesthetic reasons.
    #             inside = np.ones(points.shape[1], dtype=bool)
    #             inside = np.logical_and(inside, points[0, :] < im.size[0])
    #             inside = np.logical_and(inside, points[1, :] < im.size[1])
    #
    #             if render_outside_im:
    #                 if np.all(np.logical_not(inside)):
    #                     continue
    #             else:
    #                 if np.any(np.logical_not(inside)):
    #                     continue
    #
    #             points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
    #             polygon_proj = Polygon(points)
    #             # Filter small polygons
    #             if polygon_proj.area < min_polygon_area:
    #                 continue
    #
    #             # Project points back out
    #             pts_cc = np.dot(np.linalg.inv(cam_intrinsic), pts_unorm)
    #
    #             # Convert cam Coordinates(m) to mask(m)
    #             pts_cc_xmax, pts_cc_xmin = pts_cc[0].max(), pts_cc[0].min()
    #             pts_cc_zmax, pts_cc_zmin = pts_cc[2].max(), pts_cc[2].min()
    #             poly_canvas_size = [
    #                 int(np.round(pts_cc_zmax - pts_cc_zmin)),
    #                 int(np.round(pts_cc_xmax - pts_cc_xmin)),
    #             ]
    #             poly_mask = np.zeros(bev_canvas_size, np.uint8)
    #
    #             pts_cc_list = [
    #                 (p0 / grid_res, p1 / grid_res)
    #                 for (p0, p1) in zip(pts_cc[0], pts_cc[2])
    #             ]
    #
    #             poly_cc = Polygon(pts_cc_list)
    #             exteriors = [int_coords(poly_cc.exterior.coords)]
    #             interiors = [int_coords(pi.coords) for pi in poly_cc.interiors]
    #             offset = (int(bev_canvas_size[0] / 2), int(0))
    #
    #             cv2.fillPoly(poly_mask, exteriors, 1, offset=offset)
    #             cv2.fillPoly(poly_mask, interiors, 0, offset=offset)
    #
    #             # Add polygon mask to layer mask
    #             layer_mask += poly_mask
    #
    #             # ax.add_patch(descartes.PolygonPatch(polygon_proj, fc=self.color_map[layer_name], alpha=alpha,
    #             #                                     label=label))
    #             # ax.add_patch(descartes.PolygonPatch(polygon_proj, alpha=alpha,
    #             #                                     label=label))
    #             # plt.show()
    #     bev_masks.append(layer_mask)

    # plt.imshow(bev_masks[0] + bev_masks[1])
    # plt.show()

    # Load boxes and image.
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(
        sample_data_token, box_vis_level=BoxVisibility.ANY
    )

    vehicles = vehicle_layer_names
    # Get bounding boxes
    bev_bbox = []
    for vehicle in vehicles:
        layer_mask = np.zeros(bev_canvas_size)
        layer_instance_id_mask = np.zeros(bev_canvas_size)

        final_layers.append(vehicle)
        vehicle_check = vehicle

        box_counta = 0
        for box in boxes:
            if vehicle_check in box.name:
                # Increment box id by one (like the instance id effectively)
                box_counta += 1

                # bottom points of object box
                pts = box.bottom_corners()

                # Create mask for object
                poly_mask = np.zeros(bev_canvas_size, np.uint8)
                pts_list = [
                    (p0 / grid_res, p1 / grid_res) for (p0, p1) in zip(pts[0], pts[2])
                ]
                poly = Polygon(pts_list)
                exteriors = [int_coords(poly.exterior.coords)]
                interiors = [int_coords(pi.coords) for pi in poly.interiors]
                offset = (int(bev_canvas_size[0] / 2), int(0))
                cv2.fillPoly(poly_mask, exteriors, 1, offset=offset)
                cv2.fillPoly(poly_mask, interiors, 0, offset=offset)

                # Check if in 50x50 BEV grid
                not_in_grid = (poly_mask.sum() == 0).astype(float)

                # Add polygon mask to layer mask
                layer_mask += poly_mask

                # Create instance id for polygon mask
                poly_id_mask = poly_mask * box_counta

                # Add polygon id mask to mask for class/layer
                # To prevent overlapping additions zero out pixels for current instance
                layer_instance_id_mask *= (~poly_id_mask.astype(bool)).astype(float)
                layer_instance_id_mask += poly_id_mask

                # Create bounding box data for object
                xmax, xmin = pts[0].max(), pts[0].min()
                zmax, zmin = pts[2].max(), pts[2].min()

                x_width = np.abs(xmax - xmin)
                z_width = np.abs(zmax - zmin)

                x, z = box.center[0], box.center[2]
                # rename vehicle.construction to construction_vehicle
                if "construction" not in vehicle:
                    classname = vehicle
                else:
                    classname = "construction_vehicle"

                # get visibility level of object
                visibility = nusc.get("sample_annotation", box.token)[
                    "visibility_token"
                ]

                bev_bbox.append(
                    utils.ObjectDataBEV(
                        classname=classname,
                        x_pos=x,
                        z_pos=z,
                        x_width=x_width,
                        z_height=z_width,
                        visibility=visibility,
                        not_in_grid=not_in_grid,
                    )
                )

        bev_masks.append(layer_mask)
        bev_id_masks.append(layer_instance_id_mask)

    for idx, m in enumerate(bev_masks):
        m[m != 0] = 1
        m[m == 0] = 0

    return bev_masks, final_layers, bev_bbox, bev_id_masks


def get_object_bbox_in_bev(
    rootdir=None,
    nusc: NuScenes = None,
    sample_token: str = None,
    sample_data_token: str = None,
    camera_channel: str = "CAM_FRONT",
    alpha: float = 0.3,
    patch_radius: float = 1000,
    min_polygon_area: float = 1000,
    render_behind_cam: bool = True,
    render_outside_im: bool = True,
    map_layer_names: List[str] = None,
    vehicle_layer_names: List[str] = None,
    grid_res=0.25,
):
    """
    Render a nuScenes camera image and overlay the polygons for the
    specified map layers.
    Note that the projections are not always accurate as the localization is in 2d.
    :param nusc: The NuScenes instance to load the image from.
    :param sample_token: The image's corresponding sample_token.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param alpha: The transparency value of the layers to render in [0, 1].
    :param patch_radius: The radius in meters around the ego car in which to select map records.
    :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
    :param render_behind_cam: Whether to render polygons where any point is behind the camera.
    :param render_outside_im: Whether to render polygons where any point is outside the image.
    :param layer_names: The names of the layers to render, e.g. ['lane'].
        If set to None, the recommended setting will be used.
    :param verbose: Whether to print to stdout.
    :param out_path: Optional path to save the rendered figure to disk.
    """
    near_plane = 1e-8

    # Check that NuScenesMap was loaded for the correct location.
    sample_record = nusc.get("sample", sample_token)
    scene_record = nusc.get("scene", sample_record["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"])
    log_location = log_record["location"]
    self = NuScenesMap(dataroot=rootdir, map_name=log_location)
    assert self.map_name == log_location, (
        "Error: NuScenesMap loaded for location %s, should be %s!"
        % (self.map_name, log_location)
    )

    # Check layers whether we can render them.
    for layer_name in map_layer_names:
        assert layer_name in self.non_geometric_polygon_layers, (
            "Error: Can only render non-geometry polygons: %s" % map_layer_names
        )

    # # Grab the front camera image and intrinsics.
    # cam_token = sample_record["data"][camera_channel]
    # cam_record = nusc.get("sample_data", cam_token)
    # cam_path = nusc.get_sample_data_path(cam_token)

    # Retrieve and render each record.
    bev_bbox = []

    # Load boxes and image.
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(
        sample_data_token, box_vis_level=BoxVisibility.ANY
    )

    vehicles = vehicle_layer_names
    # get boxes.
    for vehicle in vehicles:
        vehicle_check = vehicle
        for box in boxes:
            if vehicle_check in box.name:
                pts = box.bottom_corners()
                xmax, xmin = pts[0].max(), pts[0].min()
                zmax, zmin = pts[2].max(), pts[2].min()

                x_width = np.abs(xmax - xmin)
                z_width = np.abs(zmax - zmin)

                x, z = box.center[0], box.center[2]
                # rename vehicle.construction to construction_vehicle
                if "construction" not in vehicle:
                    classname = vehicle
                else:
                    classname = "construction_vehicle"

                # get visibility level of object
                visibility = nusc.get("sample_annotation", box.token)[
                    "visibility_token"
                ]

                bev_bbox.append(
                    utils.ObjectDataBEV(
                        classname=classname,
                        x_pos=x,
                        z_pos=z,
                        x_width=x_width,
                        z_height=z_width,
                        visibility=visibility,
                    )
                )

    return bev_bbox


def create_drivable_map(bev_canvas_size, ego_pose, ego_pose_rot, self):
    # first create drivable
    patch_box = (
        ego_pose[0],
        ego_pose[1],
        100,
        100,
    )
    figsize = (10, 10)
    _, _, drivable_map = render_map_mask(
        self,
        patch_box,
        0,
        ["drivable_area"],
        canvas_size=(bev_canvas_size * 2),
        figsize=figsize,
        n_row=1,
        ego_pose=ego_pose,
        ego_pose_rot=ego_pose_rot,
    )

    # Trim drivable
    drivable_map = drivable_map[0][
        : bev_canvas_size[0], bev_canvas_size[0] // 2 : -bev_canvas_size[0] // 2
    ]
    drivable_map = np.flip(drivable_map, axis=0)
    drivable_map = np.flip(drivable_map, axis=1)
    return drivable_map

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="number of residual blocks in topdown network",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=35000,
        help="number of residual blocks in topdown network",
    )
    return parser.parse_args()


def generate_semantic_maps():
    # Generate semantic maps with objects nearest to the camera at the top of the
    # image array (i.e. near row index 0), also generate
    # mask for occluded regions and those outside FOV

    rootdir = "<root>"
    savedir = "<save_path>"

    # Make directory for semantic maps
    semantic_maps_dir = os.path.join(savedir, "semantic_maps_200x200")
    if not os.path.exists(semantic_maps_dir):
        os.mkdir(semantic_maps_dir)

    objectdata_dir = os.path.join(savedir, "objectdata_bev")

    nusc = NuScenes(version="v1.0-trainval", dataroot=rootdir, verbose=True)

    # Parse command line arguments
    args = parse_args()

    for idx_sample in range(args.start, args.end):

        sample = nusc.sample[idx_sample]
        sample_token = sample["token"]

        # Load sample record
        sample_record = nusc.get("sample", sample_token)

        # Load camera tokens, image and records
        cam_token = sample_record["data"]["CAM_FRONT"]
        image_fn = nusc.get("sample_data", cam_token)["filename"]
        im = Image.open(os.path.join(rootdir, image_fn))
        image_fn = image_fn.split("/")[-1].split(".jpg")[0]
        cam_record = nusc.get("sample_data", cam_token)

        # Render LIDAR points in the image
        # nusc.render_pointcloud_in_image(
        #     sample_token, pointsensor_channel="LIDAR_TOP"
        # )

        # Generate BEV masks for sample
        # BEV Grid dimensions in camera coordinate frame
        bev_max_z = 50
        bev_max_x = 25
        grid_res = 0.25
        scaling_factor = 1 / grid_res

        map_layer_names = [
            "drivable_area",
            # "road_segment",
            # "lane",
            # "ped_crossing",
            # "walkway",
            # "carpark_area",
        ]
        vehice_layer_names = [
            "bus",
            "bicycle",
            "car",
            "vehicle.construction",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            # "trafficcone",
            # "barrier",
        ]
        bev_masks, layers, bev_bbox, bev_id_masks = render_map_in_image(
            rootdir=rootdir,
            nusc=nusc,
            sample_token=sample_token,
            sample_data_token=cam_token,
            camera_channel="CAM_FRONT",
            map_layer_names=map_layer_names,
            vehicle_layer_names=vehice_layer_names,
            grid_res=grid_res,
        )

        layers[layers.index("vehicle.construction")] = "construction_vehicle"

        # ## Generate LIDAR ray mask (visibility mask)
        # # Load intrinsics matrix
        cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])

        # Load LIDAR
        lidar_token = sample_record["data"]["LIDAR_TOP"]
        pointsensor = nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(rootdir, pointsensor["filename"])
        pc = LidarPointCloud.from_file(lidar_path)

        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        lidar_pc = lidar_scan.reshape((-1, 5)).T
        ring_index = lidar_pc[-1]

        # Points live in the point sensor frame.
        # So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame
        # for the timestamp of the sweep.
        cs_record = nusc.get(
            "calibrated_sensor", pointsensor["calibrated_sensor_token"]
        )
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_record["translation"]))

        # Second step: transform to the global frame.
        poserecord = nusc.get("ego_pose", pointsensor["ego_pose_token"])
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc.translate(np.array(poserecord["translation"]))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get("ego_pose", cam_record["ego_pose_token"])
        pc.translate(-np.array(poserecord["translation"]))
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        pc.translate(-np.array(cs_record["translation"]))
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)
        pts_lidar_camcoords = pc.points[:3, :]

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        # Take the actual picture and get lidar points in pixel coords
        # (matrix multiplication with camera-matrix + renormalization).
        pts_lidar_pixelcoords = view_points(
            pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True
        )

        # Remove points that are either outside or behind the camera.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 1e-8)
        mask = np.logical_and(mask, pts_lidar_pixelcoords[0, :] >= 0)
        mask = np.logical_and(mask, pts_lidar_pixelcoords[0, :] <= im.size[0])

        # Apply mask to get final lidar points which are only in the image
        pts_cc = pts_lidar_camcoords[:, mask]
        pts_pc = pts_lidar_pixelcoords[:, mask]

        # Apply mask to ring index
        ring_index_cc = ring_index[mask]

        # Sort cam coords by ring index [3, N] ---> [n_rings, 3, N / n_rings]
        # also pads with zeros in order to use arrays again (changes mean)
        sorted_pts_cc = sort_pts_by_ring_index(pts_cc, ring_index_cc)
        sorted_pts_pc = sort_pts_by_ring_index(pts_pc, ring_index_cc)

        # Interpolate between points at each ring to get denser set of points
        pts_z = sorted_pts_cc[:, 2]
        pts_x = sorted_pts_cc[:, 0]
        eval_current = np.arange(pts_z.shape[-1])
        interp_z = interp1d(eval_current, pts_z, axis=1)
        interp_x = interp1d(eval_current, pts_x, axis=1)
        eval_dense = np.linspace(0, eval_current[-1], 200)
        pts_z_dense = interp_z(eval_dense)
        pts_x_dense = interp_x(eval_dense)
        pts_cc_dense = np.zeros((pts_x_dense.shape[0], 3, pts_x_dense.shape[-1]))
        pts_cc_dense[:, 0] = pts_x_dense
        pts_cc_dense[:, 2] = pts_z_dense
        pts_cc_dense = np.moveaxis(pts_cc_dense, 0, -1).reshape(3, -1)

        for idx, pt in enumerate(zip(pts_x_dense, pts_z_dense)):
            plt.scatter(pt[0], pt[1], c=plt.cm.magma(idx/len(pts_x_dense)))
        plt.show()

        plt.scatter(pts_cc_dense[0], pts_cc_dense[2])
        plt.show()

        plt.imshow(im)
        plt.plot(np.arange(im.size[0]), np.zeros(im.size[0]) + im.size[1] * 0.37)
        for idx, pt in enumerate(sorted_pts_pc):
            plt.scatter(pt[0], pt[1], c=plt.cm.magma(idx / len(sorted_pts_pc)), s=1)
        plt.show()

        # Bin lidar points in image, any empty bins below threshold
        # y-value (v-coord) in image are treated as lidar rays which return at inf
        v_min = 0
        v_max = im.size[1]
        v_threshold_ratio = 0.51

        # Count number of rings below threshold in image
        # Use max of ring v-coord as proxy for mean as
        # array is padded with zeros so mean is meaningless
        ring_v_mean = sorted_pts_pc[:,1].max(axis=1)
        n_rings = np.sum(
            (ring_v_mean > v_min) * (ring_v_mean < v_max)
        )
        bin_res_v = 2.5
        n_bins_v = n_rings / bin_res_v
        bin_size_v = (v_max - v_min) / n_bins_v
        bin_size_u = 20
        n_bins_u = im.size[0] / bin_size_u

        H, _, _ = np.histogram2d(
            x=pts_pc[1], y=pts_pc[0],
            bins=[int(n_bins_v), int(n_bins_u)],
            range=[[int(v_min), int(v_max)], [0, im.size[0]]]
        )

        # Pick threshold under which bins can be picked (round down as
        # it is better than have a smaller threshold)
        v_bin_threshold = int(n_bins_v * v_threshold_ratio)

        # Trim histogram to v-threshold
        H = H[v_bin_threshold - 1:, :]

        # Get indices of bins with zero lidar points in them
        H_zero_v_idx, H_zero_u_idx = np.where(H == 0)
        H_zero_col_idx = np.unique(H_zero_u_idx)

        if len(H_zero_col_idx) > 0:
            # Get pixel coords of top edge of bin
            bin_u_idx_l = H_zero_col_idx * bin_size_u
            bin_u_idx_r = (H_zero_col_idx + 1) * bin_size_u
            depth = 50
            zero_bins_pc_l = np.stack(
                [bin_u_idx_l, np.ones_like(bin_u_idx_l), np.ones_like(bin_u_idx_l)]
            ) * depth
            zero_bins_pc_r = np.stack(
                [bin_u_idx_r, np.ones_like(bin_u_idx_r), np.ones_like(bin_u_idx_r)]
            ) * depth

            # Convert to cam coords
            inv_cam_instrinsic = np.linalg.inv(cam_intrinsic)
            bin_l =  inv_cam_instrinsic.dot(zero_bins_pc_l)
            bin_r =  inv_cam_instrinsic.dot(zero_bins_pc_r)

            bin_length_x = np.abs(bin_l[0,0] - bin_r[0,0])
            bin_fill_x = np.arange(bin_length_x, step=grid_res)
            pts_cc_fill_x = bin_fill_x[:, None] + bin_l[0][None, :]
            pts_cc_fill = np.stack(
                [pts_cc_fill_x.reshape(-1),
                 np.ones_like(pts_cc_fill_x.reshape(-1)),
                 np.ones_like(pts_cc_fill_x.reshape(-1)) * depth])

            pts_cc_dense = np.concatenate([pts_cc, pts_cc_fill], axis=1)

            # Create bresenham lines at smaller scale and then scale up
            b_lines, lidar_ray_mask = create_visibility_mask(
                bev_max_x, bev_max_z, pts_cc_dense, scaling_factor
            )
        else:
            # Create bresenham lines at smaller scale and then scale up
            b_lines, lidar_ray_mask = create_visibility_mask(
                bev_max_x, bev_max_z, pts_cc, scaling_factor
            )
        b_lines, lidar_ray_mask = create_visibility_mask(
            bev_masks, bev_max_x, bev_max_z, pts_cc_dense, scaling_factor
        )

        # Scale lidar mask up to BEV mask resolution
        lidar_ray_mask = F.interpolate(
            torch.tensor(lidar_ray_mask).unsqueeze(0).unsqueeze(0),
            size=[int(bev_max_x * 2), int(bev_max_z)],
            mode="bilinear"
        )
        lidar_ray_mask = (lidar_ray_mask.squeeze(0).squeeze(0) > 0).float().numpy()

        # Create FOV mask
        f, cu = cam_intrinsic[0, 0], cam_intrinsic[0, 2]
        fov_x_max = bev_max_z * scaling_factor / f * cu
        fov_x_zmax = np.arange(-int(fov_x_max), int(fov_x_max))
        fov_x_zmax[fov_x_zmax == 0] = 1
        fov_b_lines_end_pts = [
            np.array([[0, 0], [x, bev_max_z * scaling_factor]]) for x in fov_x_zmax
        ]
        fov_b_lines_pts = []
        for pts in fov_b_lines_end_pts:
            p = utils.connect2(pts)
            fov_b_lines_pts.append(p)

        fov_b_lines = np.concatenate(fov_b_lines_pts, axis=0).T

        fov_bev_crop = np.ones(fov_b_lines.shape[1], dtype=bool)
        fov_bev_crop = np.logical_and(
            fov_bev_crop, fov_b_lines[0] < bev_max_x * scaling_factor
        )
        fov_bev_crop = np.logical_and(
            fov_bev_crop, fov_b_lines[0] >= -bev_max_x * scaling_factor
        )
        fov_bev_crop = np.logical_and(
            fov_bev_crop, fov_b_lines[1] < bev_max_z * scaling_factor
        )
        fov_bev_crop = np.logical_and(fov_bev_crop, fov_b_lines[1] >= 0)
        fov_b_lines = fov_b_lines[:, fov_bev_crop]

        fov_b_lines[0, :] += bev_max_x * scaling_factor
        fov_b_lines = fov_b_lines.astype(int)
        fov_mask = np.zeros_like(bev_masks[0])
        fov_mask[fov_b_lines[1], fov_b_lines[0]] = 1
        fov_mask[
            np.arange(len(fov_mask)).astype(int),
            np.zeros(len(fov_mask), dtype=int) + len(fov_mask) // 2,
        ] = 1

        ### Visualise
        pts_cc[0, :] += bev_max_x
        pts_lidar_camcoords[0, :] += bev_max_x

        bev_max = np.max(
            bev_masks * (np.arange(len(bev_masks))[:, None, None] + 1), axis=0
        )

        # Plot BEV maps
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(bev_max, cmap="magma", vmin=0, vmax=len(layers))

        # Plot cells of bresenham lines
        ax.scatter(
            np.where(lidar_ray_mask > 0)[1],
            np.where(lidar_ray_mask > 0)[0],
            s=3,
            c="cyan"
        )
        ax.scatter(b_lines[0, :], b_lines[1, :], s=1, c="yellow")

        # Plot lidar points
        ax.scatter(pts_cc[0, :] / grid_res, pts_cc[2, :] / grid_res, s=8)
        ax.scatter(pts_lidar_camcoords[0, :] / grid_res, pts_lidar_camcoords[2, :] / grid_res, s=1, c='red')
        # ax.scatter(end_pts[:, 0] + bev_max_x, end_pts[:, 1], s=1)
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(bev_max * lidar_ray_mask, cmap="magma", vmin=0, vmax=len(layers))
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(bev_max * fov_mask, cmap="magma", vmin=0, vmax=len(layers))
        plt.show()

        for idx_mask, mask in enumerate(bev_masks):
            layer_name = layers[idx_mask]
            mask = (255 * mask).astype(np.uint8)
            mask_img = Image.fromarray(mask)
            mask_img.convert("L")
            mask_img.save(
                os.path.join(
                    semantic_maps_dir, (image_fn + "___" + layer_name + ".png")
                )
            )

        for idx_mask, mask in enumerate(bev_id_masks):
            layer_name = layers[idx_mask]

            # Check
            # print(((mask > 0).astype(float) == bev_masks[idx_mask]).all())

            # Normalise mask to [0, 1]
            max = mask.max()

            if max != 0:
                norm_mask = mask / max
            else:
                norm_mask = mask

            mask = (255 * norm_mask).astype(np.uint8)
            mask_img = Image.fromarray(mask)
            mask_img.convert("L")
            mask_img.save(
                os.path.join(
                    semantic_maps_dir, (image_fn + "___" + layer_name + "__instance_mask_upated.png")
                )
            )

        im = pil.Image.open(os.path.join(
                semantic_maps_dir, (image_fn + "___" + layer_name + "instance_mask.png")
            )
        )
        im_tensor = to_tensor(im)

        layer_name = "lidar_ray_mask_dense"
        mask = (255 * lidar_ray_mask).astype(np.uint8)
        mask_img = Image.fromarray(mask)
        mask_img.convert("L")
        mask_img.save(
            os.path.join(semantic_maps_dir, (image_fn + "___" + layer_name + ".png"))
        )

        layer_name = "fov_mask"
        mask = (255 * fov_mask).astype(np.uint8)
        mask_img = Image.fromarray(mask)
        mask_img.convert("L")
        mask_img.save(os.path.join(semantic_maps_dir, (layer_name + ".png")))


def sort_pts_by_ring_index(pts_cc, ring_cc):
    sort_idx_ring_cc = np.argsort(ring_cc)
    unique_rings, n_pts_per_ring = np.unique(ring_cc, return_counts=True)
    # Cumsum num pts per ring to use as indices later
    ring_cumsum = np.cumsum(n_pts_per_ring)
    n_pts_cumsum = np.zeros(len(ring_cumsum) + 1)
    n_pts_cumsum[1:] = ring_cumsum
    n_pts_cumsum = n_pts_cumsum.astype(int)
    # Jiggery pokery to get rid of lidar issues
    # Split points by ring index
    pts = pts_cc[:, sort_idx_ring_cc]
    pts_cc_by_r_index = [
        pts[:, n_pts_cumsum[i] : n_pts_cumsum[i + 1]]
        for i in range(len(n_pts_cumsum) - 1)
    ]
    # Sort each set of points at each ring along the x axis
    srt_idx_x = [np.argsort(pts[0]) for pts in pts_cc_by_r_index]
    srt_pts_cc_by_r_index = [
        pts[:, idx] for pts, idx in zip(pts_cc_by_r_index, srt_idx_x)
    ]
    # Pad each ring of pts to the same length so we can use arrays again
    max_length = np.max(n_pts_per_ring)
    sorted_pts_cc = np.zeros((len(srt_idx_x), 3, max_length))
    for i, pts_ring in enumerate(srt_pts_cc_by_r_index):
        pts_length = pts_ring.shape[1]
        sorted_pts_cc[i, :, -pts_length:] = pts_ring
    return sorted_pts_cc


def create_visibility_mask(bev_max_x, bev_max_z, pts_cc, scale_blines):
    """
    Creates
    :param bev_masks:
    :param bev_max_x:
    :param bev_max_z:
    :param pts_cc:
    :param scale_blines:
    :return:
    """
    end_pts = pts_cc[[0, 2], :].T * scale_blines
    end_pts = end_pts.astype(np.int)

    # # Clip z values minimum so you dont have lines which go from 0,0 to 0,0
    # end_pts[:, 1] = np.clip(end_pts[:, 1], a_min=2.0, a_max=None)
    #
    # # Add 2 to x values for same reason as above
    # end_pts[:, 0][np.argwhere(end_pts[:, 0]==0)][::2] = 2
    # end_pts[:, 0][np.argwhere(end_pts[:, 0] == 0)] = -2

    b_lines = np.concatenate(
        [
            utils.connect2(np.array([[0, 0], [end_pt[0], end_pt[1]]]))
            for end_pt in end_pts
        ],
        axis=0,
    ).T

    # Final mask to fit BEV mask size
    bev_crop = np.ones(b_lines.shape[1], dtype=bool)
    bev_crop = np.logical_and(bev_crop, b_lines[0] < bev_max_x * scale_blines)
    bev_crop = np.logical_and(bev_crop, b_lines[0] > -bev_max_x * scale_blines)
    bev_crop = np.logical_and(bev_crop, b_lines[1] < bev_max_z * scale_blines)
    bev_crop = np.logical_and(bev_crop, b_lines[1] >= 0)
    b_lines = b_lines[:, bev_crop].astype(float)

    # Translate X axis to BEV mask frame
    b_lines[0, :] += bev_max_x * scale_blines

    # Create lidar ray occlusion mask
    b_lines = b_lines.astype(int)
    lidar_ray_mask = np.zeros((200, 200))
    lidar_ray_mask[b_lines[1], b_lines[0]] = 1

    return b_lines, lidar_ray_mask


def render_object_in_image(
    rootdir=None,
    nusc: NuScenes = None,
    sample_token: str = None,
    sample_data_token: str = None,
    camera_channel: str = "CAM_FRONT",
    vehicle_layer_names: List[str] = None,
    return_attn_maps=False,
):
    """
    Render a nuScenes camera image and overlay the polygons for the
    specified map layers.
    Note that the projections are not always accurate as the localization is in 2d.
    :param nusc: The NuScenes instance to load the image from.
    :param sample_token: The image's corresponding sample_token.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param alpha: The transparency value of the layers to render in [0, 1].
    :param patch_radius: The radius in meters around the ego car in which to select map records.
    :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
    :param render_behind_cam: Whether to render polygons where any point is behind the camera.
    :param render_outside_im: Whether to render polygons where any point is outside the image.
    :param layer_names: The names of the layers to render, e.g. ['lane'].
        If set to None, the recommended setting will be used.
    :param verbose: Whether to print to stdout.
    :param out_path: Optional path to save the rendered figure to disk.
    """

    sample_record = nusc.get("sample", sample_token)

    # Grab the front camera image and intrinsics.
    cam_token = sample_record["data"][camera_channel]
    cam_record = nusc.get("sample_data", cam_token)
    cam_path = nusc.get_sample_data_path(cam_token)
    im = Image.open(cam_path)
    im_size = im.size

    # Set dimensions of BEV map
    im_canvas_size = im_size[::-1]
    max_z, max_x = 50, 25
    grid_res = 0.5

    # Ray length is the one from the camera to the corner of the BEV grid
    ray_length = int(np.sqrt((max_z / grid_res) ** 2 + (max_x / grid_res) ** 2)) + 1

    image_masks = []
    polar_maps = []

    sd_record = nusc.get("sample_data", sample_data_token)
    sensor_modality = sd_record["sensor_modality"]

    # Load boxes and image.
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(
        sample_data_token, box_vis_level=BoxVisibility.ANY
    )

    vehicles = vehicle_layer_names

    # Get image masks with depth values for mask
    for vehicle in vehicles:
        vehicle_layers = []
        polar_vehicle_layers = []

        # Add mask full of zeros as baseline case
        vehicle_layers.append(np.zeros(im_canvas_size))
        polar_vehicle_layers.append(np.zeros((ray_length, im_canvas_size[-1])))

        for box in boxes:
            if vehicle in box.name:
                # Make one layer mask per object
                layer_mask = np.zeros(im_canvas_size)
                # Create empty polar map and ray
                polar_ray = np.zeros(ray_length)
                polar_map = np.zeros((len(polar_ray), im_canvas_size[-1]))

                corners_cc = box.corners()
                corners_pc = view_points(corners_cc, camera_intrinsic, normalize=True)[
                    :2
                ]

                # Select 6 corners closest to camera to create object mask
                corners_norm = [np.linalg.norm(corner) for corner in corners_cc.T]
                closest_corners_idx = np.argsort(corners_norm)[:6]
                closest_corners_pc = corners_pc[:, closest_corners_idx]
                pts = closest_corners_pc.T

                # Sort points using convex hull
                hull = ConvexHull(pts)
                pts = pts[hull.vertices]

                poly_mask = np.zeros(im_canvas_size, np.uint8)
                pts_list = [(p0, p1) for (p0, p1) in zip(pts.T[0], pts.T[1])]
                poly = Polygon(pts_list)
                exteriors = [int_coords(poly.exterior.coords)]
                interiors = [int_coords(pi.coords) for pi in poly.interiors]
                offset = (int(0), int(0))
                cv2.fillPoly(poly_mask, exteriors, 1, offset=offset)
                cv2.fillPoly(poly_mask, interiors, 0, offset=offset)

                # Add polygon mask to layer mask with average depth
                layer_mask += poly_mask * corners_cc[-1].mean()

                # Create polar map
                # Get intersection points between polar ray and object box in cam coords
                bot_corners_cc = box.bottom_corners().T[:, [0, 2]]  # [N, 2]

                # If entire object is outside BEV bounds, move on
                if np.any(
                    np.all(bot_corners_cc[:, 0] > 25)
                    or np.all(bot_corners_cc[:, 0] < -25)
                ) or np.any(
                    np.all(bot_corners_cc[:, 1] > 49)
                    or np.all(bot_corners_cc[:, 1] < 0)
                ):
                    continue

                # Clip to BEV bounds
                ## TODO Needs to be bound along slope
                bot_corners_cc = np.hstack(
                    [
                        bot_corners_cc[:, 0].clip(-25, 25)[:, None],
                        bot_corners_cc[:, 1].clip(0, 49)[:, None],
                    ]
                )

                # Sort box bottom corners using convex hull
                bev_hull = ConvexHull(bot_corners_cc)
                hull_pts = bot_corners_cc[bev_hull.vertices]
                box_start_pts = hull_pts[:-1]
                box_end_pts = hull_pts[1:]
                box = np.hstack((box_start_pts, box_end_pts))

                # Create polar rays for each vertical scan line of image-plane object
                im_col_with_obj = (
                    layer_mask.sum(axis=0) > 0
                )  # mask for columns with object

                # Get x-axis location of all image columns in pix coords
                im_col_x_loc = (
                    np.arange(layer_mask.shape[-1]) - layer_mask.shape[-1] / 2
                )

                # X-axis location of positive image columns in pix coords
                pos_im_col = im_col_x_loc[im_col_with_obj]
                f = camera_intrinsic[0, 0]  # focal length in pixels

                # Create ray start and end points
                ray_start_pt = np.hstack(
                    (
                        np.zeros(len(pos_im_col))[:, None],
                        np.zeros(len(pos_im_col))[:, None],
                    )
                )
                ray_end_pt = np.hstack(
                    (pos_im_col[:, None], np.zeros(len(pos_im_col))[:, None] + f)
                )
                scaled_ray_end_pt = ray_end_pt * 100
                rays = np.hstack((ray_start_pt, scaled_ray_end_pt))

                # Intersect rays with box and add to polar map
                for idx_ray, ray in enumerate(rays):
                    intersect_pts = []
                    for line in box:
                        A = [ray[0], ray[1]]
                        B = [ray[2], ray[3]]
                        C = [line[0], line[1]]
                        D = [line[2], line[3]]
                        # Check if line segment and ray intersect
                        if utils.check_intersect(A, B, C, D):
                            intersect_pt = utils.get_intersect(A, B, C, D)
                            intersect_pts.append(intersect_pt)

                    # Sort points by z-axis
                    if len(intersect_pts) > 0:
                        sort_idx = np.argsort(np.array(intersect_pts)[:, 1])
                        sorted_int_pts = np.array(intersect_pts)[sort_idx]

                        # Get closes and furthest intersection point
                        first_pt = sorted_int_pts[0]
                        last_pt = sorted_int_pts[-1]
                        first_pt_dist = np.linalg.norm(first_pt)
                        last_pt_dist = np.linalg.norm(last_pt)
                        # length of object along ray
                        obj_length = last_pt_dist - first_pt_dist

                        # Add object to ray
                        if obj_length < 1:
                            polar_ray[
                                int(first_pt_dist / grid_res) : int(
                                    (first_pt_dist + 1) / grid_res
                                )
                            ] = 1
                        else:
                            polar_ray[
                                int(first_pt_dist / grid_res) : int(
                                    last_pt_dist / grid_res
                                )
                            ] = 1

                        col_loc = pos_im_col[idx_ray] + layer_mask.shape[-1] / 2
                        polar_map[:, int(col_loc)] += polar_ray

                # if polar_map.sum() == 0:
                #     print('wtf')
                #
                # if (layer_mask.sum() > 0) != (polar_map.sum() > 0):
                #     print("wait")

                vehicle_layers.append(layer_mask)
                polar_vehicle_layers.append(polar_map)

        if len(polar_vehicle_layers) != len(vehicle_layers):
            print(len(polar_vehicle_layers), len(vehicle_layers))
            print("stop")

        image_masks.append(vehicle_layers)
        polar_maps.append(polar_vehicle_layers)

    plt.close("all")

    return image_masks, polar_maps
