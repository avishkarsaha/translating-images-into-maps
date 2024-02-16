import os
import numpy as np
from shapely import geometry, affinity
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.utils.data_classes import LidarPointCloud

from ..utils import transform_polygon, render_polygon, transform

CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

NUSCENES_CLASS_NAMES = [
    'drivable_area', 'ped_crossing', 'walkway', 'carpark', 'car', 'truck', 
    'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 
    'bicycle', 'traffic_cone', 'barrier'
]

STATIC_CLASSES = ['drivable_area', 'ped_crossing', 'walkway', 'carpark_area']

LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']


def iterate_samples(nuscenes, start_token):
    sample_token = start_token
    while sample_token != '':
        sample = nuscenes.get('sample', sample_token)
        yield sample
        sample_token = sample['next']
    

def get_map_masks(nuscenes, map_data, sample_data, extents, resolution):

    # Render each layer sequentially
    layers = [get_layer_mask(nuscenes, polys, sample_data, extents, 
              resolution) for layer, polys in map_data.items()]

    return np.stack(layers, axis=0)


def get_layer_mask(nuscenes, polygons, sample_data, extents, resolution):

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm)

    # Initialise the map mask
    x1, z1, x2, z2 = extents
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
                    dtype=np.uint8)

    # Find all polygons which intersect with the area of interest
    for polygon in polygons.query(map_patch):

        polygon = polygon.intersection(map_patch)
        
        # Transform into map coordinates
        polygon = transform_polygon(polygon, inv_tfm)

        # Render the polygon to the mask
        render_shapely_polygon(mask, polygon, extents, resolution)
    
    return mask.astype(np.bool)




def get_object_masks(nuscenes, sample_data, extents, resolution):

    # Initialize object masks
    nclass = len(DETECTION_NAMES) + 1
    grid_width = int((extents[2] - extents[0]) / resolution)
    grid_height = int((extents[3] - extents[1]) / resolution)
    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    for box in nuscenes.get_boxes(sample_data['token']):

        # Get the index of the class
        det_name = category_to_detection_name(box.name)
        if det_name not in DETECTION_NAMES:
            class_id = -1
        else:
            class_id = DETECTION_NAMES.index(det_name)
        
        # Get bounding box coordinates in the grid coordinate frame
        bbox = box.bottom_corners()[:2]
        local_bbox = np.dot(inv_tfm[:2, :2], bbox).T + inv_tfm[:2, 2]

        # Render the rotated bounding box to the mask
        render_polygon(masks[class_id], local_bbox, extents, resolution)
    
    return masks.astype(np.bool)


def get_sensor_transform(nuscenes, sample_data):

    # Load sensor transform data
    sensor = nuscenes.get(
        'calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_tfm = make_transform_matrix(sensor)

    # Load ego pose data
    pose = nuscenes.get('ego_pose', sample_data['ego_pose_token'])
    pose_tfm = make_transform_matrix(pose)

    return np.dot(pose_tfm, sensor_tfm)


def load_point_cloud(nuscenes, sample_data):

    # Load point cloud
    lidar_path = os.path.join(nuscenes.dataroot, sample_data['filename'])
    pcl = LidarPointCloud.from_file(lidar_path)
    return pcl.points[:3, :].T


def make_transform_matrix(record):
    """
    Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
    """
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    transform[:3, 3] = np.array(record['translation'])
    return transform


def render_shapely_polygon(mask, polygon, extents, resolution):

    if polygon.geom_type == 'Polygon':

        # Render exteriors
        render_polygon(mask, polygon.exterior.coords, extents, resolution, 1)

        # Render interiors
        for hole in polygon.interiors:
            render_polygon(mask, hole.coords, extents, resolution, 0)
    
    # Handle the case of compound shapes
    else:
        for poly in polygon:
            render_shapely_polygon(mask, poly, extents, resolution)




