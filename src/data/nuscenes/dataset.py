import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from nuscenes import NuScenes
from torchvision.transforms.functional import to_tensor

from .utils import CAMERA_NAMES, NUSCENES_CLASS_NAMES, iterate_samples
from ..utils import decode_binary_labels

class NuScenesMapDataset(Dataset):

    def __init__(self, nuscenes, map_root,  image_size=(800, 450), 
                 scene_names=None):
        
        self.nuscenes = nuscenes
        self.map_root = os.path.expandvars(map_root)
        self.image_size = image_size

        # Preload the list of tokens in the dataset
        self.get_tokens(scene_names)

        # Allow PIL to load partially corrupted images
        # (otherwise training crashes at the most inconvenient possible times!)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    

    def get_tokens(self, scene_names=None):
        
        self.tokens = list()

        # Iterate over scenes
        for scene in self.nuscenes.scene:
            
            # Ignore scenes which don't belong to the current split
            if scene_names is not None and scene['name'] not in scene_names:
                continue
             
            # Iterate over samples
            for sample in iterate_samples(self.nuscenes, 
                                          scene['first_sample_token']):
                
                # Iterate over cameras
                for camera in CAMERA_NAMES:
                    self.tokens.append(sample['data'][camera])
        
        return self.tokens


    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]

        image = self.load_image(token)
        calib = self.load_calib(token)
        labels, mask = self.load_labels(token)

        return image, calib, labels, mask

    
    def load_image(self, token):

        # Load image as a PIL image
        image = Image.open(self.nuscenes.get_sample_data_path(token))

        # Resize to input resolution
        image = image.resize(self.image_size)

        # Convert to a torch tensor
        return to_tensor(image)
    

    def load_calib(self, token):

        # Load camera intrinsics matrix
        sample_data = self.nuscenes.get('sample_data', token)
        sensor = self.nuscenes.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = torch.tensor(sensor['camera_intrinsic'])

        # Scale calibration matrix to account for image downsampling
        intrinsics[0] *= self.image_size[0] / sample_data['width']
        intrinsics[1] *= self.image_size[1] / sample_data['height']
        return intrinsics
    

    def load_labels(self, token):

        # Load label image as a torch tensor
        label_path = os.path.join(self.map_root, token + '.png')
        encoded_labels = to_tensor(Image.open(label_path)).long()

        # Decode to binary labels
        num_class = len(NUSCENES_CLASS_NAMES)
        labels = decode_binary_labels(encoded_labels, num_class + 1)
        labels, mask = labels[:-1], ~labels[-1]

        return labels, mask
    


