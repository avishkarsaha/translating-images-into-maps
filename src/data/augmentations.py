import torch
from torch.utils.data import Dataset


class AugmentedMapDataset(Dataset):

    def __init__(self, dataset, hflip=True, desired_image_size=None):
        self.dataset = dataset
        self.hflip = hflip
        self.desired_image_size = desired_image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, calib, labels, mask = self.dataset[index]

        # Apply data augmentation
        if self.hflip:
            image, labels, mask = random_hflip(image, labels, mask)
        ## TODO: add padding+cropping augmentation (copied from the original dataloader) - don't we need to change the gt and mask OR the z_intervals when the image is being changed?
        # if self.desired_image_size:
        #     image, calib = image_calib_pad_and_crop(image, calib, desired_image_size)

        return image, calib, labels, mask


def random_hflip(image, labels, mask):
    image = torch.flip(image, (-1,))
    labels = torch.flip(labels.int(), (-1,)).bool()
    mask = torch.flip(mask.int(), (-1,)).bool()
    return image, labels, mask

def image_calib_pad_and_crop(image, calib, desired_image_size):

        og_w, og_h = image.shape[-2:]
        desired_w, desired_h = desired_image_size
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
