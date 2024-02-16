import os
from torch.utils.data import DataLoader, RandomSampler
from .augmentations import AugmentedMapDataset

from nuscenes import NuScenes
from .nuscenes.dataset import NuScenesMapDataset
from .nuscenes.splits import TRAIN_SCENES, VAL_SCENES, CALIBRATION_SCENES


def build_nuscenes_datasets(config):
    print('==> Loading NuScenes dataset...')
    nuscenes = NuScenes(config.nuscenes_version,
                        os.path.expandvars(config.dataroot))

    # Exclude calibration scenes
    if config.hold_out_calibration:
        train_scenes = list(set(TRAIN_SCENES) - set(CALIBRATION_SCENES))
    else:
        train_scenes = TRAIN_SCENES

    train_data = NuScenesMapDataset(nuscenes, config.label_root,
                                    config.img_size, train_scenes)
    val_data = NuScenesMapDataset(nuscenes, config.label_root,
                                  config.img_size, VAL_SCENES)
    return train_data, val_data

def build_datasets(dataset_name, config):
    if dataset_name == 'nuscenes':
        return build_nuscenes_datasets(config)
    else:
        raise ValueError(f"Unknown dataset option '{dataset_name}'")


def build_trainval_datasets(dataset_name, config):
    # Construct the base dataset
    train_data, val_data = build_datasets(dataset_name, config)

    # Add data augmentation to train dataset
    train_data = AugmentedMapDataset(train_data, config.hflip)

    return train_data, val_data


def build_dataloaders(dataset_name, config):
    # Build training and validation datasets
    train_data, val_data = build_trainval_datasets(dataset_name, config)

    # Create training set dataloader
    sampler = RandomSampler(train_data, True, config.epoch_size)
    train_loader = DataLoader(train_data, config.batch_size, sampler=sampler,
                              num_workers=config.num_workers)

    # Create validation dataloader
    val_loader = DataLoader(val_data, config.batch_size,
                            num_workers=config.num_workers)

    return train_loader, val_loader






