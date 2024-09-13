# Suppose that the dataset structure is like
# dataset
# ├── model_0
# │   ├── info.json
# │   ├── model.io
# │   ├── model.ldr
# │   └── voxels.npy # 3D numpy array of shape (24, 40, 24)
# ├── model_1
# ......
#

import os
import torch
import numpy as np

from torch.utils.data import Dataset

from Utils.BrickSet.brick_ids import class_id_to_brick_id


def get_id2label():
    """Mapping from class index to human-readable label."""
    return class_id_to_brick_id


class LEGOOccupancyDataset(Dataset):
    def __init__(self, dataset_dir, num_classes=24):
        self.num_classes = num_classes

        self.occupancy_file_paths = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith("combined_tensor.npy"):
                    self.occupancy_file_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.occupancy_file_paths)

    def __getitem__(self, idx):
        occupancy_map_file = self.occupancy_file_paths[idx]
        occupancy_map = np.load(occupancy_map_file)
        label = torch.tensor(occupancy_map).long()
        brick_channels = []
        # Note: id 0 represents empty space
        # id self.num_classes + 1 represents the stud voxels
        # id self.num_classes + 2 represents the tube voxels
        # Therefore, there are self.num_classes + 2 classes.
        for idx in range(0, self.num_classes + 3):
            brick_channel = np.zeros_like(occupancy_map, dtype=np.uint8)
            brick_channel[occupancy_map == idx] = 1
            brick_channels.append(brick_channel)

        data = np.stack(brick_channels, axis=0)
        data = torch.tensor(data, dtype=torch.float32)
        data = data * 2 - 1
        return data, label
