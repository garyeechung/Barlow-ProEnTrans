import logging
import random

import numpy as np
import torch

from src.utils import get_labels_from_coords


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LineSeperateMasksPairs:
    def __init__(self, image_size: int = 1024, nb_prompts: int = 10,
                 nb_pairs: int = 1, angle_diff: float = 90.0) -> None:
        self.image_size = image_size
        self.nb_prompts = nb_prompts
        self.nb_pairs = nb_pairs

        assert 0 <= angle_diff <= 180, "Invalid angle difference"
        self.angle_diff = angle_diff

    def __len__(self) -> int:
        return self.nb_pairs

    def __getitem__(self, idx: int):
        center = (self.image_size // 2, self.image_size // 2)
        mesh = np.meshgrid(np.arange(self.image_size),
                           np.arange(self.image_size))
        coordinates = np.stack(mesh, axis=-1).reshape(-1, 2)

        angle_0 = random.uniform(0, 360)
        slope_0 = np.tan(np.deg2rad(angle_0))
        inter_0 = center[1] - slope_0 * center[0]
        mask_0 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if 90 < angle_0 < 270:
            mask_0[mesh[1] - slope_0 * mesh[0] < inter_0] = 1
        else:
            mask_0[mesh[1] - slope_0 * mesh[0] > inter_0] = 1
        mask_0 = torch.tensor(mask_0).unsqueeze(0).float()
        np.random.shuffle(coordinates)
        coords_0 = coordinates[:self.nb_prompts]
        coords_0 = torch.tensor(coords_0.copy()).unsqueeze(0)
        coords_0 = coords_0.flip(-1)
        labels_0 = get_labels_from_coords(mask_0, coords_0)
        sample_0 = {"masks": mask_0, "angle": angle_0, "coords": coords_0, "labels": labels_0}

        angle_1 = (angle_0 + self.angle_diff) % 360
        slope_1 = np.tan(np.deg2rad(angle_1))
        inter_1 = center[1] - slope_1 * center[0]
        mask_1 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if 90 < angle_1 < 270:
            mask_1[mesh[1] - slope_1 * mesh[0] < inter_1] = 1
        else:
            mask_1[mesh[1] - slope_1 * mesh[0] > inter_1] = 1
        mask_1 = torch.tensor(mask_1).unsqueeze(0).float()
        np.random.shuffle(coordinates)
        coords_1 = coordinates[:self.nb_prompts]
        coords_1 = torch.tensor(coords_1.copy()).unsqueeze(0)
        coords_1 = coords_1.flip(-1)
        labels_1 = get_labels_from_coords(mask_1, coords_1)
        sample_1 = {"masks": mask_1, "angle": angle_1, "coords": coords_1, "labels": labels_1}

        return sample_0, sample_1
