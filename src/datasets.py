from typing import Tuple, Union

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F

from src.utils import get_labels_from_coords


COCO_JSON_FILE = "data/fiftyone/coco-2017/validation/labels.json"


class CocoMaskAndPoints:
    def __init__(self, coco_json_file: str = COCO_JSON_FILE,
                 min_area_ratio: float = 0.1,
                 max_area_ratio: float = 0.7,
                 image_size: int = 256, nb_samples: int = 5,
                 nb_positives: Union[int, Tuple[int, int]] = (1, 5),
                 nb_negatives: Union[int, Tuple[int, int]] = (1, 5),
                 to_xy: bool = True) -> None:
        assert 0 <= min_area_ratio <= max_area_ratio <= 1, "Invalid ratios"
        self.coco = COCO(coco_json_file)

        self.ann_ids = []
        for ann_id in self.coco.getAnnIds():
            mask = self.coco.annToMask(self.coco.loadAnns(ann_id)[0])
            mask_area = mask.sum()
            total_area = np.prod(mask.shape)
            area_ratio = mask_area / total_area
            if min_area_ratio <= area_ratio <= max_area_ratio:
                self.ann_ids.append(ann_id)

        self.image_size = image_size
        self.nb_samples = nb_samples
        self.nb_positives = nb_positives
        self.nb_negatives = nb_negatives
        self.to_xy = to_xy

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, idx):
        ann_id = self.ann_ids[idx]
        ann = self.coco.loadAnns(ann_id)[0]
        mask = self.coco.annToMask(ann)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(self.image_size, self.image_size),
                             mode='bilinear', align_corners=False)
        mask = mask.squeeze(0).squeeze(0)
        mask = mask > 0.5
        mask = mask.float()

        samples = []

        for _ in range(self.nb_samples):
            # torch.manual_seed(torch.randint(0, 100000, (1,)).item())

            if isinstance(self.nb_positives, tuple):
                nb_positives = np.random.randint(self.nb_positives[0], self.nb_positives[1] + 1)
            else:
                nb_positives = self.nb_positives
            pos_coords = torch.stack(torch.where(mask > 0.5), axis=-1)
            pos_coords = pos_coords[torch.randperm(pos_coords.size(0))[:nb_positives]]

            if isinstance(self.nb_negatives, tuple):
                nb_negatives = np.random.randint(self.nb_negatives[0], self.nb_negatives[1] + 1)
            else:
                nb_negatives = self.nb_negatives
            neg_coords = torch.stack(torch.where(mask < 0.5), axis=-1)
            neg_coords = neg_coords[torch.randperm(neg_coords.size(0))[:nb_negatives]]

            coords = torch.cat([pos_coords, neg_coords], dim=0)
            if self.to_xy:
                coords = coords.flip(-1)
            coords = coords.unsqueeze(0).float()

            labels = get_labels_from_coords(mask.unsqueeze(0), coords, xy_coord=self.to_xy)
            samples.append((mask, coords, labels))

        return samples