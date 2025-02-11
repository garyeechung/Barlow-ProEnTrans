from typing import Tuple, Union

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.utils import get_labels_from_coords


COCO_JSON_FILE = "data/fiftyone/coco-2017/validation/labels.json"


class CocoMaskAndPoints:
    def __init__(self, coco_json_file: str = COCO_JSON_FILE,
                 min_area_ratio: float = 0.0,
                 max_area_ratio: float = 1.0,
                 image_size: int = 256, nb_copies: int = 5,
                 nb_positives: Union[int, Tuple[int, int]] = (1, 5),
                 nb_negatives: Union[int, Tuple[int, int]] = (1, 5),
                 to_xy: bool = True, return_side_class: bool = True,
                 validation: bool = True) -> None:
        assert 0 <= min_area_ratio <= max_area_ratio <= 1, "Invalid ratios"
        self.coco = COCO(coco_json_file)

        if min_area_ratio == 0.0 and max_area_ratio == 1.0:
            self.ann_ids = self.coco.getAnnIds()
        else:
            self.ann_ids = []
            for ann_id in self.coco.getAnnIds():
                mask = self.coco.annToMask(self.coco.loadAnns(ann_id)[0])
                mask_area = mask.sum()
                total_area = np.prod(mask.shape)
                area_ratio = mask_area / total_area
                if min_area_ratio <= area_ratio <= max_area_ratio:
                    self.ann_ids.append(ann_id)

        self.image_size = image_size
        self.nb_copies = nb_copies
        self.nb_positives = nb_positives
        self.nb_negatives = nb_negatives
        self.to_xy = to_xy
        self.return_side_class = return_side_class
        self.validation = validation
        if not self.validation:
            self.transform = T.Compose([
                # T.Resize((256, 256), interpolation=Image.NEAREST),  # Resize to 256x256
                T.RandomRotation(30, interpolation=Image.NEAREST),  # Rotate by ±30°
                T.RandomHorizontalFlip(p=0.5),  # Flip horizontally with 50% chance
                T.RandomVerticalFlip(p=0.5),  # Flip vertically with 50% chance
                T.RandomAffine(degrees=0, translate=(0.2, 0.2), interpolation=Image.NEAREST),  # Shift
            ])

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
        if not self.validation:
            mask = self.transform(mask)
        mask = mask.squeeze(0).squeeze(0)
        mask = mask > 0.5
        mask = mask.float()

        samples = []

        for _ in range(self.nb_copies):
            if self.validation:
                torch.manual_seed(42)
            else:
                torch.manual_seed(torch.randint(0, 100000, (1,)).item())

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
            sample = {
                'masks': mask,
                'point_coords': coords,
                'point_labels': labels
            }
            if self.return_side_class:
                img_mid = self.image_size // 2
                sums = mask.sum(dim=1)
                dim_0_1st_half_more = sums[:img_mid].sum() > sums[img_mid:].sum()
                sums = mask.sum(dim=0)
                dim_1_1st_half_more = sums[:img_mid].sum() > sums[img_mid:].sum()
                side_class = torch.tensor([dim_0_1st_half_more, dim_1_1st_half_more]).float()
                sample['side_class'] = side_class

            samples.append(sample)

        return samples


def batch_in_batch_out_fn(batch, device='cuda'):
    return batch


def flatten_collate_fn(batch):
    flattened_batch = [d for sublist in batch for d in sublist]
    return flattened_batch
