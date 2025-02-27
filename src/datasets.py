import logging
import os
import random
from typing import Tuple, Union

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.utils import get_labels_from_coords, get_center_by_erosion


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COCO_JSON_FILE = "data/fiftyone/coco-2017/validation/labels.json"
DATA_DIR_TRAIN = "data/fiftyone/coco-2017/validation"


class CocoMaskAndPoints:
    def __init__(self, coco_json_file: str = COCO_JSON_FILE,
                 min_area_ratio: float = 0.0,
                 max_area_ratio: float = 1.0,
                 image_size: int = 1024,
                 nb_copies: int = 8,
                 nb_positives: Union[int, Tuple[int, int]] = (1, 5),
                 nb_negatives: Union[int, Tuple[int, int]] = (1, 5),
                 to_xy: bool = True, return_side_class: bool = True,
                 validation: bool = False) -> None:
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


class CocoCenterPointPrompt:

    def __init__(self, data_dir: str = DATA_DIR_TRAIN,
                 min_mask_num: int = 1,
                 max_mask_num: int = 1000,
                 image_size: int = 1024,
                 to_xy: bool = True,
                 validation: bool = False,
                 max_nb_masks: int = 16) -> None:

        self.coco = COCO(os.path.join(data_dir, "labels.json"))
        self.data_dir = os.path.join(data_dir, "data")

        self.cat_ids = self.coco.getCatIds()
        self.img_ids = self.coco.getImgIds()
        self.img_ids = []
        for idx in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            if min_mask_num <= len(anns) <= max_mask_num:
                self.img_ids.append(idx)

        self.image_size = image_size
        self.to_xy = to_xy
        self.validation = validation
        self.max_nb_masks = max_nb_masks

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        sample_info = self.coco.loadImgs(self.img_ids[idx])[0]
        img_path = os.path.join(self.data_dir, sample_info['file_name'])
        image = Image.open(img_path)
        image = self.get_image_tensor(image)

        segmentations = self.coco.loadAnns(
            self.coco.getAnnIds(imgIds=self.img_ids[idx], iscrowd=None)
        )
        if len(segmentations) == 0:
            pseudo_mask = np.zeros((self.image_size, self.image_size))
            masks = [self.get_mask_tensor(pseudo_mask)]
        else:
            segmentations = random.sample(segmentations,
                                          min(len(segmentations),
                                              self.max_nb_masks))
            logger.debug(f"segmentations: {len(segmentations)}")
            masks = [self.get_mask_tensor(self.coco.annToMask(seg))
                     for seg in segmentations]
        masks = torch.concat(masks, dim=0) * 255
        masks = masks > 0.5
        masks = masks.float()
        # make masks' edges zero
        masks[:, 0, :] = 0
        masks[:, -1, :] = 0
        masks[:, :, 0] = 0
        masks[:, :, -1] = 0

        point_coords = get_center_by_erosion(masks, max_iter=30)
        point_labels = get_labels_from_coords(masks, point_coords)

        sample = {
            'image': image,
            'masks': masks,
            'point_coords': point_coords,
            'point_labels': point_labels,
        }
        return sample

    def get_image_tensor(self, image: Image):
        image = image.convert('RGB')
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def get_mask_tensor(self, mask: np.ndarray):
        mask = Image.fromarray(mask.astype('uint8'), 'L')

        transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])
        return transform(mask)


def batch_in_batch_out_fn(batch, device='cuda'):
    return batch


def flatten_collate_fn(batch):
    flattened_batch = [d for sublist in batch for d in sublist]
    return flattened_batch
