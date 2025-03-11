from .coco import CocoCenterPointPrompt, CocoMaskAndPoints
from .flare import FLAREDataset2D, FLAREDatasetCache2D


__all__ = ['CocoCenterPointPrompt', 'CocoMaskAndPoints',
           'FLAREDataset2D', 'FLAREDatasetCache2D']


def batch_in_batch_out_fn(batch, device='cuda'):
    return batch


def flatten_collate_fn(batch):
    flattened_batch = [d for sublist in batch for d in sublist]
    return flattened_batch
