import logging
# import os
import random
from time import localtime, strftime

from segment_anything import sam_model_registry
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from src.datasets import CocoCenterPointPrompt, batch_in_batch_out_fn
from src.losses import SoftDiceLoss
from src.modeling.interact_sam import HeuristicSAM
from src.modeling.proentrans import BarlowTwinsCosineSimilarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAM_CHECKPOINT = "model_checkpoints/sam_vit_h_4b8939.pth"
BARLOW_CHECKPOINT = "model_checkpoints/Barlow/barlow_20250213235831.pth"
MODEL_TYPE = "vit_h"
COCO_FILE_TRAIN = "data/fiftyone/coco-2017/train"


def finetune_heuristic_sam(sam_checkpoint: str,
                           barlow_checkpoint: str,
                           model_type: str,
                           coco_file_train: str,
                           steps_range: tuple = (2, 15),
                           random_seed: int = 42,
                           epochs: int = 5, learning_rate: float = 1e-4,
                           batch_size: int = 128,
                           valid_size: int = 1000,
                           pseudo_train_valid_ratio: int = 4,
                           max_nb_masks: int = 16) -> None:
    time_now = strftime('%Y%m%d%H%M%S', localtime())
    torch.manual_seed(random_seed)
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    wandb.init(project="BarlowTwins", name=f"sam-{time_now}",
               entity="garyeechung-vanderbilt-university",
               job_type="SAM", group="COCO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device)
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    blt = BarlowTwinsCosineSimilarity(sam=sam)
    blt_states = torch.load(barlow_checkpoint)
    blt.load_state_dict(blt_states["model"], strict=True)
    proentrans = blt.proentrans
    proentrans.to(device)
    for param in proentrans.parameters():
        param.requires_grad = False

    interactsam = HeuristicSAM(sam, proentrans)
    params = list(interactsam.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    loss_fn = SoftDiceLoss()

    dataset = CocoCenterPointPrompt(coco_file_train, max_nb_masks=16)
    dataset_train, dataset_valid = random_split(dataset,
                                                [len(dataset) - valid_size, valid_size])
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, collate_fn=batch_in_batch_out_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1,
                                  shuffle=False, collate_fn=batch_in_batch_out_fn)
    logger.info(f"len(dataset_train): {len(dataset_train)}")
    logger.info(f"len(dataset_valid): {len(dataset_valid)}")
    valid_per_n_step = pseudo_train_valid_ratio * len(dataset_valid) // batch_size
    logger.info(f"valid_per_n_step: {valid_per_n_step}")

    best_val_loss = float("inf")
    for epoch in range(epochs):
        for i, samples_train in tqdm(enumerate(dataloader_train)):

            # validation
            if valid_per_n_step > 0 and i % valid_per_n_step == 0:
                with torch.no_grad():
                    loss_val = 0.
                    for j, sample in enumerate(dataloader_valid, 1):
                        logger.info(f"Validating: {j: 3d}/{len(dataloader_valid)}")
                        sample = {k: v.to(device)
                                  for k, v in sample[0].items()
                                  if isinstance(v, torch.Tensor)}
                        steps = random.randint(*steps_range)
                        preds = interactsam(**sample, steps=steps)
                        loss_val += loss_fn(preds, sample["masks"]).item()
                    loss_val /= j
                    wandb.log({"loss_val": loss_val})
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    save_dict = {"model": interactsam.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "best_val_loss": best_val_loss,
                                 "epoch": epoch}
                    torch.save(save_dict, f"model_checkpoints/sam_{time_now}.pth")

            # training
            optimizer.zero_grad()
            loss_trn = 0.
            for sample in samples_train:
                sample = {k: v.to(device)
                          for k, v in sample.items()
                          if isinstance(v, torch.Tensor)}
                steps = random.randint(*steps_range)
                preds = interactsam(**sample, steps=steps)
                loss = loss_fn(preds, sample["masks"])
                loss.backward()
                loss_trn += loss.item()
            optimizer.step()
            loss_trn /= len(samples_train)
            wandb.log({"loss": loss_trn})

        # validation
        if (valid_per_n_step > 0 and i % valid_per_n_step != 0) or valid_per_n_step == 0:
            with torch.no_grad():
                loss_val = 0.
                for j, samples in enumerate(dataloader_valid, 1):
                    logger.info(f"Validating: {j: 3d}/{len(dataloader_valid)}")
                    sample = {k: v.to(device)
                              for k, v in samples[0].items()
                              if isinstance(v, torch.Tensor)}
                    steps = random.randint(*steps_range)
                    preds = interactsam(**sample, steps=steps)
                    loss_val += loss_fn(preds, sample["masks"]).item()
                loss_val /= j
                wandb.log({"loss_val": loss_val})
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                save_dict = {"model": interactsam.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "best_val_loss": best_val_loss,
                             "epoch": epoch}
                torch.save(save_dict, f"model_checkpoints/sam_{time_now}.pth")

    wandb.finish()


if __name__ == "__main__":
    finetune_heuristic_sam(sam_checkpoint=SAM_CHECKPOINT,
                           barlow_checkpoint=BARLOW_CHECKPOINT,
                           model_type=MODEL_TYPE,
                           coco_file_train=COCO_FILE_TRAIN,
                           steps_range=(2, 16),
                           batch_size=4,
                           valid_size=64,
                           max_nb_masks=8)
