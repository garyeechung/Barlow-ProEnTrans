import json
import logging
from time import localtime, strftime

from segment_anything import sam_model_registry
import torch
from torch.nn import MSELoss, DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..datasets import CocoMaskAndPoints, flatten_collate_fn
from ..modeling.proentrans import BarlowTwinsCosineSimilarity
from ..utils import get_pairwise_dices

SAM_CHECKPOINT = "model_checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
COCO_FILE_TRAIN = "data/fiftyone/coco-2017/train/labels.json"
# COCO_FILE_VALID = "data/fiftyone/coco-2017/validation/labels.json"
DATA_SPLIT_JSON = "data/fiftyone/coco-2017/train/train_valid_split.json"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_barlow(sam_checkpoint: str,
                 model_type: str,
                 coco_file_train: str,
                 coco_file_valid: str,
                 data_split_json: str = None,
                 random_seed: int = 42, epochs: int = 1000,
                 learning_rate: float = 1e-4, batch_size: int = 8,
                 nb_copies: int = 8,
                 parallel: bool = False,
                 nb_train: int = 0,
                 nb_valid: int = 0,
                 pseudo_train_valid_ratio: int = 4
                 ) -> None:

    time_now = strftime('%Y%m%d%H%M%S', localtime())
    torch.manual_seed(random_seed)
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    wandb.init(project="BarlowTwins", name=f"blt-{time_now}",
               entity="garyeechung-vanderbilt-university",
               job_type="BarlowTwins", group="COCO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sam = sam_model_registry[model_type](sam_checkpoint)
    barlow_twins = BarlowTwinsCosineSimilarity(sam=sam, nb_copies=nb_copies)
    if parallel:
        barlow_twins = DataParallel(barlow_twins)
    barlow_twins = barlow_twins.to(device)

    if data_split_json is not None:
        with open(data_split_json, "r") as f:
            data_split = json.load(f)
            ann_ids_train = data_split["train"]
            ann_ids_valid = data_split["valid"]
    else:
        ann_ids_train = None
        ann_ids_valid = None

    coco_dataset_valid = CocoMaskAndPoints(coco_file_valid,
                                           nb_positives=(1, 10),
                                           nb_negatives=(1, 10),
                                           min_area_ratio=0.,
                                           max_area_ratio=1.,
                                           nb_copies=nb_copies,
                                           validation=True)
    if ann_ids_valid is not None:
        coco_dataset_valid.ann_ids = ann_ids_valid
    if nb_valid > 0:
        coco_dataset_valid.ann_ids = coco_dataset_valid.ann_ids[:nb_valid]
    coco_loader_valid = DataLoader(coco_dataset_valid, batch_size=nb_copies,
                                   shuffle=False,
                                   collate_fn=flatten_collate_fn)

    coco_dataset_train = CocoMaskAndPoints(coco_file_train,
                                           nb_positives=(1, 10),
                                           nb_negatives=(1, 10),
                                           min_area_ratio=0.,
                                           max_area_ratio=1.,
                                           nb_copies=nb_copies,
                                           validation=False)
    if ann_ids_train is not None:
        coco_dataset_train.ann_ids = ann_ids_train
    if nb_train > 0:
        coco_dataset_train.ann_ids = coco_dataset_train.ann_ids[:nb_train]
    coco_loader_train = DataLoader(coco_dataset_train,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=flatten_collate_fn)
    logger.info(f"Number of samples in training dataset: {len(coco_dataset_train)}")
    logger.info(f"Number of samples in validation dataset: {len(coco_dataset_valid)}")
    valid_per_n_step = pseudo_train_valid_ratio * len(coco_dataset_valid) // batch_size
    logger.info(f"Validating every {valid_per_n_step} steps")

    mse = MSELoss()
    optimizer = Adam(barlow_twins.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        for j, samples_trn in tqdm(enumerate(coco_loader_train)):
            # Validation
            if valid_per_n_step > 0 and j % valid_per_n_step == 0:
                with torch.no_grad():
                    loss = 0.
                    for i, samples_val in enumerate(coco_loader_valid, 1):
                        if i % 50 == 0:
                            print(f"Validating: {i}/{len(coco_loader_valid)}", end="\r")
                        for sample in samples_val:
                            for key, value in sample.items():
                                if isinstance(value, torch.Tensor):
                                    sample[key] = value.to(device)
                        dices = get_pairwise_dices(samples_val, nb_copies)
                        dices = dices.to(device)
                        cos_sims = barlow_twins(samples_val)
                        loss += mse(dices, cos_sims).item()
                    loss /= i
                    wandb.log({"val_loss": loss})
                    print(f"\nEpoch {epoch:03d}: validation loss: {loss}")
                    if loss < best_val_loss:
                        best_val_loss = loss
                        if parallel:
                            model_dict = barlow_twins.module.state_dict()
                        else:
                            model_dict = barlow_twins.state_dict()
                        save_dict = {"model": model_dict,
                                     "optimizer": optimizer.state_dict(),
                                     "best_val_loss": best_val_loss,
                                     "epoch": epoch}
                        torch.save(save_dict, f"model_checkpoints/Barlow/barlow_{time_now}.pth")

            # Training
            barlow_twins.train()
            for sample in samples_trn:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.to(device)
            optimizer.zero_grad()
            dices = get_pairwise_dices(samples_trn, nb_copies)
            dices = dices.to(device)
            cos_sims = barlow_twins(samples_trn)
            loss = mse(dices, cos_sims)
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})

        # Validation
        with torch.no_grad():
            loss = 0.
            for i, samples_val in tqdm(enumerate(coco_loader_valid, 1)):
                if i % 25 == 0:
                    print(f"Validating: {i}/{len(coco_loader_valid)}", end="\r")
                for sample in samples_val:
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            sample[key] = value.to(device)
                dices = get_pairwise_dices(samples_val, nb_copies)
                dices = dices.to(device)
                cos_sims = barlow_twins(samples_val)
                loss += mse(dices, cos_sims).item()
            loss /= i
            wandb.log({"val_loss": loss})
            print(f"\nEpoch {epoch:03d}: validation loss: {loss}")

        if loss < best_val_loss:
            best_val_loss = loss
            if parallel:
                model_dict = barlow_twins.module.state_dict()
            else:
                model_dict = barlow_twins.state_dict()
            save_dict = {"model": model_dict,
                         "optimizer": optimizer.state_dict(),
                         "best_val_loss": best_val_loss,
                         "epoch": epoch}
            torch.save(save_dict, f"model_checkpoints/Barlow/barlow_{time_now}.pth")

    wandb.finish()


if __name__ == "__main__":

    train_barlow(sam_checkpoint=SAM_CHECKPOINT,
                 model_type=MODEL_TYPE,
                 coco_file_train=COCO_FILE_TRAIN,
                 coco_file_valid=COCO_FILE_TRAIN,
                 data_split_json=DATA_SPLIT_JSON,
                 epochs=10, learning_rate=1e-4,
                 batch_size=16, nb_copies=8,
                 pseudo_train_valid_ratio=2)
