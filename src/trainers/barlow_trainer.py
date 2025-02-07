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
COCO_FILE = "data/fiftyone/coco-2017/validation/labels.json"


def train_barlow(sam_checkpoint: str = SAM_CHECKPOINT,
                 model_type: str = MODEL_TYPE,
                 coco_file: str = COCO_FILE,
                 random_seed: int = 42, epochs: int = 1000,
                 learning_rate: float = 1e-4, batch_size: int = 8,
                 nb_copies: int = 8,
                 parallel: bool = False) -> None:

    time_now = strftime('%Y%m%d%H%M%S', localtime())
    torch.manual_seed(random_seed)
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    wandb.init(project="BarlowTwins", name=f"blt-{time_now}",
               entity="garyeechung-vanderbilt-university",
               job_type="BarlowTwins", group="COCO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    sam = sam_model_registry[model_type](sam_checkpoint)
    barlow_twins = BarlowTwinsCosineSimilarity(sam=sam, nb_copies=nb_copies)
    if parallel:
        barlow_twins = DataParallel(barlow_twins)
    barlow_twins = barlow_twins.to(device)

    coco_dataset_train = CocoMaskAndPoints(coco_file,
                                           # nb_positives=10,
                                           # nb_negatives=10,
                                           min_area_ratio=0.1,
                                           max_area_ratio=0.7,
                                           nb_copies=nb_copies,
                                           validation=False)
    coco_loader_train = DataLoader(coco_dataset_train, batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=flatten_collate_fn)
    coco_dataset_valid = CocoMaskAndPoints(coco_file,
                                           # nb_positives=10,
                                           # nb_negatives=10,
                                           min_area_ratio=0.1,
                                           max_area_ratio=0.7,
                                           nb_copies=nb_copies,
                                           validation=True)
    coco_loader_valid = DataLoader(coco_dataset_valid, batch_size=nb_copies,
                                   shuffle=False,
                                   collate_fn=flatten_collate_fn)

    mse = MSELoss()
    optimizer = Adam(barlow_twins.parameters(), lr=learning_rate)

    with torch.no_grad():
        loss = 0.
        for i, samples in enumerate(coco_loader_valid, 1):
            for sample in samples:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.to(device)
            dices = get_pairwise_dices(samples, nb_copies)
            dices = dices.to(device)
            cos_sims = barlow_twins(samples)
            loss += mse(dices, cos_sims).item()
        loss /= i
        wandb.log({"val_loss": loss})

    best_val_loss = loss

    for epoch in range(1, epochs + 1):
        barlow_twins.train()
        for samples in tqdm(coco_loader_train):
            for sample in samples:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.to(device)
            optimizer.zero_grad()
            dices = get_pairwise_dices(samples, nb_copies)
            dices = dices.to(device)
            cos_sims = barlow_twins(samples)
            loss = mse(dices, cos_sims)
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})

        with torch.no_grad():
            loss = 0.
            for i, samples in enumerate(coco_loader_valid, 1):
                for sample in samples:
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            sample[key] = value.to(device)
                dices = get_pairwise_dices(samples, nb_copies)
                dices = dices.to(device)
                cos_sims = barlow_twins(samples)
                loss += mse(dices, cos_sims).item()
            loss /= i
            wandb.log({"val_loss": loss})

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
    train_barlow()
