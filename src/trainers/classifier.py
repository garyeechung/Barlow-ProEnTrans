# import json
import logging
from time import localtime, strftime

from segment_anything import sam_model_registry
import torch
from torch.nn import BCELoss, DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..datasets import CocoMaskAndPoints, flatten_collate_fn
from ..modeling.proentrans import SidesMultiLabelClassifier


SAM_CHECKPOINT = "model_checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
COCO_FILE = "data/fiftyone/coco-2017/validation/labels.json"


def train_classifier(sam_checkpoint: str = SAM_CHECKPOINT,
                     model_type: str = MODEL_TYPE,
                     coco_file: str = COCO_FILE,
                     random_seed: int = 42, epochs: int = 1000,
                     learning_rate: float = 1e-4, batch_size: int = 64,
                     parallel: bool = False) -> None:
    time_now = strftime('%Y-%m-%d-%H:%M:%S', localtime())
    torch.manual_seed(random_seed)
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    wandb.init(project="BarlowTwins", name=time_now,
               entity="garyeechung-vanderbilt-university",
               job_type="Phase_1-classifier", group="COCO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    sam = sam_model_registry[model_type](sam_checkpoint)
    sideclassifier = SidesMultiLabelClassifier(sam=sam)
    if parallel:
        sideclassifier = DataParallel(sideclassifier)
    sideclassifier = sideclassifier.to(device)

    coco_dataset = CocoMaskAndPoints(coco_file,
                                     # nb_positives=10,
                                     # nb_negatives=10,
                                     min_area_ratio=0.1,
                                     max_area_ratio=0.7,
                                     nb_samples=8)
    coco_loader_train = DataLoader(coco_dataset, batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=flatten_collate_fn)
    coco_loader_valid = DataLoader(coco_dataset, batch_size=batch_size,
                                   shuffle=False,
                                   collate_fn=flatten_collate_fn)

    bce = BCELoss()
    optimizer = Adam(sideclassifier.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        sideclassifier.train()
        for samples in tqdm(coco_loader_train):
            optimizer.zero_grad()
            y_gt = torch.stack([sample["side_class"] for sample in samples]).to(device)
            for sample in samples:
                sample["point_coords"] = sample["point_coords"].to(device)
                sample["point_labels"] = sample["point_labels"].to(device)
            y_pred = sideclassifier(samples)
            loss = bce(y_pred, y_gt)
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})

        # sideclassifier.eval()
        with torch.no_grad():
            acc = 0.
            f1 = 0.
            loss = 0.
            for i, samples in enumerate(coco_loader_valid, 1):
                y_gt = torch.stack([sample["side_class"] for sample in samples]).to(device)
                for sample in samples:
                    sample["point_coords"] = sample["point_coords"].to(device)
                    sample["point_labels"] = sample["point_labels"].to(device)
                y_pred = sideclassifier(samples)
                acc += (y_pred.round() == y_gt).float().mean().item()
                f1 += (2 * (y_pred.round() * y_gt).sum() / (y_pred.round() + y_gt).sum()).item()
                loss += bce(y_pred, y_gt).item()
            acc /= i
            f1 /= i
            loss /= i

            wandb.log({"valid_loss": loss, "valid_acc": acc, "valid_f1": f1})
        if loss < best_val_loss:
            best_val_loss = loss
            if parallel:
                model_dict = sideclassifier.module.state_dict()
            else:
                model_dict = sideclassifier.state_dict()
            save_dict = {"model": model_dict,
                         "optimizer": optimizer.state_dict(),
                         "best_val_loss": best_val_loss,
                         "epoch": epoch}
            torch.save(save_dict, f"model_checkpoints/Barlow/classifier_{time_now}.pth")


if __name__ == "__main__":
    train_classifier()
