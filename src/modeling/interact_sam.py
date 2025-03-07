import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_center_by_erosion
from ..utils import get_labels_from_coords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractSAM(nn.Module):

    def __init__(self, sam, proentrans=None,
                 include_class_embedding=False,
                 residual_connection=False,
                 preserve_embedding=False):
        super(InteractSAM, self).__init__()
        self.sam = sam
        self.proentrans = proentrans
        self.include_class_embedding = include_class_embedding
        self.residual_connection = residual_connection
        self.preserve_embedding = preserve_embedding

    def forward(self, image, masks, point_coords, point_labels,
                steps=0, return_prompts=False, return_intermediate=False,
                multimask_output=False):
        """This function takes one image and its corresponding masks and
        point prompts at a time.

        Args:
            In the following, B refers to the number of task in an image.
            N refers to the number of point prompts in an image.
            image (torch.Tensor): ImageNet-standardized, in [3, 1024, 1024]
                                  Values are torch.float32
            masks (torch.Tensor): Ground truth masks, in [B, 1024, 1024]
                                  Values are either 0.0 or 1.0, torch.float32
            point_coords (torch.Tensor): Point prompts' coords, in [B, N, 2]
                                         in xy format, torch.float32
            point_labels (torch.Tensor)): Point prompts' labels, in [B, N]
                                          Values are either 0. or 1.
                                          torch.float32
            steps (int, optional): Steps of adding new prompt. Defaults to 1.
        """
        img_size = image.size()[-2:]
        outputs = []

        with torch.no_grad():
            image_embedding = self.sam.image_encoder(image.unsqueeze(0))
            image_pe = self.sam.prompt_encoder.get_dense_pe()

            for _ in range(steps):
                logits = self.predict_logits_256(image_embedding, image_pe,
                                                 point_coords, point_labels,
                                                 multimask_output)
                logits = F.interpolate(logits, size=img_size, mode="bilinear",
                                       align_corners=False)
                probs = torch.sigmoid(logits)
                outputs.append(probs)

                new_coords, new_labels = self.get_new_points(probs, masks, image_embedding,
                                                             point_coords, point_labels)
                point_coords = torch.cat([point_coords, new_coords], dim=1)
                point_labels = torch.cat([point_labels, new_labels], dim=1)

        output = self.predict_logits_256(image_embedding, image_pe,
                                         point_coords, point_labels,
                                         multimask_output)
        output = F.interpolate(output, size=img_size, mode="bilinear",
                               align_corners=False)
        output = torch.sigmoid(output)
        outputs.append(output)
        outputs = torch.stack(outputs, dim=0)

        if not return_intermediate:
            outputs = outputs[-1]
        if return_prompts:
            return outputs, point_coords, point_labels
        else:
            return outputs

    def predict_logits_256(self, image_embeddings, image_pe,
                           point_coords, point_labels,
                           multimask_output=False):
        points = (point_coords, point_labels)
        sparse_prompt_embeddings, dense_prompt_embeddings = self.sam.prompt_encoder(
            points=points, boxes=None, masks=None
        )
        sparse_prompt_embeddings = F.pad(sparse_prompt_embeddings,
                                         (0, 0, 0, 1), value=0)
        if self.proentrans is not None:
            if self.preserve_embedding:
                src_key_preserve_mask = torch.ones_like(sparse_prompt_embeddings, dtype=torch.float32)
                src_key_preserve_mask[:, -1] = 0
            else:
                src_key_preserve_mask = None
            sparse_prompt_embeddings = self.proentrans.encoder(sparse_prompt_embeddings,
                                                               residual_connection=self.residual_connection,
                                                               src_key_preserve_mask=src_key_preserve_mask)
        if not self.include_class_embedding:
            sparse_prompt_embeddings = sparse_prompt_embeddings[:, :-1, :]
        logits, _ = self.sam.mask_decoder(image_embeddings=image_embeddings,
                                          image_pe=image_pe,
                                          sparse_prompt_embeddings=sparse_prompt_embeddings,
                                          dense_prompt_embeddings=dense_prompt_embeddings,
                                          multimask_output=multimask_output)
        return logits

    def get_new_points(self, probs, masks, image_embedding,
                       point_coords, point_labels):
        raise NotImplementedError


class HeuristicSAM(InteractSAM):

    def get_new_points(self, probs, masks, image_embedding,
                       point_coords, point_labels):

        probs = probs.mean(dim=1, keepdim=False)
        preds = torch.greater(probs, 0.5).float()
        wrong_preds = torch.logical_xor(preds, masks)
        new_coords = get_center_by_erosion(wrong_preds, max_iter=20)
        new_coords = new_coords.to(masks.device)
        new_labels = get_labels_from_coords(masks, new_coords)
        return new_coords, new_labels
