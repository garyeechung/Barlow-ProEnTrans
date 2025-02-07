import logging

import numpy as np
from scipy.ndimage import binary_erosion
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_center_by_erosion(masks, to_xy: bool = True,
                          max_iter: int = 1,
                          return_eroded_masks=False):
    """_summary_

    Args:
        masks (np.Array): in shape [B, H, W] or [B, D, H, W]
        to_xy (bool, optional): originally in ij for 2D. If True, turn it to
                                xy for SAM point input. Defaults to True.
        max_iter (int, optional): _description_. Defaults to 50.

    Returns:
        center (np.Array): in shape [B, N=1, ndim=2, 3]
    """
    # device = masks.device
    if torch.is_tensor(masks):
        masks = masks.clone().cpu().numpy()
        # masks = masks.cpu().numpy()
    masks = masks.astype(np.uint8)

    assert masks.ndim in [3, 4], f"[B, H, W] or [B, D, H, W], got {masks.shape}"

    coords = []
    eroded_masks = []
    for mask in masks:
        if mask.sum() == 0:
            logger.warning("Empty mask found!")
            d, w, h = mask.shape
            coords.append(np.array([[d // 2, w // 2, h // 2]]))
            continue

        iter_count = 0
        mask_temp = np.copy(mask)
        while iter_count < max_iter:
            mask_temp = binary_erosion(mask, iterations=1).astype(mask.dtype)
            if (np.sum(mask_temp) == 0) or np.array_equal(mask, mask_temp):
                break
            else:
                mask = mask_temp
                iter_count += 1

        eroded_masks.append(mask)

        mask_coords = np.stack(np.where(mask), axis=-1)
        num_candidate = mask_coords.shape[0]
        idx = np.random.randint(0, num_candidate, 1)
        coords.append(mask_coords[idx])

    coords = np.stack(coords, axis=0)
    coords = torch.Tensor(coords)
    if to_xy:
        if masks.ndim == 3:
            coords = coords.flip(-1)
        elif masks.ndim == 4:
            # only swap the last two dimensions
            coords = coords[:, :, [0, 2, 1]]
    if return_eroded_masks:
        return coords, np.stack(eroded_masks, axis=0)
    else:
        return coords


def get_labels_from_coords(masks: torch.Tensor, coords, xy_coord=True):
    assert masks.ndim in [3, 4], f"masks shape [B, H, W] or [B, D, H, W], got {masks.shape}"
    assert coords.ndim == 3, f"coords should be in shape [B, N=1, 2 or 3], got {coords.shape}"
    # if not torch.is_tensor(coords):
    #     coords = torch.Tensor(coords)
    # if not torch.is_tensor(masks):
    #     masks = torch.Tensor(masks)

    if masks.ndim == 3:
        b, h, w = masks.size()
        flat_masks = masks.reshape(b, -1)
        if xy_coord:
            coords = coords.flip(-1)
        flat_indices = coords[:, :, 0] * w + coords[:, :, 1]
        flat_indices = flat_indices.long()
        labels = torch.gather(flat_masks, dim=-1, index=flat_indices)
    else:
        b, d, h, w = masks.size()
        flat_masks = masks.reshape(b, -1)
        if xy_coord:
            coords = coords[:, :, [0, 2, 1]]
        flat_indices = (coords[:, :, 0] * w * h +
                        coords[:, :, 1] * w +
                        coords[:, :, 2])
        flat_indices = flat_indices.long()
        labels = torch.gather(flat_masks, dim=-1, index=flat_indices)
    return labels


def get_coords_of_tensor_max(tensor: torch.Tensor, to_xy=True):
    """Get the coordinates of the maximum value in the tensor.

    Args:
        tensor (torch.Tensor): [B, C=1, H, W] or [B, C=1, D, H, W]
        to_xy (bool, optional): Whether to convert the coordinates to xy format
                                Only for 2D tensor. Defaults to True.

    Returns:
        coords (torch.Tensor): [B, C=1, 2 or 3]
    """
    assert tensor.ndim in [4, 5], "tensor shape [B, C=1, (D), H, W]"
    if tensor.ndim == 4:
        n, c, h, w = tensor.size()
        flat_tensor = tensor.view(n, c, -1)
        flat_indices = flat_tensor.argmax(dim=-1)
        coords = torch.stack((flat_indices // w,
                              flat_indices % w), dim=-1)
        if to_xy:
            coords = coords.flip(-1)
    elif tensor.ndim == 5:
        n, c, d, h, w = tensor.size()
        flat_tensor = tensor.view(n, c, -1)
        flat_indices = flat_tensor.argmax(dim=-1)
        coords = torch.stack((flat_indices // (w * h),
                              (flat_indices % (w * h)) // w,
                              flat_indices % w), dim=-1)
        if to_xy:
            coords = coords[:, :, [0, 2, 1]]
    return coords


def get_eig_from_probs(probs, original_size=(1024, 1024),
                       blur_sigma=10., blur_size=21, eps=1e-10):
    """This function calculates the full resolution eigenvalues. It takes
    the low resolution probs from multi-output SAM and calculates the EIG
    with the close-form solution. Then it upsamples the EIG to the original
    size and applies Gaussian blur.

    Args:
        probs (torch.Tensor): the low res. probability inferenced from
                               multi-output SAM, in [B, C=3, (D), H=256, W=256]
                               torch.float32
        original_size (tuple, optional): The original size of the image.
                                         Defaults to (1024, 1024).
        blur_sigma (float, optional): Sigma of the Gaussian blur.
                                      Defaults to 10.0
        blur_size (int, optional): The size of the Gaussian blur filter.
                                   positive odd number defaults to 21.

    Returns:
        eig_full (torch.Tensor): The full resolution EIG map, in
                                 [B, N=1, D_high, H_high, W_high]
                                 torch.float32
    """

    left = (probs ** probs) * ((1 - probs) ** (1 - probs))
    left = torch.clamp(left, eps, 1 - eps)
    left = torch.log(left)
    left = torch.mean(left, dim=1, keepdim=True)
    theta_bar = torch.mean(probs, dim=1, keepdim=True)
    right = (theta_bar ** theta_bar) * ((1 - theta_bar) ** (1 - theta_bar))
    right = torch.clamp(right, eps, 1 - eps)
    right = torch.log(right)
    eig_low_res = left - right
    if len(original_size) == 2:
        eig_high_res = F.interpolate(eig_low_res, size=original_size,
                                     mode='bilinear', align_corners=False)
    elif len(original_size) == 3:
        eig_high_res = F.interpolate(eig_low_res, size=original_size,
                                     mode='trilinear', align_corners=False)
    # eig_high_res = GaussianBlur(blur_size, blur_sigma)(eig_high_res)
    return eig_high_res


def get_eig_from_probs_nmc(probs, original_size=(1024, 1024),
                           blur_sigma=10., blur_size=21, eps=1e-10,
                           inner_n=15, outer_n=225):
    """This function calculates the full resolution eigenvalues. It takes
    the low resolution probs from multi-output SAM and calculates the EIG
    with nested Monte Carlo estimation. Then it upsamples the EIG to the
    original size and applies Gaussian blur.

    Args:
        probs (torch.Tensor): the low res. probability inferenced from
                               multi-output SAM, in [B, C=3, (D), H=256, W=256]
                               torch.float32
        original_size (tuple, optional): The original size of the image.
                                         Defaults to (1024, 1024).
        blur_sigma (float, optional): Sigma of the Gaussian blur.
                                      Defaults to 10.0
        blur_size (int, optional): The size of the Gaussian blur filter.
                                   positive odd number defaults to 21.
        inner_n (int, optional): The number of inner Monte Carlo samples.
                                 Defaults to 15.
        outer_n (int, optional): The number of outer Monte Carlo samples.
                                 Defaults to 225.

    Returns:
        eig_full (torch.Tensor): The full resolution EIG map, in
                                    [B, N=1, D_high, H_high, W_high]
                                    torch.float32
    """
    eigs = []
    num_heads = probs.size(1)
    # make num_heads a simple integer

    for _ in range(outer_n):
        outer_indices = np.random.randint(0, num_heads, 1).tolist()
        outer_probs = probs[:, outer_indices]  # [B, 1, (D), H, W]
        p = torch.rand_like(outer_probs)
        y = (p < outer_probs).float()
        eig_outer = (outer_probs ** y) * ((1 - outer_probs) ** (1 - y))
        eig_outer = torch.clamp(eig_outer, eps, 1 - eps)
        eig_outer = torch.log(eig_outer)

        inner_idx = np.random.randint(0, num_heads, inner_n).tolist()
        inner_probs = probs[:, inner_idx]  # [B, inner_n, (D), H, W]
        eig_inner = (inner_probs ** y) * ((1 - inner_probs) ** (1 - y))
        eig_inner = eig_inner.mean(dim=1, keepdim=True)
        eig_inner = torch.clamp(eig_inner, eps, 1 - eps)
        eig_inner = torch.log(eig_inner)

        eigs.append(eig_outer - eig_inner)
    eigs = torch.concatenate(eigs, dim=1)  # [B, outer_n, (D), H, W]
    eigs = eigs.mean(dim=1, keepdim=True)
    if len(original_size) == 2:
        eig_high_res = F.interpolate(eigs, size=original_size,
                                     mode='bilinear', align_corners=False)
        # eig_high_res = GaussianBlur(blur_size, blur_sigma)(eig_high_res)
    elif len(original_size) == 3:
        eig_high_res = F.interpolate(eigs, size=original_size,
                                     mode='trilinear', align_corners=False)
    return eig_high_res


def get_bbox_from_mask(masks: torch.Tensor, extend_ratio=0.1, xyzxyz=False):
    """Get the bounding box from the mask. The bounding box is the smallest
    rectangle that covers the mask.

    Args:
        mask (np.Array): in shape [B, H, W] or [B, D, H, W]
        extend_ratio (float, optional): The ratio to extend the bounding box.
                                        Defaults to 0.1.
    """
    device = masks.device
    # Get the coordinates of the non-zero elements in the mask
    nonzero_coords = masks.nonzero()
    # Get the bounding box for each channel
    bboxes = []
    for n in nonzero_coords[:, 0].unique():
        # Get the minimum and maximum coordinates for each channel
        box_min = nonzero_coords[nonzero_coords[:, 0] == n, 1:].min(dim=0)[0]
        box_max = 1 + nonzero_coords[nonzero_coords[:, 0] == n, 1:].max(dim=0)[0]
        # Calculate the range of the bounding box
        box_range = box_max - box_min
        # Extend the bounding box by the extend_ratio
        max_shape = torch.tensor(masks.shape[1:]).to(device)
        box_min = torch.clamp((box_min - box_range * extend_ratio).int(), min=0)
        box_max = torch.clamp((box_max + box_range * extend_ratio).int(), max=max_shape)
        if xyzxyz:
            bboxes.append(torch.cat([box_min, box_max]).unsqueeze(0))
        else:
            bboxes.append(torch.stack([box_min, box_max]))
    bboxes = torch.stack(bboxes)
    return bboxes


def get_pairwise_dices(samples, nb_copies, smooth_factor=1e-3):
    assert (len(samples) % nb_copies) == 0, f"{len(samples)} % {nb_copies} = {len(samples) % nb_copies}"
    nb_samples = len(samples)
    batch_size = nb_samples // nb_copies

    dice_lookup = torch.zeros((batch_size, batch_size))
    for i in range(0, nb_samples, nb_copies):
        mask_i = samples[i]["masks"]
        for j in range(i, len(samples), nb_copies):
            mask_j = samples[j]["masks"]
            numerator = 2 * (mask_i * mask_j).sum() + smooth_factor
            denominator = mask_i.sum() + mask_j.sum() + smooth_factor
            dice = numerator / denominator
            dice_lookup[i // nb_copies, j // nb_copies] = dice

    dices = []
    for i in range(nb_samples):
        for j in range(i, nb_samples):
            dice = dice_lookup[i // nb_copies, j // nb_copies]
            dices.append(dice)
    dices = torch.stack(dices)

    return dices
