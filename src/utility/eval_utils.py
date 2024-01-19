import pdb

import numpy as np
import torch
from torch import nn
import os


from src.utility.debug_utils import time_since_previous_log
from src.utility.viz_utils import calc_loss_metrics, colorize, denormalize, log_images

from torchvision import transforms


# shamelessly inspired by Zoedepth
def filter_valid(
    gt,
    pred,
    uncertainty,
    interpolate=True,
    garg_crop=True,
    eigen_crop=False,
    dataset="kitti",
    min_depth_eval=1e-3,
    max_depth_eval=80,
    **kwargs,
):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics."""
    if "config" in kwargs:
        config = kwargs["config"]
        eigen_crop = config.eval_eigen_crop
        garg_crop = config.eval_garg_crop

        min_depth_eval = config.dataset_params.min_depth
        max_depth_eval = config.dataset_params.max_depth
    # garg_crop = None
    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(pred, gt.shape[-2:], mode="bilinear", align_corners=True)

    pred = pred.cpu()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.cpu()
    valid_mask = np.logical_and(gt_depth >= min_depth_eval, gt_depth <= max_depth_eval)
    if garg_crop or eigen_crop:
        _, _, gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[
                :,
                :,
                int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
            ] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == "kitti":
                eval_mask[
                    :,
                    :,
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[:, :, 45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return gt_depth[valid_mask], pred[valid_mask], uncertainty[valid_mask]


def save_eval_images(images, depths, preds, uncertainty, date_and_time_and_model, cfg):
    for j in range(min(images.shape[0], 6)):
        image = images[j, :, :, :]
        depth = depths[j, :, :, :]

        if cfg.models.model_type not in [
            "Posthoc_Laplace",
            "Online_Laplace",
        ]:  # then collapse on different predictions.
            pred = torch.mean(
                preds[:, j, :, :, :], dim=0
            )  # pred is average prediction, preds ModelxBatchxColorxHxW

            std_dev = (
                torch.mean(preds[:, j, :, :, :] ** 2, dim=0) - pred**2
            ).sqrt()  # var is variance over model. since color dimension is 1, it is by pixel.

        else:  # laplace has already collapsed across model dim.
            pred = preds[j, :, :, :]
            std_dev = uncertainty[j, :, :, :]

        os.makedirs(
            os.path.join(cfg.save_images_path, "images/", date_and_time_and_model),
            exist_ok=True,
        )
        d = torch.tensor(np.transpose(colorize(torch.squeeze(depth, dim=0), 0, 80), (2, 0, 1)))
        p = torch.tensor(np.transpose(colorize(torch.squeeze(pred, dim=0), 0, 80), (2, 0, 1)))
        sd = torch.tensor(
            np.transpose(
                colorize(
                    torch.squeeze(std_dev, dim=0),
                    vmin=0,
                    vmax=torch.quantile(std_dev, 0.95),
                ),
                (2, 0, 1),
            )  # None,None = v.min(), v.max() for color range
        )

        diff = torch.abs(depth - pred)

        diff_colored = torch.tensor(
            np.transpose(
                colorize(
                    torch.squeeze(diff, dim=0),
                    vmin=0,
                    vmax=80,
                ),
                (2, 0, 1),
            )  # None,None = v.min(), v.max() for color range
        )

        # Get scale-normalized uncertainty:
        scaled_uncertainty_uncolored = std_dev / pred
        scaled_uncertainty = torch.tensor(
            np.transpose(
                colorize(
                    torch.squeeze(scaled_uncertainty_uncolored, dim=0),
                    vmin=0,
                    vmax=torch.quantile(scaled_uncertainty_uncolored, 0.95),
                ),
                (2, 0, 1),
            )  # None,None = v.min(), v.max() for color range
        )

        if cfg.models.model_type == "ZoeNK":
            im = transforms.ToPILImage()(image)  # dont denormalize; it is the original image.
        else:
            im = transforms.ToPILImage()(denormalize(image).cpu())
        print(type(d))

        # --------------------------------------  LOG AND SAVE  ---------------------------------------------------------------------------------
        im.save(
            os.path.join(
                cfg.save_images_path,
                "images/",
                date_and_time_and_model,
                f"{j}_{cfg.models.model_type}_img.png",
            )
        )

        print(d.shape, p.shape, image.shape, pred.shape, depth.shape)

        for key, value in {
            "depth": d,
            "pred": p,
            "diff": diff_colored,
            "sd": sd,
            "ScaledUncertainty": scaled_uncertainty,
        }.items():  # save images/depths/etc to file
            transforms.ToPILImage()(value).save(
                os.path.join(
                    cfg.save_images_path,
                    "images/",
                    date_and_time_and_model,
                    f"{j}_{cfg.models.model_type}_{key}.png",
                )
            )

        for key, value in {
            "img": image,
            "depth": depth,
            "preds": pred,
            "std_dev": sd,
            "diff": diff_colored,
            "ScaledUncertainty": scaled_uncertainty,
        }.items():
            if key == "img":
                np.save(
                    os.path.join(
                        cfg.save_images_path,
                        "images/",
                        date_and_time_and_model,
                        f"np_{key}_{cfg.models.model_type}_{j}.npy",
                    ),
                    torch.squeeze(denormalize(value), dim=0).numpy(force=True),
                )
            else:
                np.save(
                    os.path.join(
                        cfg.save_images_path,
                        "images/",
                        date_and_time_and_model,
                        f"np_{key}_{cfg.models.model_type}_{j}.npy",
                    ),
                    torch.squeeze(value, dim=0).numpy(force=True),
                )

        log_images(
            img=image.detach(),
            depth=depth.detach(),
            pred=pred.detach(),
            vmin=cfg.dataset_params.min_depth,
            vmax=cfg.dataset_params.max_depth,
            step=(j + 1) * 5000000,
        )
