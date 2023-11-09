import torch
from tqdm import tqdm
import pdb
import time
from src.utility.debug_utils import time_since_previous_log
from src.models.predict_model import infer
from src.utility.other_utils import RunningAverageDict
from src.utility.eval_utils import compute_metrics
import numpy as np
import os
import timeit
import wandb
from PIL import Image
import torchvision.transforms as transforms
from src.utility.viz_utils import colorize, denormalize


@torch.no_grad()
def eval_model(model, model_name, test_loader, config, round_vals=True, round_precision=3):
    model.eval().to("cuda")
    metrics = RunningAverageDict()
    i = 0
    for sample in tqdm(test_loader, total=len(test_loader)):
        if "has_valid_depth" in sample:
            if not sample["has_valid_depth"]:
                print("validDEPTH?")
                continue
        images, _, depths = sample
        images = images.to(device="cuda")
        depths = depths.to(device="cuda")
        preds = infer(model, images)
        # Save image, depth, pred for visualization
        if config.save_images and i == 0:
            for j in range(min(images.shape[0], 6)):
                image = images[j, :, :, :]
                depth = depths[j, :, :, :]
                pred = preds[j, :, :, :]

                os.makedirs(config.save_images_path, exist_ok=True)
                d = colorize(depth, 0, 80)
                p = colorize(pred, 0, 80)
                im = transforms.ToPILImage()(denormalize(image).cpu())
                im.save(os.path.join(config.save_images_path, f"{j}_{model_name}_img.png"))
                print(d.shape, p.shape, image.shape, pred.shape, depth.shape)
                transforms.ToPILImage()(np.squeeze(d, axis=0)).save(
                    os.path.join(config.save_images_path, f"{j}_{model_name}_depth.png")
                )
                transforms.ToPILImage()(np.squeeze(p, axis=0)).save(
                    os.path.join(config.save_images_path, f"{j}_{model_name}_pred.png")
                )

                np.save(
                    os.path.join(config.save_images_path, f"np_img_{model_name}_{j}.npy"),
                    torch.squeeze(denormalize(image), dim=0).numpy(force=True),
                )
                np.save(
                    os.path.join(config.save_images_path, f"np_depth_{model_name}_{j}.npy"),
                    torch.squeeze(depth, dim=0).numpy(force=True),
                )
                np.save(
                    os.path.join(config.save_images_path, f"np_preds_{model_name}_{j}.npy"),
                    torch.squeeze(pred, dim=0).numpy(force=True),
                )
        if config.in_debug and i > 0:
            continue
        i += 1

        time_prev = time.time()
        # print(depth.shape, pred.shape)
        print("hi")
        time.sleep(0.02)

        losses = compute_metrics(depths, preds, config=config)
        time_prev = time_since_previous_log(time_prev, "compute_metrics")
        print(time.time() - time_prev)
        time.sleep(0.02)
        metrics.update(losses)
        print("hi2")
        print(time.time() - time_prev)
        time_prev = time_since_previous_log(time_prev, "updatingmetrics")
    if round_vals:

        def r(m):
            return torch.round(m, decimals=round_precision)

    else:

        def r(m):
            return m

    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics
