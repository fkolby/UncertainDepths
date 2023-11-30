import os
import pdb
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch_laplace import MSEHessianCalculator
from pytorch_laplace.laplace.diag import DiagLaplace
from pytorch_laplace.optimization.prior_precision import optimize_prior_precision
from tqdm import tqdm
from datetime import datetime

import wandb
from src.utility.debug_utils import time_since_previous_log
from src.utility.eval_utils import calc_loss_metrics, filter_valid
from src.utility.other_utils import RunningAverageDict
from src.utility.viz_utils import colorize, denormalize, log_images


@torch.no_grad()
def eval_model(
    model,
    test_loader,
    cfg,
    dataloader_for_hessian=None,
    round_vals=True,
    round_precision=3,
):
    metrics = RunningAverageDict()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    date_and_time_and_model = (
        datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + "_" + cfg.models.model_type
    )
    i = 0
    if cfg.models.model_type == "stochastic_unet":
        hessian_calculator = MSEHessianCalculator(
            hessian_shape="diag", approximation_accuracy="approx"
        )
        hessian = torch.zeros_like(torch.nn.utils.parameters_to_vector(model.parameters()))
        laplace = DiagLaplace()
        print("Calcing hessian")
        a = 0
        for image, _, _ in dataloader_for_hessian:  # should be train.
            a += 1
            # compute hessian approximation
            print(a, "of ", len(dataloader_for_hessian), flush=True)
            hessian += hessian_calculator.compute_hessian(
                x=image.to(device),
                model=model.stochastic_net,
            )
            if cfg.in_debug and a > 25:
                break

        print("Done calcing hessian")
        mean_parameter = torch.nn.utils.parameters_to_vector(model.parameters())
        with torch.enable_grad():
            prior_precision = optimize_prior_precision(
                mu_q=mean_parameter,
                hessian=hessian,
                prior_prec=torch.tensor([1.0], device=device),
                n_steps=500,
            )
        print(f"prior precision: {prior_precision}")

    # ======================================================   PREDICTING  ===============================================

    depths_all_samples = torch.Tensor().to(device)
    preds_all_samples = torch.Tensor().to(device)
    uncertainty_all_samples = torch.Tensor().to(device)

    for i, sample in enumerate(tqdm(test_loader, total=len(test_loader))):
        if i > 50 and cfg.in_debug:  # skip if too long.
            continue

        if "has_valid_depth" in sample:
            if not sample["has_valid_depth"]:
                print("validDEPTH?")
                continue
        images, depths, _ = sample
        images = images.to(device=device)
        depths = depths.to(device=device)  # BxCxHxW

        match cfg.models.model_type:
            case "stochastic_unet":
                preds, uncertainty = laplace.laplace(
                    x=images,
                    model=model,
                    hessian=hessian,
                    prior_prec=prior_precision,
                    n_samples=cfg.models.n_models,
                )  ###NOTE TO SELF: MUST BE NON-flipped for other models as well
                print("shapes:", preds.shape, uncertainty.shape)
                print(uncertainty)
                print(uncertainty[0, :, :, :])
                # dimensions: BatchxColorxHxW (no model-dim.)
            case "Ensemble":
                j = 0
                preds = torch.zeros(
                    size=(
                        cfg.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        cfg.dataset_params.input_height,
                        cfg.dataset_params.input_width,
                    ),
                    device=device,
                )  # dimensions: model, batch, color, height, width.
                for model_idx in range(cfg.models.n_models):
                    model.load_state_dict(torch.load(f"{cfg.models.model_type}_{model_idx}.pt"))
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        model(images), dim=0
                    )  # unsqueeze to have a model-dimension (first)
                    j += 1
            case "Dropout":  # Only one model, but is different every call.
                preds = torch.zeros(
                    size=(
                        cfg.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        cfg.dataset_params.input_height,
                        cfg.dataset_params.input_width,
                    ),
                    device=device,
                )  # dimensions: model, batch, color, height, width.
                for j in range(cfg.models.n_models):
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        model(images), dim=0
                    )  # unsqueeze to have a model-dimension (first)

            case "ZoeNK":
                preds = torch.zeros(
                    size=(
                        cfg.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        cfg.dataset_params.input_height,
                        cfg.dataset_params.input_width,
                    ),
                    device=device,
                )  # dimensions: model, batch, color, height, width.
                print(preds.shape)
                for j in range(cfg.models.n_models):
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        torchvision.transforms.Resize(
                            (cfg.dataset_params.input_height, cfg.dataset_params.input_width)
                        )(model(images)),
                        dim=0,
                    )  # unsqueeze to have a model-dimension (first)

        # -----------------------------------  Save image, depth, pred for visualization -------------------------------------------------

        if cfg.save_images and i == 0:
            for j in range(min(images.shape[0], 6)):
                image = images[j, :, :, :]
                depth = depths[j, :, :, :]

                if (
                    cfg.models.model_type != "stochastic_unet"
                ):  # then collapse on different predictions.
                    pred = torch.mean(
                        preds[:, j, :, :, :], dim=0
                    )  # pred is average prediction, preds ModelxBatchxColorxHxW

                    var = (
                        torch.mean(preds[:, j, :, :, :] ** 2, dim=0) - pred**2
                    )  # var is variance over model. since color dimension is 1, it is by pixel.

                else:  # laplace has already collapsed across model dim.
                    pred = preds[j, :, :, :]
                    var = uncertainty[j, :, :, :]

                os.makedirs(
                    os.path.join(cfg.save_images_path, "images/", date_and_time_and_model),
                    exist_ok=True,
                )
                d = torch.tensor(
                    np.transpose(colorize(torch.squeeze(depth, dim=0), 0, 80), (2, 0, 1))
                )
                p = torch.tensor(
                    np.transpose(colorize(torch.squeeze(pred, dim=0), 0, 80), (2, 0, 1))
                )
                v = torch.tensor(
                    np.transpose(
                        colorize(torch.squeeze(var, dim=0), None, None), (2, 0, 1)
                    )  # None,None = v.min(), v.max() for color range
                )
                if cfg.models.model_type == "ZoeNK":
                    im = transforms.ToPILImage()(
                        image
                    ).cpu()  # dont denormalize; it is the original image.
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

                transforms.ToPILImage()(d).save(
                    os.path.join(
                        cfg.save_images_path,
                        "images/",
                        date_and_time_and_model,
                        f"{j}_{cfg.models.model_type}_depth.png",
                    )
                )
                transforms.ToPILImage()(p).save(
                    os.path.join(
                        cfg.save_images_path,
                        "images/",
                        date_and_time_and_model,
                        f"{j}_{cfg.models.model_type}_pred.png",
                    )
                )
                transforms.ToPILImage()(v).save(
                    os.path.join(
                        cfg.save_images_path,
                        "images/",
                        date_and_time_and_model,
                        f"{j}_{cfg.models.model_type}_var.png",
                    )
                )

                np.save(
                    os.path.join(
                        cfg.save_images_path,
                        "images/",
                        date_and_time_and_model,
                        f"np_img_{cfg.models.model_type}_{j}.npy",
                    ),
                    torch.squeeze(denormalize(image), dim=0).numpy(force=True),
                )
                np.save(
                    os.path.join(
                        cfg.save_images_path,
                        "images/",
                        date_and_time_and_model,
                        f"np_depth_{cfg.models.model_type}_{j}.npy",
                    ),
                    torch.squeeze(depth, dim=0).numpy(force=True),
                )
                np.save(
                    os.path.join(
                        cfg.save_images_path,
                        "images/",
                        date_and_time_and_model,
                        f"np_preds_{cfg.models.model_type}_{j}.npy",
                    ),
                    torch.squeeze(pred, dim=0).numpy(force=True),
                )

                log_images(
                    img=image.detach(),
                    depth=depth.detach(),
                    pred=pred.detach(),
                    vmin=cfg.dataset_params.min_depth,
                    vmax=cfg.dataset_params.max_depth,
                    step=(j + 1) * 5000000,
                )

        if (
            cfg.models.model_type != "stochastic_unet"
        ):  # then collapse on different predictions by each model.
            pred = torch.mean(preds, dim=0)

            uncertainty = torch.var(preds, dim=0)
        else:
            pred = preds  # only named preds for consistency with other models - naming here might be worth rewriting.
            # uncertainty already=uncertainty

        depths, pred, uncertainty = filter_valid(depths, pred, uncertainty=uncertainty, config=cfg)
        depths.to(device)
        pred.to(device)
        uncertainty.to(device)
        print(depths.device, depths_all_samples.device)
        print(depths_all_samples.shape, uncertainty_all_samples.shape, preds_all_samples.shape)
        print(depths.shape, uncertainty.shape, pred.shape)

        depths_all_samples = torch.cat(
            [depths_all_samples, torch.flatten(depths).to(device)]
        )  # flatten to ensure sorting by uncertainty.
        preds_all_samples = torch.cat([preds_all_samples, torch.flatten(pred).to(device)])
        uncertainty_all_samples = torch.cat(
            [uncertainty_all_samples, torch.flatten(uncertainty).to(device)]
        )
        if cfg.in_debug and i > 0:
            continue
        i += 1

    _, sort_by_uncertainty_ascending_indices = torch.sort(uncertainty_all_samples)
    losses = calc_loss_metrics(
        depths_all_samples[sort_by_uncertainty_ascending_indices],
        preds_all_samples[sort_by_uncertainty_ascending_indices],
    )

    def uncertainty_results_df(fineness, sorted_preds, sorted_targets):
        df = pd.DataFrame(
            columns=["Share"]
            + [el for el in calc_loss_metrics(sorted_preds, sorted_targets).keys()]
        )
        for i in range(fineness):  # increment in thousandths
            print(f"{i} of {fineness}", flush=True)
            df.loc[i] = [i / fineness] + [
                el.numpy(force=True)
                for el in calc_loss_metrics(
                    sorted_preds[int(i / fineness * len(sorted_preds)) :],
                    sorted_targets[
                        int(i / fineness * len(sorted_preds)) :
                    ],  # sorted_preds/targets should be same length.
                ).values()
            ]
        return df

    uncertainty_df = uncertainty_results_df(
        1000,
        depths_all_samples[sort_by_uncertainty_ascending_indices],
        preds_all_samples[sort_by_uncertainty_ascending_indices],
    )

    def save_uncertainty_plots(df, file_prefix):
        for i, c in enumerate(df.columns):
            if c != "Share":
                # axs[i].plot(df[[c]])
                plt.plot(df[["Share"]], df[[c]])
                plt.xlabel("Uncertainty: Share of predictions")
                plt.ylabel(c)
                plt.title(f"{c} as uncertainty increases")
                plt.savefig(fname=file_prefix + c)
                plt.clf()

    os.makedirs(
        os.path.join(cfg.save_images_path, "uncertainty_df/", date_and_time_and_model),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(cfg.save_images_path, "plots/", date_and_time_and_model), exist_ok=True
    )
    uncertainty_df.to_csv(
        os.path.join(
            cfg.save_images_path,
            "uncertainty_df/",
            date_and_time_and_model,
            f"uncertainty_df_{cfg.models.model_type}",
        ),
        index=False,
    )
    save_uncertainty_plots(
        uncertainty_df,
        os.path.join(
            cfg.save_images_path,
            "plots/",
            date_and_time_and_model,
            f"plot_{cfg.models.model_type}_",
        ),
    )
    print("Updating metrics")
    metrics.update(losses)
    if round_vals:

        def r(m):
            return torch.round(m, decimals=round_precision)

    else:

        def r(m):
            return m

    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics
