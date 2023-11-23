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
from tqdm import tqdm

import wandb
from src.models.predict_model import infer
from src.utility.debug_utils import time_since_previous_log
from src.utility.eval_utils import calc_loss_metrics, filter_valid
from src.utility.other_utils import RunningAverageDict
from src.utility.viz_utils import colorize, denormalize, log_images


@torch.no_grad()
def eval_model(
    model,
    model_name,
    test_loader,
    config,
    dataloader_for_hessian=None,
    round_vals=True,
    round_precision=3,
    test_data_img_nums=-1,
):
    metrics = RunningAverageDict()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    i = 0
    if config.models.model_type == "stochastic_unet":
        hessian_calculator = MSEHessianCalculator(
            hessian_shape="diag", approximation_accuracy="approx"
        )
        hessian = torch.zeros_like(torch.nn.utils.parameters_to_vector(model.parameters()))
        laplace = DiagLaplace()
        print("Calcing hessian")
        a = 0
        for image, _, _ in dataloader_for_hessian:  # should be val, when on test-set.
            a += 1
            # compute hessian approximation
            print(a, "of ", len(dataloader_for_hessian))
            hessian += hessian_calculator.compute_hessian(
                x=image.to(device),
                model=model.stochastic_net,
            )
            if config.in_debug and a > 25:
                break

        print("Done calcing hessian")
        mean_parameter = torch.nn.utils.parameters_to_vector(
            model.parameters()
        ).clone()  # clone as we are replacing parameters in a second.
        standard_deviation = laplace.posterior_scale(hessian, prior_prec=1)

    # ---------------------------------------------------------------------      Predicting  ------------------------------------

    depths_all_samples = torch.Tensor().to(device)
    preds_all_samples = torch.Tensor().to(device)
    uncertainty_all_samples = torch.Tensor().to(device)

    for i, sample in enumerate(tqdm(test_loader, total=len(test_loader))):
        if "has_valid_depth" in sample:
            if not sample["has_valid_depth"]:
                print("validDEPTH?")
                continue
        # images, _, depths = sample OLD _ why did i have old untransformed?
        images, depths, _ = sample
        images = images.to(device=device)
        depths = depths.to(device=device)  # BxCxHxW

        match config.models.model_type:
            case "stochastic_unet":
                preds = torch.zeros(
                    size=(
                        config.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        config.dataset_params.input_height,
                        config.dataset_params.input_width,
                    ),
                    device=device,
                )  # dimensions: model, batch, color, height, width."""
                for j in range(config.models.n_models):
                    # get samples of Neural Networks according to the Gaussian N(mu="mean_parameter", Sigma=hessian^{-1})
                    samples = laplace.sample_from_normal(
                        mean_parameter, standard_deviation, n_samples=1
                    )  # slow - speed up with more samples per iteration.
                    torch.nn.utils.vector_to_parameters(samples[0, :], model.parameters())
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        infer(model, images), dim=0
                    )  # unsqueeze to have a model-dimension (first)
            case "Ensemble":  # Model is then actually a dict of models:
                j = 0
                preds = torch.zeros(
                    size=(
                        config.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        config.dataset_params.input_height,
                        config.dataset_params.input_width,
                    ),
                    device=device,
                )  # dimensions: model, batch, color, height, width.
                for name, mod in model.items():
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        infer(mod, images), dim=0
                    )  # unsqueeze to have a model-dimension (first)
                    j += 1
            case "Dropout":  # Only one model, but is different every call.
                preds = torch.zeros(
                    size=(
                        config.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        config.dataset_params.input_height,
                        config.dataset_params.input_width,
                    ),
                    device=device,
                )  # dimensions: model, batch, color, height, width.
                for j in range(config.models.n_models):
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        infer(model, images), dim=0
                    )  # unsqueeze to have a model-dimension (first)

            case "BaseUNet":
                preds = torch.zeros(
                    size=(
                        config.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        config.dataset_params.input_height,
                        config.dataset_params.input_width,
                    ),
                    device=device,
                )  # dimensions: model, batch, color, height, width.
                for j in range(config.models.n_models):
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        infer(model, images), dim=0
                    )  # unsqueeze to have a model-dimension (first)

            case "ZoeNK":
                preds = torch.zeros(
                    size=(
                        config.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        config.dataset_params.input_height,
                        config.dataset_params.input_width,
                    ),
                    device=device,
                )  # dimensions: model, batch, color, height, width.
                print(preds.shape)
                for j in range(config.models.n_models):
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        torchvision.transforms.Resize(
                            (config.dataset_params.input_height, config.dataset_params.input_width)
                        )(infer(model, images)),
                        dim=0,
                    )  # unsqueeze to have a model-dimension (first)

        # -----------------------------------  Save image, depth, pred for visualization -------------------------------------------------

        if config.save_images and i == 0:
            for j in range(min(images.shape[0], 6)):
                image = images[j, :, :, :]
                depth = depths[j, :, :, :]
                pred = torch.mean(preds[:, j, :, :, :], dim=0)  # pred is average prediction
                var = (
                    torch.mean(preds[:, j, :, :, :] ** 2, dim=0) - pred**2
                )  # var is variance over model. since color dimension is 1, it is by pixel.

                os.makedirs(config.save_images_path, exist_ok=True)
                d = torch.tensor(
                    np.transpose(colorize(torch.squeeze(depth, dim=0), 0, 80), (2, 0, 1))
                )
                p = torch.tensor(
                    np.transpose(colorize(torch.squeeze(pred, dim=0), 0, 80), (2, 0, 1))
                )
                v = torch.tensor(
                    np.transpose(colorize(torch.squeeze(var, dim=0), 0, 80), (2, 0, 1))
                )
                if config.models.model_type == "ZoeNK":
                    im = transforms.ToPILImage()(
                        image
                    ).cpu()  # dont denormalize; it is the original image.
                else:
                    im = transforms.ToPILImage()(denormalize(image).cpu())
                print(type(d))

                # --------------------------------------  LOG AND SAVE  ---------------------------------------------------------------------------------

                log_images(
                    img=image.detach(),
                    depth=depth.detach(),
                    pred=pred.detach(),
                    vmin=config.dataset_params.min_depth,
                    vmax=config.dataset_params.max_depth,
                    step=j * 1e10,
                )
                im.save(
                    os.path.join(config.save_images_path, "/images/", f"{j}_{model_name}_img.png")
                )
                print(d.shape, p.shape, image.shape, pred.shape, depth.shape)
                transforms.ToPILImage()(d).save(
                    os.path.join(config.save_images_path, "/images/", f"{j}_{model_name}_depth.png")
                )
                transforms.ToPILImage()(p).save(
                    os.path.join(config.save_images_path, "/images/", f"{j}_{model_name}_pred.png")
                )
                transforms.ToPILImage()(v).save(
                    os.path.join(config.save_images_path, "/images/", f"{j}_{model_name}_var.png")
                )

                np.save(
                    os.path.join(
                        config.save_images_path, "/images/", f"np_img_{model_name}_{j}.npy"
                    ),
                    torch.squeeze(denormalize(image), dim=0).numpy(force=True),
                )
                np.save(
                    os.path.join(
                        config.save_images_path, "/images/", f"np_depth_{model_name}_{j}.npy"
                    ),
                    torch.squeeze(depth, dim=0).numpy(force=True),
                )
                np.save(
                    os.path.join(
                        config.save_images_path, "/images/", f"np_preds_{model_name}_{j}.npy"
                    ),
                    torch.squeeze(pred, dim=0).numpy(force=True),
                )
                uncertainty = torch.var(preds, dim=0)
        preds = torch.mean(preds, dim=0)  # pred is average prediction

        depths, preds, uncertainty = filter_valid(
            depths, preds, uncertainty=uncertainty, config=config
        )
        depths.to(device)
        preds.to(device)
        uncertainty.to(device)
        print(depths.device, depths_all_samples.device)

        depths_all_samples = torch.cat(
            [depths_all_samples, torch.flatten(depths).to(device)]
        )  # flatten to ensure sorting by uncertainty.
        preds_all_samples = torch.cat([preds_all_samples, torch.flatten(preds).to(device)])
        uncertainty_all_samples = torch.cat(
            [uncertainty_all_samples, torch.flatten(uncertainty).to(device)]
        )
        if config.in_debug and i > 0:
            continue
        i += 1

    _, sort_by_uncertainty_ascending_indices = torch.sort(uncertainty_all_samples)
    losses = calc_loss_metrics(
        depths_all_samples[sort_by_uncertainty_ascending_indices],
        preds_all_samples[sort_by_uncertainty_ascending_indices],
    )

    def uncertainty_results_df(fineness, sorted_preds, sorted_targets):
        df = pd.DataFrame(
            columns=["Share"] + [el for el in calc_loss_metrics(preds, depths).keys()]
        )
        for i in range(len(sorted_preds) // fineness + 1):  # increment in thousandths
            print(
                calc_loss_metrics(
                    sorted_preds[i * fineness :], sorted_targets[i * fineness :]
                ).values()
            )
            df.loc[i] = [(i * fineness) / (len(sorted_preds))] + [
                el.numpy(force=True)
                for el in calc_loss_metrics(
                    sorted_preds[i * 10 :], sorted_targets[i * 10 :]
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

    uncertainty_df.to_csv(
        os.path.join(
            config.save_images_path, "/uncertainty_df/" f"uncertainty_df_{config.models.model_type}"
        ),
        index=False,
    )
    save_uncertainty_plots(
        uncertainty_df,
        os.path.join(config.save_images_path, "/plots/" f"plot_{config.models.model_type}_"),
    )

    metrics.update(losses)
    if round_vals:

        def r(m):
            return torch.round(m, decimals=round_precision)

    else:

        def r(m):
            return m

    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics
