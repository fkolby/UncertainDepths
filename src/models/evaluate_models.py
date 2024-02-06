import os
import pdb
import timeit
from datetime import datetime
import psutil

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch_laplace import MSEHessianCalculator
from pytorch_laplace.laplace.diag import DiagLaplace
from pytorch_laplace.optimization.prior_precision import optimize_prior_precision
from torchinfo import summary
from tqdm import tqdm
import copy

from sklearn.metrics import roc_curve

import wandb
from src.utility.debug_utils import time_since_previous_log
from src.utility.eval_utils import calc_loss_metrics, filter_valid, save_eval_images
from src.utility.other_utils import RunningAverageDict
from src.utility.train_utils import seed_everything


@torch.no_grad()
def eval_model(
    model,
    test_loader,
    cfg,
    dataloader_for_hessian=None,
    round_vals=True,
    round_precision=3,
    dont_log_wandb=False,
    **kwargs,
):
    metrics = RunningAverageDict()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    #set up saving directories
    date_and_time_and_model = (
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        + "_"
        + str(os.environ.get("SLURM_JOB_ID"))
        + "_"
        + cfg.models.model_type
    )
    os.makedirs(
        os.path.join(cfg.save_images_path, "uncertainty_df/", date_and_time_and_model),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(cfg.save_images_path, "roc_curves/", date_and_time_and_model),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(cfg.save_images_path, "plots/", date_and_time_and_model), exist_ok=True
    )
    i = 0
    

    #Calculate hessians and priors for those models
    if cfg.models.model_type == "Posthoc_Laplace":
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
            )*cfg.models.hessian_multiplicator
            if cfg.in_debug and a > 25:
                break
        
        print("Done calcing hessian")
        mean_parameter = torch.nn.utils.parameters_to_vector(model.parameters())

        with torch.enable_grad():
            prior_precision = optimize_prior_precision(
                mu_q=mean_parameter,
                hessian=hessian,
                prior_prec=torch.tensor([float(cfg.models.prior_prec)], device=device),
                n_steps=5000,
            )
        print(prior_precision)
        print(kwargs.get("dont_optimize_prior_prec", False))
        print(kwargs)
        if kwargs.get("dont_optimize_prior_prec", False):
            print("prior_pres")
            prior_precision = cfg.prior_prec
            
        print(f"prior precision: {prior_precision}")
    if cfg.models.model_type == "Online_Laplace":
        hessian = kwargs.pop("online_hessian")
        mean_parameter = torch.nn.utils.parameters_to_vector(model.parameters())
        with torch.enable_grad():
            prior_precision = optimize_prior_precision(
                mu_q=mean_parameter,
                hessian=hessian,
                prior_prec=torch.tensor([float(cfg.models.prior_prec)], device=device),
                n_steps=5000,
            )
        if kwargs.get("dont_optimize_prior_prec", False):
            prior_precision = cfg.prior_prec   
        print(f"prior precision: {prior_precision}")



    # ======================================================   PREDICTING  ===============================================

    filtered_depths_all_samples = torch.Tensor().to(device="cpu")
    filtered_preds_all_samples = torch.Tensor().to(device="cpu")
    filtered_uncertainty_all_samples = torch.Tensor().to(device="cpu")
    filtered_scaled_uncertainty_all_samples = torch.Tensor().to(device="cpu")
    scaled_uncertainty_all_pixels = torch.tensor([]).to(device="cuda")
    uncertainty_all_pixels = torch.tensor([]).to(device="cuda")
    OOD_class_all_pixels= torch.tensor([]).to(device="cuda")


    if cfg.OOD.use_white_noise_box_test:
        # in that case we should sort by class, so see whether uncertainties are higher on OOD.
        uncertainty_in_distribution = torch.Tensor().to(device="cpu")
        uncertainty_out_of_distribution = torch.Tensor().to(device="cpu")
        scaled_uncertainty_in_distribution = torch.Tensor().to(device="cpu")
        scaled_uncertainty_out_of_distribution = torch.Tensor().to(device="cpu")
    seed_everything(cfg.seed)

    for i, sample in enumerate(tqdm(test_loader, total=len(test_loader))):
        if i > 3 and cfg.in_debug and cfg.dont_test_oom:  # skip if too long.
            continue

        if "has_valid_depth" in sample:
            if not sample["has_valid_depth"]:
                print("validDEPTH?")
                continue
        images, depths, OOD_class = sample
        images = images.to(device=device)
        depths = depths.to(device="cpu")  # BxCxHxW

        match cfg.models.model_type:
            case "Posthoc_Laplace":
                preds, uncertainty = laplace.laplace(
                    x=images,
                    model=model,
                    hessian=hessian,
                    prior_prec=prior_precision,
                    n_samples=cfg.models.n_models,
                )  ###NOTE TO SELF: MUST BE NON-flipped for other models as well
                preds = preds.detach().to("cpu")
                uncertainty = uncertainty.to("cpu")
                print("shapes:", preds.shape, uncertainty.shape)
                print(uncertainty)
                print(uncertainty[0, :, :, :])
                # dimensions: BatchxColorxHxW (no model-dim.)
            case "Online_Laplace":
                laplace = DiagLaplace()
                preds, uncertainty = laplace.laplace(
                    x=images,
                    model=model,
                    hessian=hessian,
                    prior_prec=prior_precision,
                    n_samples=cfg.models.n_models,
                )  ###NOTE TO SELF: MUST BE NON-flipped for other models as well
                preds = preds.detach().to("cpu")
                uncertainty = uncertainty.to("cpu")
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
                    device="cpu",
                )  # dimensions: model, batch, color, height, width.
                for model_idx in range(cfg.models.n_models):
                    model_path = kwargs.get("model_path")
                    model.load_state_dict(torch.load(os.path.join(model_path,f"{cfg.models.model_type}_{model_idx}.pt")))
                    print(model)
                    preds[j, :, :, :, :] = torch.unsqueeze(model(images), dim=0).to(
                        device="cpu"
                    )  # unsqueeze to have a model-dimension (first)
                    j += 1
            case "Dropout":  # Only one model, but is different every call.
                model = model.train()
                preds = torch.zeros(
                    size=(
                        cfg.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        cfg.dataset_params.input_height,
                        cfg.dataset_params.input_width,
                    ),
                    device="cpu",
                )  # dimensions: model, batch, color, height, width.
                for j in range(cfg.models.n_models):
                    preds[j, :, :, :, :] = torch.unsqueeze(model(images), dim=0).to(
                        device="cpu"
                    )  # unsqueeze to have a model-dimension (first) - move to cpu to free memory on gpu

            case "ZoeNK":
                print(
                    (
                        cfg.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        cfg.dataset_params.input_height,
                        cfg.dataset_params.input_width,
                    )
                )
                preds = torch.zeros(
                    size=(
                        cfg.models.n_models,
                        depths.shape[0],
                        depths.shape[1],
                        cfg.dataset_params.input_height,
                        cfg.dataset_params.input_width,
                    ),
                    device="cpu",
                )  # dimensions: model, batch, color, height, width.
                for j in range(cfg.models.n_models):
                    print(model(images))
                    preds[j, :, :, :, :] = torch.unsqueeze(
                        torchvision.transforms.Resize(
                            (cfg.dataset_params.input_height, cfg.dataset_params.input_width)
                        )(model(images)["metric_depth"]),
                        dim=0,
                    ).to(
                        device="cpu"
                    )  # unsqueeze to have a model-dimension (first)

        # -----------------------------------  Save image, depth, pred for visualization -------------------------------------------------

        if cfg.save_images and i == 0:
            if cfg.models.model_type not in [
            "Posthoc_Laplace",
            "Online_Laplace",]:
                save_eval_images(
                    images=images,
                    depths=depths,
                    preds=preds,
                    uncertainty=None,
                    date_and_time_and_model=date_and_time_and_model,
                    cfg=cfg,
                    dont_log_wandb=dont_log_wandb,
                )
            else:
                save_eval_images(
                    images=images,
                    depths=depths,
                    preds=preds,
                    uncertainty=uncertainty,
                    date_and_time_and_model=date_and_time_and_model,
                    cfg=cfg,
                    dont_log_wandb=dont_log_wandb,
                )

        # -----------------------------------  Compute uncertainties and sort descending ----------------------------------------------------

        if cfg.models.model_type not in [
            "Posthoc_Laplace",
            "Online_Laplace",
        ]:  # then collapse on different predictions by each model.
            pred = torch.mean(preds, dim=0)

            uncertainty = torch.var(preds, dim=0).sqrt()
        else:
            pred = preds  # only named preds for consistency with other models - naming here might be worth rewriting.
            # uncertainty already=uncertainty
        
        print(psutil.virtual_memory(), flush=True)

        scaled_uncertainty_uncolored = uncertainty / pred

        print(
            depths.shape,
            pred.shape,
            OOD_class.shape,
            images.shape,
            scaled_uncertainty_uncolored,
            uncertainty.shape,
        )
        if cfg.OOD.use_white_noise_box_test:
            # in that case we should sort by class, so see whether uncertainties are higher on OOD.
            OOD_class_unfiltered = copy.deepcopy(OOD_class).to("cuda")
            # Obtain indices for valid uncertainty estimates (e.g depth within 0-80 meters (based on ground truth - which is not masked by white noise - so e.g. a white-noise box placed in sky would not be counted here (as depth>80 m))).
            print(
                "OOD:",
                np.mean(OOD_class.numpy()),
                OOD_class.shape,
                OOD_class[OOD_class == 0].shape,
                OOD_class[OOD_class == 1].shape,
                flush=True,
            )
            _, _, uncertainty_whitenoise = filter_valid(
                gt=depths, pred=OOD_class, uncertainty=uncertainty, config=cfg
            )  # hacky way of obtaining valid pixels for OOD-class - we only filter on depths (ground truth)
            _, OOD_class, scaled_uncertainty_whitenoise = filter_valid(
                gt=depths, pred=OOD_class, uncertainty=scaled_uncertainty_uncolored, config=cfg
            )
            print(
                "OOD:",
                np.mean(OOD_class.numpy()),
                OOD_class.shape,
                OOD_class[OOD_class == 0].shape,
                OOD_class[OOD_class == 1].shape,
                flush=True,
            )

            # aggregate white noise estimates.
            uncertainty_in_distribution = torch.cat(
                [uncertainty_in_distribution, uncertainty_whitenoise[OOD_class == 0].flatten()]
            )
            uncertainty_out_of_distribution = torch.cat(
                [uncertainty_out_of_distribution, uncertainty_whitenoise[OOD_class == 1].flatten()]
            )
            scaled_uncertainty_in_distribution = torch.cat(
                [
                    scaled_uncertainty_in_distribution,
                    scaled_uncertainty_whitenoise[OOD_class == 0].flatten(),
                ]
            )
            scaled_uncertainty_out_of_distribution = torch.cat(
                [
                    scaled_uncertainty_out_of_distribution,
                    scaled_uncertainty_whitenoise[OOD_class == 1].flatten(),
                ]
            )
        #need in full length for kdeplot. as running out of cpu mem, dont move to cpu.
        
        if i <= 50:
            uncertainty_all_pixels = torch.concat([uncertainty_all_pixels,uncertainty.to(device="cuda")])
            OOD_class_all_pixels = torch.concat([OOD_class_all_pixels,OOD_class_unfiltered])
            scaled_uncertainty_all_pixels = torch.concat([scaled_uncertainty_all_pixels,uncertainty.to(device="cuda").sqrt()/pred.to(device="cuda")])
            print("uncertainty:",uncertainty_all_pixels.shape,"ood:" , OOD_class_all_pixels.shape, "scaled:", scaled_uncertainty_all_pixels.shape)

        depths, pred, uncertainty = filter_valid(depths, pred, uncertainty=uncertainty, config=cfg)

        depths.to("cpu")
        pred.to("cpu")
        uncertainty.to("cpu")
        scaled_uncertainty_uncolored = uncertainty.sqrt() / pred


        print(depths.device, filtered_depths_all_samples.device)
        print(filtered_depths_all_samples.shape, filtered_uncertainty_all_samples.shape, filtered_preds_all_samples.shape)
        print(depths.shape, uncertainty.shape, pred.shape)

        print(torch.cuda.memory_summary())

        # protect against ram, cpu-ram OOM. 
        filtered_depths_all_samples = torch.cat(
            [filtered_depths_all_samples, torch.flatten(depths).to("cpu")]
        )  # flatten to ensure sorting by uncertainty.
        filtered_preds_all_samples = torch.cat([filtered_preds_all_samples, torch.flatten(pred).to("cpu")])
        filtered_uncertainty_all_samples = torch.cat(
            [filtered_uncertainty_all_samples, torch.flatten(uncertainty).to("cpu")]
        )
        filtered_scaled_uncertainty_all_samples = torch.cat(
            [filtered_scaled_uncertainty_all_samples, torch.flatten(scaled_uncertainty_uncolored).to("cpu")]
        )
        print(psutil.virtual_memory(), flush=True)
        i += 1
    

    if cfg.OOD.use_white_noise_box_test:  # log it.
        print(
            "=" * 20,
            "\n",
            "uncertainty In distribution:",
            uncertainty_in_distribution.mean(),
            "uncertainty out of distribution",
            uncertainty_out_of_distribution.mean(),
            "scaled uncertainty in distribution",
            scaled_uncertainty_in_distribution.mean(),
            "scaled uncertainty out of distribution",
            scaled_uncertainty_out_of_distribution.mean(),
            "\n",
            "=" * 20,flush=True,
        )
        print(psutil.virtual_memory(), flush=True)
        unscaled_kde_data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Class": "ID",
                        "Uncertainty": uncertainty_in_distribution,
                    }
                ),
                pd.DataFrame(
                    {
                        "Class": "OOD",
                        "Uncertainty": uncertainty_out_of_distribution,
                    }
                ),
            ]
        )

        scaled_kde_data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Class": "ID",
                        "Uncertainty": scaled_uncertainty_in_distribution,
                    }
                ),
                pd.DataFrame(
                    {
                        "Class": "OOD",
                        "Uncertainty": scaled_uncertainty_out_of_distribution,
                    }
                ),
            ]
        )
        print("gothere!", 1, flush=True)

        unscaled_kde_data.to_csv(
        os.path.join(
            cfg.save_images_path,
            "uncertainty_df/",
            date_and_time_and_model,
            f"unscaled_kde_data_{cfg.models.model_type}.csv",
        ),
        )
        scaled_kde_data.to_csv(
        os.path.join(
            cfg.save_images_path,
            "uncertainty_df/",
            date_and_time_and_model,
            f"scaled_kde_data_{cfg.models.model_type}.csv",
        ),
        )


        ## GENERATES OOM
        """  
        sns.kdeplot(unscaled_kde_data, x="Uncertainty", hue="Class", common_norm=False,bw_adjust=0.5).get_figure().savefig(
            os.path.join(
                cfg.save_images_path,
                "plots/",
                date_and_time_and_model,
                f"predict_OOD_by_variance_{cfg.models.model_type}",
            )
        )
        sns.kdeplot(scaled_kde_data, x="Uncertainty", hue="Class",common_norm=False,bw_adjust=0.5).get_figure().savefig(
            os.path.join(
                cfg.save_images_path,
                "plots/",
                date_and_time_and_model,
                f"predict_OOD_by_variance_scaled_{cfg.models.model_type}",
            )
        ) """

    print("gothere!", 2, flush=True)
    _, sort_by_uncertainty_ascending_indices = torch.sort(filtered_uncertainty_all_samples)
    _, sort_by_scaled_uncertainty_ascending_indices = torch.sort(filtered_scaled_uncertainty_all_samples)

    del filtered_scaled_uncertainty_all_samples
    del filtered_uncertainty_all_samples
    del model


    print(
        "sum of absolute differences in ranking:",
        torch.sum(
            torch.abs(
                torch.tensor(sort_by_scaled_uncertainty_ascending_indices)
                - torch.tensor(sort_by_uncertainty_ascending_indices)
            )
        ),
        flush=True
    )
    print(
        "count of absolute differences in ranking:",
        torch.sum(
            torch.abs(
                torch.tensor(sort_by_scaled_uncertainty_ascending_indices)
                - torch.tensor(sort_by_uncertainty_ascending_indices)
            )
            > 0
        ),
        flush=True
    )

    losses = calc_loss_metrics(
        targets=filtered_depths_all_samples[sort_by_uncertainty_ascending_indices],
        preds=filtered_preds_all_samples[sort_by_uncertainty_ascending_indices],
    )

    def uncertainty_results_df(fineness, sorted_preds, sorted_targets):
        if cfg.in_debug:
            fineness=10
        df = pd.DataFrame(
            columns=["Share"]
            + [el for el in calc_loss_metrics(sorted_preds, sorted_targets).keys()]
        )
        for i in range(fineness):  # increment in thousandths
            print(f"{i} of {fineness}", flush=True)
            df.loc[i] = [i / fineness] + [
                el.numpy(force=True)
                for el in calc_loss_metrics(
                    preds=sorted_preds[int(i / fineness * len(sorted_preds)) :],
                    targets=sorted_targets[
                        int(i / fineness * len(sorted_preds)) :
                    ],  # sorted_preds/targets should be same length.
                ).values()
            ]
        return df

    print("gothere!", 3, flush=True)
    uncertainty_df = uncertainty_results_df(
        1000,
        sorted_targets=filtered_depths_all_samples[sort_by_uncertainty_ascending_indices],
        sorted_preds=filtered_preds_all_samples[sort_by_uncertainty_ascending_indices],
    )
    scaled_uncertainty_df = uncertainty_results_df(
        1000,
        sorted_targets=filtered_depths_all_samples[sort_by_scaled_uncertainty_ascending_indices],
        sorted_preds=filtered_preds_all_samples[sort_by_scaled_uncertainty_ascending_indices],
    )
    print("gothere!", 4, flush=True)

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
            cfg.save_images_path,
            "uncertainty_df/",
            date_and_time_and_model,
            f"uncertainty_df_{cfg.models.model_type}.csv",
        ),
        index=False,
    )
    scaled_uncertainty_df.to_csv(
        os.path.join(
            cfg.save_images_path,
            "uncertainty_df/",
            date_and_time_and_model,
            f"scaled_uncertainty_df_{cfg.models.model_type}.csv",
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
    save_uncertainty_plots(
        scaled_uncertainty_df,
        os.path.join(
            cfg.save_images_path,
            "plots/",
            date_and_time_and_model,
            f"scaled_plot_{cfg.models.model_type}_",
        ),
    )
    


    print(OOD_class_all_pixels.shape, scaled_uncertainty_all_pixels.shape)

    torch.save(
            OOD_class_all_pixels,
            os.path.join(
            cfg.save_images_path,
            "roc_curves/",
            date_and_time_and_model,
            f"OOD_class_all_pixels.pt",
            )            
        )
    torch.save(
            scaled_uncertainty_all_pixels,
            os.path.join(
            cfg.save_images_path,
            "roc_curves/",
            date_and_time_and_model,
            f"scaled_uncertainty_all_pixels.pt",
            )            
        )
    torch.save(
            uncertainty_all_pixels,
            os.path.join(
            cfg.save_images_path,
            "roc_curves/",
            date_and_time_and_model,
            f"_uncertainty_all_pixels.pt",
            )            
        )
    
    """  
    # Classify OOD by variance ROC
    scaled_roc_curve_fpr, scaled_roc_curve_tpr, _ = roc_curve(
        y_true=OOD_class_all_pixels.flatten().numpy(force=True), y_score=scaled_uncertainty_all_pixels.flatten().numpy(force=True)
    )  # using variance as prediction, we compute tpr and fpr for different thresholds of variance
    
    unscaled_roc_curve_fpr, unscaled_roc_curve_tpr, _ = roc_curve(
        y_true=OOD_class_all_pixels.flatten().numpy(force=True), y_score=uncertainty_all_pixels.flatten().numpy(force=True)
    )  # using variance as prediction, we compute tpr and fpr for different thresholds of variance


    print("scaled fpr/tpr", scaled_roc_curve_fpr, scaled_roc_curve_tpr,"unscaled", unscaled_roc_curve_fpr, unscaled_roc_curve_tpr)
    print("scaled fpr/tpr", scaled_roc_curve_fpr.shape, scaled_roc_curve_tpr.shape,"unscaled", unscaled_roc_curve_fpr.shape, unscaled_roc_curve_tpr.shape)

    print(pd.DataFrame(
                    data={
                        "Model": cfg.models.model_type,
                        "tpr": scaled_roc_curve_tpr,
                        "fpr": scaled_roc_curve_fpr,
                        "Scaled": "Scaled",
                    }
                ))

    roc_curves_df = pd.concat(
            [
                pd.DataFrame(
                    data={
                        "Model": cfg.models.model_type,
                        "tpr": scaled_roc_curve_tpr,
                        "fpr": scaled_roc_curve_fpr,
                        "Scaled": "Scaled",
                    }
                ),
                pd.DataFrame(
                    data={
                        "Model": cfg.models.model_type,
                        "tpr": unscaled_roc_curve_tpr,
                        "fpr": unscaled_roc_curve_fpr,
                        "Scaled": "Unscaled",
                    }
                ),
            ]
        )

    print(roc_curves_df)

    roc_curves_df.to_csv(
        os.path.join(
            cfg.save_images_path,
            "uncertainty_df/",
            date_and_time_and_model,
            f"roc_curves_{cfg.models.model_type}.csv",
        ),
        index=False,
    )
    """
    print("Updating metrics")
    metrics.update(losses)
    if round_vals:

        def r(m):
            return torch.round(m, decimals=round_precision)

    else:

        def r(m):
            return m

    metrics = {k: r(v) for k, v in metrics.get_value().items()}

    save_metrics = {k: [v.numpy(force=True)] for k, v in metrics.items()}
    file_name = os.path.join(cfg.save_images_path, cfg.results_df_prefix + "output_results.csv")
    try:
        results_df = pd.read_csv(file_name)
        out_df = pd.DataFrame(data=save_metrics)

        out_df["uncertainty_in_dist"] = (float(uncertainty_in_distribution.mean()),)
        out_df["uncertainty_out_of_dist"] = (float(uncertainty_out_of_distribution.mean()),)
        out_df["scaled_uncertainty_in_dist"] = (float(scaled_uncertainty_in_distribution.mean()),)
        out_df["scaled_uncertainty_out_of_dist"] = (
            float(scaled_uncertainty_out_of_distribution.mean()),
        )
        out_df["model_type"] = cfg.models.model_type
        out_df["identification"] = "_".join(date_and_time_and_model.split("_")[:-1])
        pd.concat([out_df, results_df]).to_csv(file_name, index=False)
    except FileNotFoundError:
        out_df = pd.DataFrame(data=save_metrics)
        out_df["model_type"] = cfg.models.model_type

        out_df["uncertainty_in_dist"] = (float(uncertainty_in_distribution.mean()),)
        out_df["uncertainty_out_of_dist"] = (float(uncertainty_out_of_distribution.mean()),)
        out_df["scaled_uncertainty_in_dist"] = (float(scaled_uncertainty_in_distribution.mean()),)
        out_df["scaled_uncertainty_out_of_dist"] = (
            float(scaled_uncertainty_out_of_distribution.mean()),
        )
        out_df["identification"] = "_".join(date_and_time_and_model.split("_")[:-1])
        out_df.to_csv(file_name, index=False)
    
    if not dont_log_wandb:
            wandb.log(
                {
                    "uncertainty In distribution:": uncertainty_in_distribution.mean(),
                    "uncertainty out of distribution": uncertainty_out_of_distribution.mean(),
                    "scaled uncertainty in distribution": scaled_uncertainty_in_distribution.mean(),
                    "scaled uncertainty out of distribution": scaled_uncertainty_out_of_distribution.mean(),
                }
            )



    return metrics
