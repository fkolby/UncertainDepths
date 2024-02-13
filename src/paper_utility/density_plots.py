import torch
import numpy as np

import pandas as pd
from typing import List
import seaborn as sns
import os
import matplotlib.pyplot as plt

from src.utility.train_utils import seed_everything
from sklearn.metrics import roc_curve, auc

seed_everything(seed=442)


def density_plot(idents: List[str]) -> None:
    """Produces an aggregated roc-curve based on predicting OOD. 
    For each ident in idents, it generates such a roc-curve by predicting high uncertainty=OOD. 
    idents is a folder path with predictions of uncertainty of pixels and labelling of whether they are OOD. For more info, see report."""

    folder_location = "/home/jbv415/UncertainDepths/src/models/outputs/roc_curves/"
    lst = []
    for i, ident in enumerate(idents):
        if ident.split("_")[-1] == "Laplace":
            model = " ".join(ident.split("_")[-2:])
        else:
            model = ident.split("_")[-1]
        unc = torch.load(
            os.path.join(folder_location, ident, "_uncertainty_all_pixels.pt"),
            map_location=torch.device("cpu"),
        )
        sca = torch.load(
            os.path.join(folder_location, ident, "scaled_uncertainty_all_pixels.pt"),
            map_location=torch.device("cpu"),
        )
        OOD = torch.load(
            os.path.join(folder_location, ident, "OOD_class_all_pixels.pt"),
            map_location=torch.device("cpu"),
        )

        data = pd.DataFrame(
            data={
                "Scaled Uncertainty": sca.numpy(force=True).astype(np.float16).flatten(),
                "Uncertainty": unc.numpy(force=True).astype(np.float16).flatten(),
                "OOD class?": OOD.numpy(force=True).flatten().astype(bool),
            }
        )
        random_data = data.loc[np.random.choice(len(data), size=100000, replace=False), :]
        print(ident)

        plot = sns.histplot(
            data=random_data,
            x="Scaled Uncertainty",
            hue="OOD class?",
            binwidth=0.02,
            stat="probability",
            common_norm=False,
        )
        plot.get_figure().savefig(f"{model}_density_plot.png")
        plot.clear()

        if i == 0:
            random_data_df = pd.DataFrame(random_data)
            random_data_df["Model"] = model
        else:
            rd_df = pd.DataFrame(random_data)
            rd_df["Model"] = model

            random_data_df = pd.concat([random_data_df, rd_df])

        scaled_roc_curve_fpr, scaled_roc_curve_tpr, _ = roc_curve(
            y_true=random_data["OOD class?"], y_score=random_data["Scaled Uncertainty"]
        )
        roc_auc = round(auc(scaled_roc_curve_fpr, scaled_roc_curve_tpr), 3)

        lst.append(
            pd.DataFrame(
                data={
                    "TPR": scaled_roc_curve_tpr,
                    "FPR": scaled_roc_curve_fpr,
                    "Model": f"{model}, AUROC: {roc_auc}",
                }
            )
        )

    roc_df = pd.concat(lst)

    roc_plot = sns.lineplot(data=roc_df, y="TPR", x="FPR", hue="Model")
    sns.lineplot(x=[0, 1], y=[0, 1])
    roc_plot.get_figure().savefig("ROC_curve_classify_OOD.png")

    faceted_plot = sns.FacetGrid(random_data_df, col="Model", col_wrap=2)
    faceted_plot.map_dataframe(
        sns.histplot,
        x="Scaled Uncertainty",
        hue="OOD class?",
        binwidth=0.01,
        stat="probability",
        common_norm=False,
        binrange=(0, 0.6),
    )
    faceted_plot.savefig("test_distribution.png")


if __name__ == "__main__":
    density_plot(
        [
            "2024_01_31_01_02_04_185_Dropout",
            "2024_01_31_04_51_37_185_Ensemble",
            "2024_01_30_21_55_26_185_Posthoc_Laplace",
            "2024_01_30_18_59_42_185_Online_Laplace",
        ]
    )
