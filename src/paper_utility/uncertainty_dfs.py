import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import List


def gather_model_dfs(
    identification_strings: List[str], df_prefixs: List[str] = ["scaled_", ""]
) -> pd.DataFrame:
    """Join category, model dataframes based into longformat. Here category is a dataframe with uncertainties of given type, and identification strings
    Give identifications of model."""

    scoring_metrics = {
        "delta1": "Delta 1",
        "delta2": "Delta 2",
        "delta3": "Delta 3",
        "abs_rel": "Absolute Relative Error",
        "rmse": "Root Mean Squared Error",
        "log_10": "Log error",
        "rmse_log": "Root Mean Squared Error of Logs",
        "silog": "Scale Invariant Log-loss",
        "sq_rel": "Squared relative error",
        "silogloss_loss_func": "Scale Invariant Logloss",
        "mse_loss_func": "Mean Squared Error",
    }
    folder_location = "/home/jbv415/UncertainDepths/src/models/outputs/uncertainty_df"

    for j, df_prefix in enumerate(df_prefixs):
        for i, run in enumerate(identification_strings):
            model_type = run.split("_")[-1]
            if model_type == "Laplace":
                model_type = "_".join(run.split("_")[-2:])

            path = os.path.join(
                folder_location, run, df_prefix + "uncertainty_df_" + model_type + ".csv"
            )

            if i + j == 0:
                concatenated_df = pd.read_csv(path)
                concatenated_df["Model Type"] = " ".join(model_type.split("_"))
                concatenated_df["Uncertainty type"] = str.capitalize(df_prefix.strip("_")) + " SD"
                if model_type.startswith("Laplace"):
                    concatenated_df["identifier"] = "_".join(run.split("_")[:-3])
                else:
                    print("runsplit", run.split("_")[:-1])
                    concatenated_df["identifier"] = "_".join(run.split("_")[:-2])
            else:
                out = pd.read_csv(path)
                out["Uncertainty type"] = str.capitalize(df_prefix.strip("_")) + " SD"
                out["Model Type"] = " ".join(model_type.split("_"))
                if model_type.startswith("Laplace"):
                    out["identifier"] = "_".join(run.split("_")[:-2])
                else:
                    out["identifier"] = "_".join(run.split("_")[:-1])
                concatenated_df = pd.concat([concatenated_df, out])
    concatenated_df = concatenated_df.rename(columns=scoring_metrics)

    return concatenated_df


def save_uncertainty_plots(df, file_prefix):
    """Generates monotonicty plots of different metrics, e.g. RMSE, MSE or the like, sorted by some other metric, e.g. uncertainty"""

    for i, c in enumerate(df.columns):
        if c not in ["Share", "Model Type", "identifier", "Uncertainty type"]:
            print(c)

            # axs[i].plot(df[[c]])
            # g = sns.FacetGrid(df, col="Uncertainty type")
            # plot = g.map_dataframe(sns.lineplot, x="Share", y=c, hue="Model Type")
            if c in ["Mean Squared Error", "Root Mean Squared Error"]:
                metric_relevant_uncertainty = "Scaled SD"  # " SD"
            else:
                metric_relevant_uncertainty = "Scaled SD"
            df_c = df.loc[df["Uncertainty type"] == metric_relevant_uncertainty]
            plot = sns.lineplot(data=df_c, hue="Model Type", x="Share", y=c)
            # plot.set_xlabel("Share - " + metric_relevant_uncertainty)
            plot.set_xlabel("Share of pixels - sorted by uncertainty")

            fig = plot.get_figure()
            fig.savefig(fname=file_prefix + c + ".png")
            print(file_prefix + c)
            fig.clear()
            """ plt.plot(x = df[["Share"]], df[[c]])
                plt.xlabel("Uncertainty: Share of predictions")
                plt.ylabel(c)
                plt.title(f"{c} as uncertainty increases")
                plt.savefig(fname=file_prefix + c)
                plt.clf() """


if __name__ == "__main__":
    aggregate_dfs = gather_model_dfs(
        [
            "2024_02_15_09_38_48_6142_Dropout",
            "2024_02_15_11_58_42_6215_Ensemble",
            "2024_02_15_09_38_48_6141_Posthoc_Laplace",
            "2024_02_15_09_38_05_6139_Online_Laplace",
        ]
    )
    print("Hi")
    print(aggregate_dfs.loc[aggregate_dfs["Uncertainty type"] != "scaled"])
    save_uncertainty_plots(aggregate_dfs, "final_report")
