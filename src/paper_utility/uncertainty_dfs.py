import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def gather_model_dfs(identification_strings: [str], df_prefixs=["scaled_", ""]) -> pd.DataFrame:
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

            path = os.path.join(folder_location, run, df_prefix + "uncertainty_df_" + model_type + ".csv")

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
    for i, c in enumerate(df.columns):
        if c not in ["Share", "Model Type", "identifier", "Uncertainty type"]:
            print(c)

            # axs[i].plot(df[[c]])
            #g = sns.FacetGrid(df, col="Uncertainty type")
            #plot = g.map_dataframe(sns.lineplot, x="Share", y=c, hue="Model Type")
            if c in ["Mean Squared Error", "Root Mean Squared Error"]:
                metric_relevant_uncertainty = " SD"
            else:
                metric_relevant_uncertainty = "Scaled SD"
            df_c = df.loc[df["Uncertainty type"] == metric_relevant_uncertainty]
            plot = sns.lineplot(data=df_c, hue="Model Type", x = "Share", y=c)
            plot.set_xlabel("Share - " + metric_relevant_uncertainty)
            

            fig = plot.get_figure()
            fig.get_figure().savefig(fname=file_prefix + c)
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
            "2024_01_31_01_02_04_185_Dropout",
            "2024_01_31_04_51_37_185_Ensemble",
            "2024_01_30_21_55_26_185_Posthoc_Laplace",
            "2024_01_30_18_59_42_185_Online_Laplace",
        ]
    )
    print(aggregate_dfs.loc[aggregate_dfs["Uncertainty type"] != "scaled"])
    save_uncertainty_plots(aggregate_dfs, "test_mini_report")
