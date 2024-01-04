import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns



def gather_model_dfs(identification_strings:[str], df_prefix= "scaled_")-> pd.DataFrame:

    scoring_metrics = {
        "delta1":"Delta 1",
        "delta2":"Delta 2",
        "delta3":"Delta 3",
        "abs_rel": "Absolute Relative Error",
        "rmse": "Root Mean Sq. Error",
        "log_10": "Log error",
        "rmse_log": "Root Mean Sq. Error of Logs",
        "silog":"Scale Invariant Log-loss",
        "sq_rel": "Squared relative error",
        "silogloss_loss_func": "Scale Invariant Log-loss", 
    }
    folder_location = "/home/jbv415/UncertainDepths/src/models/outputs/uncertainty_df"

    for i,run in enumerate(identification_strings):
        model_type = run.split("_")[-1]
        if model_type=="Laplace":
            model_type = "_".join(run.split("_")[-2:])
        
        path = os.path.join(folder_location, run, df_prefix + "uncertainty_df_" + model_type)

        if i==0:
            concatenated_df = pd.read_csv(path)
            concatenated_df["Model Type"] = " ".join(model_type.split("_"))
            concatenated_df["df_type"] = df_prefix.strip("_")
            if model_type.startswith("Laplace"):
                concatenated_df["identifier"] = "_".join(run.split("_")[:-3])
            else:
                print("runsplit", run.split("_")[:-1])
                concatenated_df["identifier"] = "_".join(run.split("_")[:-2])
        else:
            out = pd.read_csv(path)
            out["df_type"] = df_prefix.strip("_")
            out["Model Type"] = " ".join(model_type.split("_"))
            if model_type.startswith("Laplace"):
                out["identifier"] = "_".join(run.split("_")[:-2])
            else:
                out["identifier"] = "_".join(run.split("_")[:-1])
            concatenated_df = pd.concat([concatenated_df, out])

    return concatenated_df

def save_uncertainty_plots(df, file_prefix):
        for i, c in enumerate(df.columns):
            if c not in ["Share","Model Type","identifier","df_type"]:
                # axs[i].plot(df[[c]])

                plot = sns.lineplot(df,x="Share", y=c, hue = "Model Type")
                fig = plot.get_figure()
                fig.savefig(fname=file_prefix + c)
                fig.clear()
                """ plt.plot(x = df[["Share"]], df[[c]])
                plt.xlabel("Uncertainty: Share of predictions")
                plt.ylabel(c)
                plt.title(f"{c} as uncertainty increases")
                plt.savefig(fname=file_prefix + c)
                plt.clf() """






if __name__=="__main__":
    aggregate_dfs = gather_model_dfs(["01_01_2024_11_11_24_Posthoc_Laplace", "01_01_2024_13_50_34_Online_Laplace", "19_12_2023_07_07_45_Ensemble"])
    print(aggregate_dfs)
    save_uncertainty_plots(aggregate_dfs, "test_df_")