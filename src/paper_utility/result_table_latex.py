import pandas as pd
import os
import torch
from typing import List


def df_to_latex(
    filter_identities: List[str],
    cols_of_interest: List[str] = [
        "model_type",
        "delta1",
        "delta2",
        "delta3",
        "abs_rel",
        "rmse",
        "silog",
    ],
    file_name: str = "test_resultsoutput_results.csv",
):
    """Turns a dataframe into latex, by converting names into prettier code. Also selects col and row of interest., row being based on filter identitites"""

    scoring_metrics = {
        "delta1": "$\delta_1\\uparrow$",
        "delta2": "$\delta_2\\uparrow$",
        "delta3": "$\delta_3\\uparrow$",
        "abs_rel": "$REL\\downarrow$",
        "rmse": "$RMSE\\downarrow$",
        "log_10": "Log error",
        "rmse_log": "Root Mean Squared Error of Logs",
        "silog": "$SIL\\downarrow$",
        "sq_rel": "Squared relative error",
        "silogloss_loss_func": "Scale Invariant Log loss function",
        "mse_loss_func": "Mean Squared Error",
        "model_type": "Model type",
        "uncertainty_in_dist": "$\sigma_{ID}$",
        "uncertainty_out_of_dist": "$\sigma_{OOD}$",
        "scaled_uncertainty_in_dist": "$\sigma^{scaled}_{ID}$",
        "scaled_uncertainty_out_of_dist": "$\sigma^{scaled}_{OOD}$",
    }
    folder_location = "/home/jbv415/UncertainDepths/src/models/outputs/"

    identification_ids = []

    model_names = {
        "Posthoc_Laplace": "Posthoc Laplace",
        "Online_Laplace": "Online Laplace",
        "Ensemble": "Ensemble",
        "Dropout": "Dropout",
        "ZoeNK": "ZoeNK",
    }

    for f in filter_identities:
        id = f.split("_")[:-1]
        id = "_".join(id)

        identification_ids.append(id)
    print("ids", identification_ids)

    df = pd.read_csv(os.path.join(folder_location, "test_output_results.csv"))
    print(df)
    print(df.columns)
    df = df.loc[df["identification"].isin(identification_ids)]
    df = df[cols_of_interest]
    df["model_type"] = df["model_type"].map(lambda x: model_names[x])

    df = df.rename(columns=scoring_metrics)

    for c in df.columns:
        if df[c].dtype == object and c != "Model type":
            print(df[c])

            print(c)
            if df[c][1].startswith("tensor("):
                df[c] = df[c].map(lambda h: float(h.strip("tensor(").strip(")")))

    print(df.to_latex(index=False, float_format="%.3f", escape=False))
    return True


if __name__ == "__main__":
    filter_identities = [
            "2024_02_15_09_38_48_6142_Dropout",
            "2024_02_15_11_58_42_6215_Ensemble",
            "2024_02_15_09_38_48_6141_Posthoc_Laplace",
            "2024_02_15_09_38_05_6139_Online_Laplace",
        ]    
    df_to_latex(filter_identities=filter_identities)
    df_to_latex(
        filter_identities=filter_identities,
        cols_of_interest=[
            "model_type",
            "delta1",
            "delta2",
            "delta3",
            "abs_rel",
            "rmse",
            "silog"
        ],
    )
