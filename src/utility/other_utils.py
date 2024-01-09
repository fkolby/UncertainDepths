import torch
import numpy as np
from pprint import pprint
import pandas as pd
from src.utility.viz_utils import calc_loss_metrics


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """A dictionary of running averages."""

    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


 
if __name__=="__main__":
    test = calc_loss_metrics(
            targets= torch.rand((2,2,2)),
            preds=torch.rand((2,2,2)),
        )
    
    test = {k: [v.numpy(force=True)] for k,v in test.items()}
    file_name = "output_results.csv"
    try:
        results_df = pd.read_csv(file_name)
        out_df = pd.DataFrame(data = test )
        out_df["model_type"] = "pizza"
        out_df["identification"] = "1-2-3"
        pd.concat([out_df, results_df]).to_csv(file_name, index=False)
    except FileNotFoundError:
        out_df = pd.DataFrame(data = test)
        out_df["model_type"] = "pizza"
        out_df["identification"] = "1-2-3"
        out_df.to_csv(file_name, index=False)
        


  