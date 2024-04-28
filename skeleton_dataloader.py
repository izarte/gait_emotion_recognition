import json
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from torch_geometric.data import Dataset, DataLoader
import torch

mapping = {
    "Low": 0,
    "Mild": 0,
    "Minor Distress": 0,
    "Normal / Low": 1,
    "Moderate": 1,
    "Moderate / Low": 1,
    "Typical": 1,
    "Normal": 2,
    "Major Distress": 2,
    "Normal / High": 3,
    "Moderate / High": 3,
    "Severe": 3,
    "High": 4,
    "Extremely Severe": 4,
}


class SkeletonDataloader(Dataset):
    def __init__(self, data, labels_path):
        super(SkeletonDataloader, self).__init__()
        self.data = data
        self.read_labels(labels_path)

    def read_labels(self, labels_path):
        Y = pd.read_csv(labels_path, nrows=10)
        # Y = pd.read_csv(labels_path)
        Y.drop(
            columns=[
                "ATTR_Weight(kg)",
                "ATTR_Height(cm)",
                "ATTR_Age",
                "ATTR_Gender",
                "ATTR_bmi",
            ],
            inplace=True,
        )

        for column in Y.columns.drop("ID"):
            Y[column] = Y[column].replace(mapping)

        self.labels_df = Y
        # Display the DataFrame
        # print(Y.columns)
        # print(Y.query("ID == 3").drop(["ID"], axis=1).to_numpy()[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        skeletons = self.data[idx]["skeletons"]
        labels = (
            self.labels_df.query(f"ID == {self.data[idx]['id']}")
            .drop(["ID"], axis=1)
            .to_numpy()[0]
        )
        return skeletons, labels
