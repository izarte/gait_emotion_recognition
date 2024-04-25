from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "psimo_reduced"
LABELS_PATH = "metadata_labels_v3.csv"

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


def prepare_dataset():
    # Create a Path object for the base path
    dataset_path = Path(DATASET_PATH)

    # Append the labels filename to the base path
    labels_path = dataset_path / LABELS_PATH

    df = pd.read_csv(labels_path, nrows=10)
    # df = pd.read_csv(labels_path)
    df.drop(
        columns=[
            "ATTR_Weight(kg)",
            "ATTR_Height(cm)",
            "ATTR_Age",
            "ATTR_Gender",
            "ATTR_bmi",
        ],
        inplace=True,
    )

    for column in df.columns.drop("ID"):
        df[column] = df[column].replace(mapping)

    # Display the DataFrame
    print(df)


if __name__ == "__main__":
    prepare_dataset()
