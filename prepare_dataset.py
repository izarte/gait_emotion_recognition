from pathlib import Path
import pandas as pd
import json

DATASET_PATH = "psimo_reduced"
LABELS_PATH = "metadata_labels_v3.csv"
DATA_PATH = "semantic_data"


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


def load_keypoints(json_file_path, id):
    # Open and read the JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # Extract 'keypoints' from each dictionary in the list
    keypoints_list = [item["keypoints"] for item in data if "keypoints" in item]

    # Create a DataFrame from the list of keypoints
    df = pd.DataFrame(keypoints_list)
    df["ID"] = id

    return df


def prepare_dataset():
    # Create a Path object for the base path
    dataset_path = Path(DATASET_PATH)

    # Append the labels filename to the base path
    data_path = dataset_path / DATA_PATH / "skeletons"
    labels_path = dataset_path / LABELS_PATH

    experiments = [d.name for d in data_path.iterdir() if d.is_dir()]

    dfs = []
    for experiment in experiments:
        print(experiment)
        experiment_path = data_path / experiment
        json_files = [f.name for f in experiment_path.glob("*.json")]

        experiments_dfs = []
        for json in json_files:
            json_path = experiment_path / json
            experiments_dfs.append(load_keypoints(json_path, experiment))

        dfs.append(pd.concat(experiments_dfs, ignore_index=True))

    df = pd.concat(dfs, ignore_index=True)

    print(df["ID"].unique())

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

    # Display the DataFrame
    print(Y)


if __name__ == "__main__":
    prepare_dataset()
