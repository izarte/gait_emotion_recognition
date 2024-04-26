from pathlib import Path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    keypoints_list = np.array(
        [item["keypoints"] for item in data if "keypoints" in item]
    )
    return {"id": id, "keypoints": keypoints_list}


def get_max_min_avg(data):
    distribution = []
    for d in data:
        points = len(d["keypoints"])
        distribution.append(points)

    distribution = np.array(distribution)

    max_value = np.max(distribution)
    min_value = np.min(distribution)
    avg = np.mean(distribution)
    min_crop = np.percentile(distribution, 25)
    max_crop = np.percentile(distribution, 85)
    print(
        "percenteiles: ",
        min_crop,
        np.percentile(distribution, 50),
        max_crop,
    )
    plt.figure(figsize=(10, 6))
    sns.histplot(distribution, bins=30, kde=True, color="green")
    plt.title("Combined Histogram and Density Plot")
    plt.xlabel("Values")
    plt.ylabel("Frequency/Density")
    plt.show()

    return max_value, min_value, avg, min_crop, max_crop


def normalize_distribution(data, min_value, max_value):
    normalized_data = []
    for d in data:
        sample = d
        points = len(d["keypoints"])
        if points < min_value:
            continue
        # Move this line inside if below to just adjust data only when max value is exceeded and change new_length for max_value
        new_length = np.random.randint(min_value, max_value)
        if points > new_length:
            sample["keypoints"] = downsample_keypoints(
                d["keypoints"], points, new_length
            )
        normalized_data.append(sample)

    return normalized_data


def downsample_keypoints(keypoints, initial_size, new_size):
    keypoints = np.array(keypoints)
    steps = initial_size / new_size

    idxs = []
    acc = steps
    for i in range(initial_size):
        if acc < 2:
            idxs.append(i)
            acc += steps - 1
        else:
            acc -= 1

    new_data = keypoints[idxs]

    return new_data


def prepare_dataset():
    # Create a Path object for the base path
    dataset_path = Path(DATASET_PATH)

    # Append the labels filename to the base path
    data_path = dataset_path / DATA_PATH / "skeletons"
    labels_path = dataset_path / LABELS_PATH

    experiments = [d.name for d in data_path.iterdir() if d.is_dir()]

    data = []
    for experiment in experiments:
        print(experiment)
        experiment_path = data_path / experiment
        json_files = [f.name for f in experiment_path.glob("*.json")]

        for json in json_files:
            json_path = experiment_path / json
            data.append(load_keypoints(json_path, int(experiment)))

    max_p, min_p, avg, min_crop, max_crop = get_max_min_avg(data)
    print("max: ", max_p, " min: ", min_p, "avg: ", avg)

    normalized_data = normalize_distribution(data, min_crop, max_crop)
    max_p, min_p, avg, min_crop, max_crop = get_max_min_avg(normalized_data)

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
    print(Y.columns)
    print(Y.query("ID == 3").drop(["ID"], axis=1).to_numpy()[0])


if __name__ == "__main__":
    prepare_dataset()
