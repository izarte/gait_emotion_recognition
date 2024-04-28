from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import Data, DataLoader
from skeleton_dataloader import SkeletonDataloader


DATASET_PATH = "psimo_reduced"
LABELS_PATH = "metadata_labels_v3.csv"
DATA_PATH = "semantic_data"


def convert_to_points(flat_list):
    # Get consecutive points x y and remove third element (confidence)
    return [flat_list[i : i + 2] for i in range(0, len(flat_list), 3)]


def calculate_skeleton(skeleton, verbose):
    points = convert_to_points(skeleton)
    x = np.array(
        points
    )  # Convert list of points into a numpy array for easier indexing

    # Define connections
    edges = (
        torch.tensor(
            [
                # Face
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                # Neck to shoulders
                [0, 5],
                [0, 6],
                # Shoulders to arms
                [5, 7],
                [7, 9],
                [6, 8],
                [8, 10],
                # Shoulders to body
                [5, 11],
                [6, 12],
                # Body (spine)
                [11, 12],
                # Hips to legs
                [11, 13],
                [13, 15],
                [12, 14],
                [14, 16],
            ],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot nodes
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="blue", label="Joints")
        ax.scatter(x[5, 0], x[5, 1], x[5, 2], c="red", label="Joints")
        ax.scatter(x[6, 0], x[6, 1], x[6, 2], c="red", label="Joints")

        # Plot edges based on defined connections
        for start, end in edges:
            if start < len(x) and end < len(x):  # Check to avoid index error
                ax.plot(
                    [x[start, 0], x[end, 0]],
                    [x[start, 1], x[end, 1]],
                    [x[start, 2], x[end, 2]],
                    "k-",
                    lw=2,
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.title("Skeleton Visualization")
        plt.grid(True)
        plt.show()

    data = Data(x=x, edge_index=edges)
    return data


def load_keypoints(json_file_path, id):
    # Open and read the JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)
    # Extract 'keypoints' from each dictionary in the list
    keypoints_list = np.array([d["keypoints"] for d in data])
    return {"id": id, "keypoints": keypoints_list}


def get_max_min_avg(data, verbose):
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
    if verbose:
        plt.figure(figsize=(10, 6))
        sns.histplot(distribution, bins=30, kde=True, color="green")
        plt.title("Combined Histogram and Density Plot")
        plt.xlabel("Values")
        plt.ylabel("Frequency/Density")
        plt.show()

    return max_value, min_value, avg, min_crop, max_crop


def normalize_distribution(data, min_value, max_value):
    normalized_size = []
    for d in data:
        sample = d
        points = len(d["keypoints"])
        if points < min_value:
            continue
        # Move this line inside if below to just adjust data only when max value is exceeded and change new_length for max_value
        new_length = np.random.randint(min_value, max_value)
        new_length = min_value
        if points > new_length:
            sample["keypoints"] = downsample_keypoints(
                d["keypoints"], points, new_length
            )
        normalized_size.append(sample)

    return normalized_size


def center_skeleton(skeleton, root_index=0):
    root = skeleton.x[root_index]
    skeleton.x -= root
    return skeleton


def scale_skeleton(skeleton, desired_distance=1):
    # Use shoulder to normalize distance
    current_distance = np.linalg.norm(skeleton.x[5] - skeleton.x[6])
    scale_factor = desired_distance / current_distance if current_distance != 0 else 1
    skeleton.x *= scale_factor
    return skeleton


def get_normalize_data_skeletons(data, verbose):
    skeletons = []
    for d in data:
        instance_skeletons = []
        for instance in d["keypoints"]:
            skeleton = calculate_skeleton(instance, verbose)
            centered_skeleton = center_skeleton(skeleton=skeleton)
            scaled_skeleton = scale_skeleton(centered_skeleton)
            instance_skeletons.append(scaled_skeleton)
        skeletons.append({"id": d["id"], "skeletons": instance_skeletons})

    return skeletons


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


def prepare_dataset(verbose):
    # Create a Path object for the base path
    dataset_path = Path(DATASET_PATH)

    # Append the labels filename to the base path
    data_path = dataset_path / DATA_PATH / "skeletons"
    labels_path = dataset_path / LABELS_PATH

    experiments = [d.name for d in data_path.iterdir() if d.is_dir()]

    data = []
    for experiment in experiments:
        experiment_path = data_path / experiment
        json_files = [f.name for f in experiment_path.glob("*.json")]

        for json_file in json_files:
            json_path = experiment_path / json_file
            data.append(load_keypoints(json_path, int(experiment)))

    max_p, min_p, avg, min_crop, max_crop = get_max_min_avg(data, verbose)
    print(
        "max: ",
        max_p,
        " min: ",
        min_p,
        "avg: ",
        avg,
        "percentil 25%: ",
        min_crop,
        "percentil 85%: ",
        max_crop,
    )

    normalized_size_data = normalize_distribution(data, min_crop, max_crop)
    normalized_skeletons = get_normalize_data_skeletons(normalized_size_data, verbose)

    dataloader_path = dataset_path / "skeletons_data.pth"
    print("Saving dataloader to: ", dataloader_path)
    dataset = SkeletonDataloader(data=normalized_skeletons, labels_path=labels_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    torch.save(normalized_skeletons, dataloader_path)


if __name__ == "__main__":
    prepare_dataset(verbose=False)
