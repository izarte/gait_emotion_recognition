from pathlib import Path
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.nn import TransformerConv
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from skeleton_dataloader import SkeletonDataloader


DATASET_PATH = "psimo_reduced"
LABELS_PATH = "metadata_labels_v3.csv"


class SkeletonGNN(nn.Module):
    def __init__(
        self, node_features, hidden_dim, num_classes, num_heads=4, num_layers=2
    ):
        super(SkeletonGNN, self).__init__()
        self.conv1 = GraphConv(node_features, hidden_dim)
        self.transformer_conv = TransformerConv(
            hidden_dim, hidden_dim, heads=num_heads, dropout=0.6, concat=True
        )
        self.lin1 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.skip = nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # print(f"Initial x shape: {x.shape}, edge_index shape: {edge_index.shape}")

        # First convolutional layer
        x = F.relu(self.conv1(x, edge_index))

        # Transformer convolutional layer with skip connection
        skip_out = self.skip(x.reshape(-1, x.shape[-1]))
        try:
            x = F.relu(self.transformer_conv(x, edge_index) + skip_out)
        except RuntimeError as e:
            print(f"Error in TransformerConv: {str(e)}")
            raise

        # print(f"After transformer conv, x shape: {x.shape}")

        # Linear layer after transformer convolution
        try:
            x = F.relu(self.lin1(x))
        except RuntimeError as e:
            print(f"Error in linear layer: {str(e)}")
            raise

        # print(f"After linear layer, x shape: {x.shape}")

        # Additional convolutional layers
        for layer in self.layers:
            x_residual = x
            try:
                x = F.relu(layer(x, edge_index) + x_residual)
            except RuntimeError as e:
                print(f"Error in additional layers: {str(e)}")
                raise

        # print(f"After layers, x shape: {x.shape}")
        # Global pooling
        batch = data.batch
        try:
            x = global_mean_pool(x, batch)
        except RuntimeError as e:
            print(f"Error in pooling: {str(e)}")
            raise

        # print(f"After pooling 1, x shape: {x.shape}")
        x = x.view(-1, 65, x.shape[1]).mean(1).float()
        # print(f"After pooling 2, x shape: {x.shape}")

        # Final linear layer
        try:
            x = self.fc(x).float()
        except RuntimeError as e:
            print(f"Error in final linear layer: {str(e)}")
            raise
        # print(f"After fc, final layer {x.shape}")
        return x


def collate_fn(batch):
    data_list = [skeleton for item in batch for skeleton in item[0]]

    # Ensure valid conversion to tensors and batch the data
    batches = []
    for i in range(0, len(data_list), 65):  # Assuming each batch has 65 elements
        current_batch = data_list[i : i + 65]
        batched_data = []
        for data in current_batch:
            # Convert `x` and `edge_index` to tensors and ensure consistent dtypes
            if isinstance(data.x, list) or isinstance(data.x, np.ndarray):
                data.x = torch.tensor(np.array(data.x), dtype=torch.float32)
            if isinstance(data.edge_index, list) or isinstance(
                data.edge_index, np.ndarray
            ):
                data.edge_index = torch.tensor(np.array(data.edge_index))
            # Check consistency of node features and edges
            if data.x.shape[0] != len(set(data.edge_index.flatten().tolist())):
                raise ValueError(
                    f"Inconsistent node features and edges in data: {data}"
                )

            batched_data.append(data)
        batches.append(Batch.from_data_list(batched_data))

    # Ensure we have exactly 10 batches
    # assert len(batches) == 10
    batched_data = Batch.from_data_list(batches)
    # Convert labels to a tensor
    labels_np = np.array([item[1] for item in batch])
    labels = torch.tensor(labels_np).float()

    # Check the content of `batched_data`
    # print(f"Nodes: {batches[0].x.shape}, Edges: {batches[0].edge_index.shape}")
    # print(f"Batch size in collate: {len(batched_data)}")

    return batched_data, labels


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for i, (batched_data, labels) in enumerate(train_loader):
        try:
            # Ensure data is moved to the device
            batched_data = batched_data.to(device)
            # Ensure labels are detached from any computational graph
            labels = labels.clone().detach().to(device)
        except RuntimeError as e:
            print(f"Error during batched data on batch {i}: {str(e)}")
            raise
        try:
            # Log tensor shapes
            # print(
            #     f"Batch {i} - Input data shape: {batched_data.x.shape}, Labels shape: {labels.shape}"
            # )

            # Forward pass
            outputs = model(batched_data)
            # print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
        except RuntimeError as e:
            print(f"Error during output calculation on batch {i}: {str(e)}")
            raise
        try:
            # Compute loss
            loss = criterion(outputs, labels)
            # print(f"Loss: {loss.item()}")
        except RuntimeError as e:
            print(f"Error during loss on batch {i}: {str(e)}")
            raise
        try:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        except RuntimeError as e:
            print(f"Error during training on batch {i}: {str(e)}")
            raise

    return epoch_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batched_skeletons, labels in loader:
            batched_skeletons = batched_skeletons.to(device)
            labels = labels.to(device)

            outputs = model(batched_skeletons)
            outputs = outputs.view(-1, outputs.size(-1))
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.round(outputs)

            # Calculate correct predictions
            correct = torch.sum(predicted == labels.float()).item()
            total_correct += correct

            # Collect all labels and predictions for F1 score computation later
            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())

    # Concatenate all batches for F1 score calculation
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    # Calculate F1 scores
    f1_scores = []
    for i in range(all_labels.shape[1]):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], average="macro")
        f1_scores.append(f1)

    # Average F1 scores across all outputs
    avg_f1_score = np.mean(f1_scores)

    avg_loss = total_loss / len(loader)
    accuracy = correct / (len(loader.dataset) * labels.size(1))
    return avg_loss, accuracy, avg_f1_score


def train():
    dataset_path = Path(DATASET_PATH)

    data_saved_path = dataset_path / "skeletons_data.pth"
    labels_path = dataset_path / LABELS_PATH

    data = torch.load(data_saved_path)
    dataset = SkeletonDataloader(data, labels_path)
    # loader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn
    )

    node_features = 2
    hidden_dim = 64
    num_classes = 17

    model = SkeletonGNN(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_heads=4,
        num_layers=2,
    )
    print(model)

    # Setup the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = (
        nn.CrossEntropyLoss()
    )  # Adjust the loss function based on your specific needs

    # Optional: Setup a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_accuracy, f1_score = evaluate(
                model, val_loader, criterion, device
            )

            tepoch.set_postfix(
                train_loss=train_loss,
                accuracy=100.0 * val_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_f1=f1_score,
            )
            # print(
            #     f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            # )

            # Update the learning rate
            scheduler.step()

    val_loss, val_accuracy, f1_score = evaluate(model, val_loader, criterion, device)
    print(
        f"Evaluation loss: {val_loss} evaluation accuracy: {val_accuracy} f1 score: {f1_score}"
    )
    # Save the model
    torch.save(model.state_dict(), "trained_skeleton_gnn.pth")


if __name__ == "__main__":
    train()
