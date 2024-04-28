from pathlib import Path
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.nn import TransformerConv
import torch.optim as optim

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
        self.skip = nn.Linear(node_features, hidden_dim * num_heads, bias=False)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        print("x shape:", x.shape)  # Should be [num_nodes, node_features]
        print("edge_index shape:", edge_index.shape)  # Should be [2, num_edges]
        x = F.relu(self.conv1(x, edge_index))
        skip_out = self.skip(x)
        x = F.relu(self.transformer_conv(x, edge_index) + skip_out)
        x = F.relu(self.lin1(x))

        for layer in self.layers:
            x_residual = x
            x = F.relu(layer(x, edge_index) + x_residual)

        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


def collate_fn(batch):
    data_list = [item[0] for item in batch]  # Collect all Data objects
    print(data_list)
    print(len(data_list))
    print(type(data_list))
    print(type(data_list[0]))
    labels_list = [
        torch.tensor(item[1], dtype=torch.long) for item in batch
    ]  # Collect all labels and ensure they are tensors

    # Convert the list of Data objects into a single Batch object
    batched_data = Batch.from_data_list(data_list)

    # Convert the list of label tensors into a single tensor
    labels_tensor = torch.stack(labels_list)

    print("Batched Data:", type(batched_data), batched_data.num_nodes)
    print("Labels Tensor:", type(labels_tensor), labels_tensor.shape)

    return batched_data, labels_tensor


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batched_skeletons, labels in loader:
        batched_skeletons = batched_skeletons.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batched_skeletons)  # Pass the entire batched object
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batched_skeletons, labels in loader:
            batched_skeletons = batched_skeletons.to(device)
            labels = labels.to(device)

            outputs = model(batched_skeletons)  # Pass the entire batched object
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / (len(loader.dataset) * labels.size(1))
    return avg_loss, accuracy


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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # Update the learning rate
        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), "trained_skeleton_gnn.pth")


if __name__ == "__main__":
    train()
