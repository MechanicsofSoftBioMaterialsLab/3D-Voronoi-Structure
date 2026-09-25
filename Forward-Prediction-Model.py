# ==== PART 1: Imports and Dataset Class ====

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Custom dataset for Voronoi structures
class VoronoiDataset(Dataset):
    def __init__(self, structures_dir, labels_dir, num_samples=195):
        self.structures_dir = structures_dir
        self.labels_dir = labels_dir
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # idx is zero-based, file names are 1-based
        sample_id = idx + 1

        # --- Load structure (25 x 3 npy) ---
        structure_path = os.path.join(
            self.structures_dir, f"structure-{sample_id}.npy"
        )
        structure = np.load(structure_path).astype(np.float32)
        structure /= 50.0  # normalize coords to [0,1]

        # --- Load label (11 x 1 csv) ---
        label_path = os.path.join(
            self.labels_dir, f"RF-{sample_id}.csv"
        )
        label = np.loadtxt(label_path, delimiter=",").astype(np.float32)
        label = label.reshape(-1)  # flatten to shape (11,)

        # Convert to torch tensors
        structure = torch.tensor(structure)   # (25, 3)
        label = torch.tensor(label)           # (11,)

        return structure, label


# ==== PART 2: Train/Val Split and DataLoaders ====

from torch.utils.data import random_split

# Paths
structures_dir = "/home/melhachimi/RP/3D-Voronoi/Direct-Design/STRUCTURES"
labels_dir     = "/home/melhachimi/RP/3D-Voronoi/Direct-Design/Reaction-Forces"

# Full dataset (818 samples now, can be 1000 later)
full_dataset = VoronoiDataset(structures_dir, labels_dir, num_samples=818)

# Split into 80% train, 20% val
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# ---- Label stats from TRAIN set (mean/std per of the 11 outputs) ----
from torch.utils.data import DataLoader

def compute_label_stats(subset):
    tmp_loader = DataLoader(subset, batch_size=64, shuffle=False)
    ys = []
    for _, y in tmp_loader:
        ys.append(y)
    Y = torch.cat(ys, dim=0).float()
    mean = Y.mean(dim=0)
    std  = Y.std(dim=0).clamp_min(1e-6)
    return mean, std

y_mean, y_std = compute_label_stats(train_dataset)
y_mean = y_mean.to(device)
y_std  = y_std.to(device)

# ==== PART 3: DGCNN Model ====

import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    # x: (B, F, N)  -> features
    # return indices of k nearest neighbors
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)      # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]     # (B, N, k)
    return idx

def get_graph_feature(x, k=8):
    # x: (B, F, N)
    B, F, N = x.size()
    idx = knn(x, k=k)                                # (B, N, k)
    device = x.device

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()               # (B, N, F)
    feature = x.view(B * N, -1)[idx, :]
    feature = feature.view(B, N, k, F)
    x = x.view(B, N, 1, F).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    # output: (B, 2F, N, k)
    return feature

class DGCNN(nn.Module):
    def __init__(self, k=8, emb_dims=256, output_channels=11):
        super(DGCNN, self).__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2))

        # Fully connected head
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_channels)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.permute(0, 2, 1)  # (B, 3, N)

        x1 = self.conv1(get_graph_feature(x, k=self.k))  # (B, 64, N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]            # (B, 64, N)

        x2 = self.conv2(get_graph_feature(x1, k=self.k))
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = self.conv3(get_graph_feature(x2, k=self.k))
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x4 = self.conv4(get_graph_feature(x3, k=self.k))
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)          # (B, 512, N)
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)  # (B, 512)

        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.dropout(x)
        x = self.fc3(x)  # (B, 11)

        return x
# ==== PART 4: Training Setup ====

import torch.optim as optim

# Model
model = DGCNN(k=8, emb_dims=256, output_channels=11).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training function
def train_one_epoch(loader, model, optimizer, criterion, y_std, lambda_smooth=1e-3, jitter_sigma=0.01):
    model.train()
    total_loss = 0.0
    for structures, labels in loader:
        structures, labels = structures.to(device), labels.to(device)

        # light coordinate jitter (augmentation)
        if jitter_sigma > 0:
            structures = torch.clamp(structures + jitter_sigma * torch.randn_like(structures), 0.0, 1.0)

        optimizer.zero_grad()
        outputs = model(structures)  # (B, 11)

        # normalized MSE (equivalent to z-scoring targets)
        base = ((outputs - labels) / y_std).pow(2).mean()

        # curve smoothness penalty across the 11 steps
        smooth = (outputs[:, 1:] - outputs[:, :-1]).pow(2).mean()

        loss = base + lambda_smooth * smooth
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * structures.size(0)
    return total_loss / len(loader.dataset)

def eval_one_epoch(loader, model, y_std):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for structures, labels in loader:
            structures, labels = structures.to(device), labels.to(device)
            outputs = model(structures)
            base = ((outputs - labels) / y_std).pow(2).mean()
            total_loss += base.item() * structures.size(0)
    return total_loss / len(loader.dataset)

from math import inf
best_path = "/home/melhachimi/RP/3D-Voronoi/Direct-Design/dgcnn_best.pth"

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
best_val, wait, patience = inf, 0, 20



# Main training loop
EPOCHS = 100
# Keep track of loss curves
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(train_loader, model, optimizer, criterion, y_std)
    val_loss   = eval_one_epoch(val_loader, model, y_std)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    scheduler.step(val_loss)

    # save best
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        torch.save(model.state_dict(), best_path)
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break




# ==== PART 5: Save, Load, and Predict ====

loaded_model = DGCNN(k=8, emb_dims=256, output_channels=11).to(device)
loaded_model.load_state_dict(torch.load(best_path, map_location=device))
loaded_model.eval()
print("Loaded best checkpoint:", best_path)

def predict_one(sample_id, model):
    struct_path = os.path.join(structures_dir, f"structure-{sample_id}.npy")
    structure = np.load(struct_path).astype(np.float32)
    structure /= 50.0
    structure = torch.tensor(structure).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(structure)
    return pred.cpu().numpy().flatten()

pred_rf = predict_one(10, loaded_model)
print("Predicted Reaction Forces:", pred_rf)


# ==== PART 6: Plotting ====
import matplotlib.pyplot as plt

# 1. Plot Predicted vs True RF curve
def plot_prediction(sample_id, model):
    # Load true RF
    rf_path = os.path.join(labels_dir, f"RF-{sample_id}.csv")
    true_rf = np.loadtxt(rf_path, delimiter=",").astype(np.float32).flatten()

    # Predict
    pred_rf = predict_one(sample_id, model)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(true_rf, "o-", label="True", linewidth=2)
    plt.plot(pred_rf, "s--", label="Predicted", linewidth=2)
    plt.xlabel("Displacement Step")
    plt.ylabel("Reaction Force")
    plt.title(f"Structure-{sample_id}: True vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()


# 2. Plot Loss vs Epochs
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==== PART 7: Save Plots ====

# 1. Save Predicted vs True RF curve
def save_prediction_plot(sample_id, model, save_dir="/home/melhachimi/RP/3D-Voronoi/Direct-Design"):
    rf_path = os.path.join(labels_dir, f"RF-{sample_id}.csv")
    true_rf = np.loadtxt(rf_path, delimiter=",").astype(np.float32).flatten()
    pred_rf = predict_one(sample_id, model)

    plt.figure(figsize=(6, 4))
    plt.plot(true_rf, "o-", label="True", linewidth=2)
    plt.plot(pred_rf, "s--", label="Predicted", linewidth=2)
    plt.xlabel("Displacement Step")
    plt.ylabel("Reaction Force")
    plt.title(f"Structure-{sample_id}: True vs Predicted")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, f"Prediction-Structure-{sample_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot at {save_path}")


# 2. Save Loss curve
def save_loss_plot(train_losses, val_losses, save_dir="/home/melhachimi/RP/3D-Voronoi/Direct-Design"):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "Loss-Curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve at {save_path}")
    
# Save the loss curve PNG
save_loss_plot(train_losses, val_losses, save_dir="/home/melhachimi/RP/3D-Voronoi/Direct-Design")

# Save a prediction-vs-true PNG for, say, structure #10
save_prediction_plot(10, loaded_model, save_dir="/home/melhachimi/RP/3D-Voronoi/Direct-Design")

