import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import OrderedDict

# -------------------
# ResNet-34
# -------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def stratified_subset(dataset, total_samples, random_state=42):
    """
    Create a stratified random subset from an ImageFolder dataset.
    """
    labels = np.array(dataset.targets)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=total_samples,
        random_state=random_state
    )

    subset_indices, _ = next(
        splitter.split(np.zeros(len(labels)), labels)
    )

    return Subset(dataset, subset_indices)

# -------------------
# Training function
# -------------------
def train_one_round(train_dir, test_dir, device):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Use only a subset of the dataset
    # -------------------------------------------------
    #train_data = stratified_subset(train_data, total_samples=20000)
    #test_data  = stratified_subset(test_data, total_samples=5000)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=2).to(device)  # ResNet-34
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    start_train = time.time()
    model.train()
    for _ in range(10):  # 10 epochs
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    end_train = time.time()
    train_time = end_train - start_train

    # Test
    start_test = time.time()
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    end_test = time.time()
    test_time = end_test - start_test

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # True Positive Rate (Sensitivity / Recall)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # True Negative Rate (Specificity)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return acc, prec, rec, f1, tpr, tnr, train_time, test_time

# -------------------
# Main Loop
# -------------------
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "siphash_datasets_2blocks_tag_rp" # folder containing RP images

os.makedirs("result", exist_ok=True)
result_file = os.path.join("result", "result_rs34.txt")

# Match your exact pipeline configurations
configurations = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]
hamming_weights = range(1, 6)  # 1 to 5

round_dirs = []
for c, d in configurations:
    for hw in hamming_weights:
        # Construct path fragments matching your folders
        train_path = os.path.join(BASE_DIR, f"c{c}_d{d}_hw{hw}_train")
        test_path = os.path.join(BASE_DIR, f"c{c}_d{d}_hw{hw}_test")

        # Verify the folders exist before adding them to your list
        if os.path.exists(train_path) and os.path.exists(test_path):
            # Normalize slashes to forward slashes if needed for your ML pipeline
            train_path_clean = train_path.replace("\\", "/")
            test_path_clean = test_path.replace("\\", "/")

            round_dirs.append((train_path_clean, test_path_clean))

with open(result_file, "w") as f:
    # Header
    f.write("Round\tAccuracy\tPrecision\tRecall\tF1\tTPR\tTNR\tTrain_Time(s)\tTest_Time(s)\n")

    for i in trange(len(round_dirs), desc="Rounds", ncols=100):
        train_dir, test_dir = round_dirs[i]
        acc, prec, rec, f1, tpr, tnr, train_time, test_time = train_one_round(train_dir, test_dir, device)
        f.write(f"{i+1}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{tpr:.4f}\t{tnr:.4f}\t{train_time:.2f}\t{test_time:.2f}\n")
        f.flush()
    f.close()