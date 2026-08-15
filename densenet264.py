import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import trange
from collections import OrderedDict

# -------------------
# DenseNet-264
# -------------------
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        num_features = num_input_features
        for i in range(num_layers):
            layer = _DenseLayer(num_features, growth_rate, bn_size, drop_rate)
            layers.append(layer)
            num_features += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x

class DenseNet264(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, growth_rate=32,
                 block_config=(6, 12, 64, 48), bn_size=4, compression=0.5, drop_rate=0.0):
        super().__init__()

        # Initial convolution
        num_init_features = 64
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks and transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = _Transition(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.relu_final = nn.ReLU(inplace=True)

        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

        # He init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.relu_final(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
        transforms.Resize((224, 224)),  # standard input size
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    # -------------------------------------------------
    # Use only a subset of the dataset
    # -------------------------------------------------
    #train_data = stratified_subset(train_data, total_samples=20000)
    #test_data  = stratified_subset(test_data, total_samples=5000)

    #print(f"Training Images : {len(train_data)}")
    #print(f"Testing Images  : {len(test_data)}")


    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


    model = DenseNet264(in_channels=3, num_classes=2).to(device)
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
result_file = os.path.join("result", "result_264.txt")

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
