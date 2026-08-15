import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier

# -------------------
# Helper: flatten images for sklearn
# -------------------
def prepare_data(data_loader):
    X, y = [], []
    for images, labels in data_loader:
        images = images.view(images.size(0), -1)  # flatten
        X.extend(images.numpy())
        y.extend(labels.numpy())
    return X, y

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
def train_one_round(train_dir, test_dir):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # standard input size
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    # -------------------------------------------------
    # Use only a subset of the dataset
    # -------------------------------------------------
    #train_data = stratified_subset(train_data, total_samples=65536)
    #test_data  = stratified_subset(test_data, total_samples=21846)

    print(f"Training Images : {len(train_data)}")
    print(f"Testing Images  : {len(test_data)}")


    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Prepare flat data
    X_train, y_train = prepare_data(train_loader)
    X_test, y_test = prepare_data(test_loader)

   # Define K-Nearest Neighbors Classifier for Ascon Image Profiles
    model = KNeighborsClassifier(
    n_neighbors=5,          # Odd number prevents ties in binary classification (0 or 1)
    weights="distance",     # Closer structural images influence the vote more heavily
    metric="manhattan",     # L1 distance performs better than Euclidean in high dimensions
    algorithm="auto",       # Lets sklearn choose between ball_tree or brute force
    n_jobs=-1)               # Vital: utilizes all CPU cores for the 102,400 feature array


    # Train
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train

    # Test
    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()
    test_time = end_test - start_test

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
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
result_file = os.path.join("result", "result_knn.txt")

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
        acc, prec, rec, f1, tpr, tnr, train_time, test_time = train_one_round(train_dir, test_dir)
        f.write(f"{i+1}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{tpr:.4f}\t{tnr:.4f}\t{train_time:.2f}\t{test_time:.2f}\n")
        f.flush()
    f.close()