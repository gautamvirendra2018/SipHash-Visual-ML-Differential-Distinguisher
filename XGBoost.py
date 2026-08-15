import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import trange
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore', message=".*use_label_encoder.*")

# -------------------
# Helper: flatten images
# -------------------
def prepare_data(data_loader):
    X, y = [], []
    for images, labels in data_loader:
        images = images.view(images.size(0), -1)  # flatten into 1D vector
        X.extend(images.numpy())
        y.extend(labels.numpy())
    return X, y

# -------------------
# Training function (XGBoost)
# -------------------
def train_one_round(train_dir, test_dir):
    # Use raw tensors (no resize/normalize here)
    transform = transforms.ToTensor()

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Prepare tabular-like data
    X_train, y_train = prepare_data(train_loader)
    X_test, y_test = prepare_data(test_loader)

    # Define XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

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

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return acc, prec, rec, f1, tpr, tnr, train_time, test_time
# -------------------
# Main Loop
# -------------------
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "siphash_datasets_2blocks_tag_rp" # folder containing RP images

os.makedirs("result", exist_ok=True)
result_file = os.path.join("result", "result_xgb.txt")

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

with open(result_file, "w", encoding="utf-8") as f:
    if not os.path.exists("results_xgb.txt") or os.path.getsize("results_xgb.txt") == 0:
        f.write("Round\tAccuracy\tPrecision\tRecall\tF1\tTPR\tTNR\tTrain_Time(s)\tTest_Time(s)\n")

    for i in trange(len(round_dirs), desc="Rounds", ncols=100):
        train_dir, test_dir = round_dirs[i]
        acc, prec, rec, f1, tpr, tnr, train_time, test_time = train_one_round(train_dir, test_dir)
        f.write(f"{i + 1}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{tpr:.4f}\t{tnr:.4f}\t{train_time:.2f}\t{test_time:.2f}\n")
        f.flush()
    f.close()