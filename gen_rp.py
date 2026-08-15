import os
import glob
import numpy as np
import pandas as pd
from pyts.image import RecurrencePlot
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ─── CONFIG ─────────────────────────────────────────────
DATA_DIR = "siphash_datasets_1block_tag" # folder containing CSV files
OUTPUT_DIR = "siphash_datasets_1block_tag_rp" # folder to save the generated RP images

TEST_SIZE = 50_000
RANDOM_STATE = 42

# RP parameters
rp = RecurrencePlot(dimension=1, time_delay=1, threshold=0.1, flatten=False)
# ───────────────────────────────────────────────────────

# 🔹 Get all CSV files automatically
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

print(f"Found {len(csv_files)} CSV files")

# 🔹 Process each file
for csv_file in csv_files:
    print(f"\nProcessing: {csv_file}")

    # Extract structure:
    # siphash_train_c3_d0_hw4.csv → name[3]='c3', name[4]='d0', name[5]='hw4'
    name = os.path.basename(csv_file).replace(".csv", "").split("_")

    try:
        c, d, hw = name[1], name[2], name[3]
    except:
        print(f"[!] Skipping malformed filename: {csv_file}")
        continue

    # Load data
    df = pd.read_csv(csv_file)

    if 'label' not in df.columns:
        print(f"[!] No 'label' column in {csv_file}, skipping.")
        continue

    X = df.drop('label', axis=1).values
    y = df['label'].values.astype(int)

    print(f"Samples: {len(X)}")

    # Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 🔹 Function to generate images
    def generate_images(X_data, y_data, split):
        # Flattens directory hierarchy into 'c1_d1_hw1_train' format
        split_folder_name = f"{c}_{d}_{hw}_{split}"
        target_dir = os.path.join(OUTPUT_DIR, split_folder_name)

        for lbl in ['0', '1']:
            os.makedirs(os.path.join(target_dir, lbl), exist_ok=True)

        for idx in tqdm(range(len(X_data)), desc=f"{os.path.basename(csv_file)}-{split}"):
            ts = X_data[idx].reshape(1, -1)
            R = rp.fit_transform(ts)[0]

            img = (R * 255).astype(np.uint8)

            label = str(y_data[idx])
            # Saves exactly as OUTPUT_DIR/c1_d1_hw1_train/0/000000.png
            save_path = os.path.join(target_dir, label, f"{idx:06d}.png")

            Image.fromarray(img, mode='L').save(save_path)

    # Generate
    generate_images(X_train, y_train, "train")
    generate_images(X_test, y_test, "test")

print("\n✅ All files processed successfully!")
