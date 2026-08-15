# =========================================================
# 128-bit Message Reduced-Round SipHash Dataset Generator
# =========================================================

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

MASK64 = 0xffffffffffffffff

# ---------------------------------------------------------
# Rotate left (64-bit)
# ---------------------------------------------------------
def rotl(x, b):
    return ((x << b) & MASK64) | (x >> (64 - b))

# ---------------------------------------------------------
# SipRound
# ---------------------------------------------------------
def sipround(v0, v1, v2, v3):
    v0 = (v0 + v1) & MASK64
    v2 = (v2 + v3) & MASK64
    v1 = rotl(v1, 13)
    v3 = rotl(v3, 16)
    v1 ^= v0
    v3 ^= v2
    v0 = rotl(v0, 32)

    v2 = (v2 + v1) & MASK64
    v0 = (v0 + v3) & MASK64
    v1 = rotl(v1, 17)
    v3 = rotl(v3, 21)
    v1 ^= v2
    v3 ^= v0
    v2 = rotl(v2, 32)

    return v0, v1, v2, v3

# ---------------------------------------------------------
# SipHash-c-d for 128-bit message (two 64-bit blocks)
# ---------------------------------------------------------
def siphash_c_d_128(m0, m1, k0, k1, c, d):
    v0 = 0x736f6d6570736575 ^ k0
    v1 = 0x646f72616e646f6d ^ k1
    v2 = 0x6c7967656e657261 ^ k0
    v3 = 0x7465646279746573 ^ k1

    # Absorb first block
    v3 ^= m0
    for _ in range(c):
        v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
    v0 ^= m0

    # Absorb second block
    v3 ^= m1
    for _ in range(c):
        v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
    v0 ^= m1

    # Finalization
    v2 ^= 0xff
    for _ in range(d):
        v0, v1, v2, v3 = sipround(v0, v1, v2, v3)

    return v0 ^ v1 ^ v2 ^ v3

# ---------------------------------------------------------
# Generate 128-bit delta with given Hamming weight
# ---------------------------------------------------------
def generate_delta_128(hw):
    positions = random.sample(range(128), hw)
    delta = 0
    for p in positions:
        delta |= (1 << p)
    return delta

# ---------------------------------------------------------
# Dataset generator (correct single-block injection)
# ---------------------------------------------------------
def generate_dataset(n_samples, c, d, delta, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    X, Y = [], []

    k0 = random.getrandbits(64)
    k1 = random.getrandbits(64)

    delta0 = delta & MASK64
    delta1 = (delta >> 64) & MASK64

    for _ in range(n_samples):
        m0 = random.getrandbits(64)
        m1 = random.getrandbits(64)

        # ----- real differential -----
        t1 = siphash_c_d_128(m0, m1, k0, k1, c, d)

        # Introducing the differential in the second block
        t2 = siphash_c_d_128(m0, m1 ^ delta1, k0, k1, c, d)

        diff = t1 ^ t2
        X.append([(diff >> i) & 1 for i in range(64)])
        Y.append(1)

        # ----- random differential -----
        r1 = random.getrandbits(64)
        #r2 = random.getrandbits(64)
        diff_r = t1 ^ r1
        X.append([(diff_r >> i) & 1 for i in range(64)])
        Y.append(0)

    return np.array(X, dtype=np.uint8), np.array(Y, dtype=np.uint8)

def save_dataset_to_csv(X, Y, c, d, hw, out_dir):
    # Vectorized list creation (significantly faster)
    columns = [f"bit_{i}" for i in range(64)]

    df = pd.DataFrame(X, columns=columns)
    df["label"] = Y

    filename = f"siphash_c{c}_d{d}_hw{hw}_pos1.csv"
    path = os.path.join(out_dir, filename)

    df.to_csv(path, index=False)
    return path

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    n_samples = 115536

    out_dir = "siphash_datasets_2blocks_tag"
    os.makedirs(out_dir, exist_ok=True)

    counter = 0

    for c, d in [(1, 0),(1, 1),(1, 2),(2, 0),(2, 1),(3, 0)
             ]:
            for hw in range(1, 3): # Set Hamming Weight to create input Delta

                delta = generate_delta_128(hw)

                if c>2:
                    n_samples = 181072

                X, Y = generate_dataset(n_samples//2, c, d, delta, seed=0)

                path = save_dataset_to_csv(X, Y, c, d, hw, out_dir)

                counter += 1
                print(f"Saved → {path}")
                print(f"File_No={counter}")
