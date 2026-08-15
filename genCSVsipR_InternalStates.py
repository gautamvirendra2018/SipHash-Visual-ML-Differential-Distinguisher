# =========================================================
# ML-Assisted Differential Dataset Generator for SipHash
#
# GOAL:
# -----
# Distinguish between:
#
#   REAL:
#       related-message SipHash internal differentials
#
#   RANDOM:
#       unrelated-message SipHash internal differentials
#
# FEATURES:
# ---------
#   deltav0, deltav1, deltav2, deltav3
#
# This fixes the previous flaw where random bits were used.
#
# =========================================================

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
# =========================================================
# Directory Setup for Dataset Storage
# =========================================================
# Automatically grabs the exact folder where this Python file is saved
current_dir = Path(__file__).parent.resolve()

# Combines that folder with your dataset folder name
out_dir = str(current_dir / "siphash_internal_ml_dataset")

# =========================================================
# 64-bit arithmetic
# =========================================================

MASK64 = 0xffffffffffffffff

def rotl(x, b):
    return ((x << b) & MASK64) | (x >> (64 - b))

# =========================================================
# SipRound
# =========================================================

def sipround(v0, v1, v2, v3):

    # ----- first half -----

    v0 = (v0 + v1) & MASK64
    v2 = (v2 + v3) & MASK64

    v1 = rotl(v1, 13)
    v3 = rotl(v3, 16)

    v1 ^= v0
    v3 ^= v2

    v0 = rotl(v0, 32)

    # ----- second half -----

    v2 = (v2 + v1) & MASK64
    v0 = (v0 + v3) & MASK64

    v1 = rotl(v1, 17)
    v3 = rotl(v3, 21)

    v1 ^= v2
    v3 ^= v0

    v2 = rotl(v2, 32)

    return v0, v1, v2, v3

# =========================================================
# SipHash Internal States
# =========================================================

def siphash_internal_states(m, k0, k1, c, d):

    v0 = 0x736f6d6570736575 ^ k0
    v1 = 0x646f72616e646f6d ^ k1
    v2 = 0x6c7967656e657261 ^ k0
    v3 = 0x7465646279746573 ^ k1

    # =====================================================
    # Compression
    # =====================================================

    v3 ^= m

    for _ in range(c):

        v0, v1, v2, v3 = sipround(v0, v1, v2, v3)

    v0 ^= m

    # =====================================================
    # Finalization
    # =====================================================

    v2 ^= 0xff

    for _ in range(d):

        v0, v1, v2, v3 = sipround(v0, v1, v2, v3)

    return v0, v1, v2, v3

# =========================================================
# Generate input difference Δ
# =========================================================

def generate_delta(hw):

    positions = random.sample(range(64), hw)

    delta = 0

    for p in positions:
        delta |= (1 << p)

    return delta

# =========================================================
# Convert uint64 -> 64 bits
# =========================================================

def int_to_bits(x):

    return [(x >> i) & 1 for i in range(64)]

# =========================================================
# Dataset Generator
# =========================================================

def generate_dataset(
    n_samples,
    c,
    d,
    delta,
    fixed_key=True,
    seed=None
):

    if seed is not None:

        random.seed(seed)
        np.random.seed(seed)

    X = []
    Y = []

    # =====================================================
    # Fixed key setting
    # =====================================================

    if fixed_key:

        k0 = random.getrandbits(64)
        k1 = random.getrandbits(64)

    # =====================================================
    # Generate samples
    # =====================================================

    for _ in range(n_samples):

        # -------------------------------------------------
        # Key setup
        # -------------------------------------------------

        if not fixed_key:

            k0 = random.getrandbits(64)
            k1 = random.getrandbits(64)

        # =================================================
        # REAL DIFFERENTIAL SAMPLE
        # =================================================
        #
        # related messages:
        #
        #   m2 = m1 XOR delta
        #
        # =================================================

        m1 = random.getrandbits(64)
        m2 = m1 ^ delta

        v0a, v1a, v2a, v3a = siphash_internal_states(
            m1, k0, k1, c, d
        )

        v0b, v1b, v2b, v3b = siphash_internal_states(
            m2, k0, k1, c, d
        )

        deltav0 = v0a ^ v0b
        deltav1 = v1a ^ v1b
        deltav2 = v2a ^ v2b
        deltav3 = v3a ^ v3b

        features_real = []

        features_real.extend(int_to_bits(deltav0))
        features_real.extend(int_to_bits(deltav1))
        features_real.extend(int_to_bits(deltav2))
        features_real.extend(int_to_bits(deltav3))

        X.append(features_real)

        # label = REAL
        Y.append(1)

        # =================================================
        # RANDOM SAMPLE
        # =================================================
        #
        # unrelated random messages
        #
        # =================================================

        rm1 = random.getrandbits(64)
        rm2 = random.getrandbits(64)

        rv0a, rv1a, rv2a, rv3a = siphash_internal_states(
            rm1, k0, k1, c, d
        )

        rv0b, rv1b, rv2b, rv3b = siphash_internal_states(
            rm2, k0, k1, c, d
        )

        rdeltav0 = rv0a ^ rv0b
        rdeltav1 = rv1a ^ rv1b
        rdeltav2 = rv2a ^ rv2b
        rdeltav3 = rv3a ^ rv3b

        features_random = []

        features_random.extend(int_to_bits(rdeltav0))
        features_random.extend(int_to_bits(rdeltav1))
        features_random.extend(int_to_bits(rdeltav2))
        features_random.extend(int_to_bits(rdeltav3))

        X.append(features_random)

        # label = RANDOM
        Y.append(0)

    return np.array(X, dtype=np.uint8), np.array(Y)

def save_dataset_to_csv(X, Y, c, d, hw,out_dir):

    columns = []

    for i in range(256):
        columns.append(f"bit_{i}")

    df = pd.DataFrame(X, columns=columns)
    df["label"] = Y

    filename = f"siphash_c{c}_d{d}_hw{hw}.csv"
    path = os.path.join(out_dir, filename)

    df.to_csv(path, index=False)

    return path

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    n_samples = 115536

    os.makedirs(out_dir, exist_ok=True)

    counter = 0

    # =====================================================
    # Reduced-round variants
    # =====================================================

    variants = [

        (1,0),
        (1,1),
        (1,2),
        (1,3),
        (1,4),

        (2,0),
        (2,1),
        (2,2),
        (2,3),
        (2,4),

        (3,0),
        (3,1),
        (3,2),
        (3,3),
        (3,4)
    ]

    # =====================================================
    # Generate datasets
    # =====================================================

    for c, d in variants:

        for hw in range(1, 3): # Set Hamming weight for Delta

            delta = generate_delta(hw)
            if c > 2:
                n_samples = 181072

            print("\n")
            print("=" * 70)
            print(f"Generating Dataset")
            print(f"(c,d)=({c},{d})")
            print(f"HW(delta)={hw}")
            print("=" * 70)

            X, Y = generate_dataset(n_samples=(n_samples//2), c=c, d=d, delta=delta, fixed_key=True, seed=0)

            # =================================================
            # DataFrame
            # =================================================
            path = save_dataset_to_csv(X, Y, c, d, hw, out_dir)

            counter += 1

            print(f"\nSaved:=> ", path)
            print(f"\nFile No.: {counter}")

    print("\n")
    print("=" * 70)
    print("ALL DATASETS GENERATED")
    print("=" * 70)
