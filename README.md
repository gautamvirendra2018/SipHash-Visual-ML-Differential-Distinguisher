# SipHash Visual ML Differential Distinguisher

This repository contains the Python implementation and experimental scripts accompanying the research paper:

Evaluating the Trust and Provenance Guarantees of Multimedia Authentication Codes: a Visualization-Driven Deep Cryptanalytic Assessment of SipHash

The work investigates visualization-driven, machine-learning-assisted differential distinguishers for reduced-round SipHash. It studies both ''intermediate internal-state differences'' and ''final authentication-tag differences'', using recurrence plots as visual representations of cryptographic differential information.

## Associated Publication

## Authors
Virendra Kumar Gautam, Dipesh Makwane, Anurag Dutta, and Rajat Subhra Chakraborty.

In Impact-aware Multimodal Persuasive Analysis and Contextual Trust (IMPACT '26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, 9 pages.

# DOI
DOI: https://doi.org/10.1145/3841453.3841498

The paper reports the methodology, experimental configuration, and results reproduced/extended by the scripts in this repository.

# Overview

SipHash is an ARX-based keyed hash/MAC construction operating on 64-bit message blocks and maintaining a 256-bit internal state composed of four 64-bit words. An instance is denoted SipHash-c-d, where c is the number of compression rounds and d is the number of finalization rounds.

This project evaluates whether machine-learning models can distinguish structured differential pairs from random pairs in reduced-round SipHash before complete diffusion is achieved.

The framework has two main phases:

Dataset generation: construct balanced differential datasets from SipHash intermediate states or authentication tags.

Visualization and learning: transform differential vectors into recurrence-plot images and train machine-learning/deep-learning classifiers as distinguishers.

This is a differential distinguisher/security-evaluation framework, not a complete key-recovery attack or forgery-generation system.

# Methodology

For a fixed input difference Δ, message pairs are generated under two conditions:

Random pair: randomly sampled message difference, label y = 0.

Fixed-differential pair: prescribed message difference Δ, label y = 1.

For authentication tags T and T', the tag-difference vector is computed as:

x = T XOR T'

For intermediate-state experiments, corresponding differences between SipHash internal states are generated in the same manner.

The resulting differential vectors are converted into binary recurrence plots. The paper uses:

Embedding dimension (m) = 1
Time delay (τ)         = 1
Threshold (ε)          = 0.1

The recurrence matrix is used as a single-channel image representation. Training and testing datasets are kept disjoint for each round to avoid data leakage.

Reduced-Round SipHash Variants

The experiments investigate reduced-round configurations including:

SipHash-1-0
SipHash-1-1
SipHash-1-2
SipHash-2-0
SipHash-2-1
SipHash-2-2
SipHash-3-0

The paper identifies reliable ML-assisted distinguishers for:

(1,0), (1,1), (1,2), (2,0), (2,1), (3,0)

while SipHash-2-2 approaches random-guessing performance in the reported experiments.

# Dataset Generation

## 1-block and 2-block tag datasets

genCSVsiphash_1blc.py
genCSVsiphash_2blc.py

These scripts generate datasets based on authentication-tag differences for 1-block and 2-block message inputs.

One of our small sized dataset associated to 1-block input message can be downlaoded from https://zenodo.org/records/21946217,
which further can be utilized to generate its associated recurrent plots as input dataset to Machine Learning and Deep Learning Models.

## Intermediate-state datasets

genCSVsipR_InternalStates.py

This script generates differential datasets from SipHash intermediate internal states.

The paper reports balanced datasets with separate training and testing sets. For the reported experiments, the 1-block tag setting used 2^16 training samples and 50,000 testing samples, while the 2-block tag and intermediate-state settings used 2^17 training samples and 50,000 testing samples.

# Recurrence Plot Generation

gen_rp.py

This script converts generated differential vectors into recurrence-plot images for subsequent ML/deep-learning experiments.

The recurrence representation is intended to expose statistical structure associated with incomplete diffusion. In the paper's interpretation, longer diagonal structures can indicate weaker diffusion/local correlation, whereas increasingly sparse and irregular structures are associated with stronger mixing.

# Machine-Learning Models

The repository contains implementations of the following classical ML models:

DecisionTree.py
RandomForest.py
kNearestNeighbor.py
LogisticRegression.py
svmachines.py
MLP.py
XGBoost.py
LightGBM.py

The paper's principal comparative evaluation focuses on MLP, XGBoost, and LightGBM, alongside the deep-learning models below. The additional classical ML implementations are provided as experimental baselines.

# Deep-Learning Models

## ResNet

resnet18.py
resnet34.py
resnet50.py
resnet101.py
resnet152.py

## DenseNet

densenet121.py
densenet169.py
densenet201.py
densenet264.py

The paper evaluates recurrence plots as single-channel binary inputs to ResNet and DenseNet architectures.

# Experimental Configuration

The reported experiments were implemented in Python using PyTorch. The paper reports:

10 training epochs

Adam optimizer

Learning rate: 1 × 10^-2

DenseNet batch size: 16

ResNet batch size: 32

Separate training/test datasets for each round

Evaluation using accuracy and F1-score

The reported experiments were conducted on a server with two Intel Xeon Gold 5218R processors and two NVIDIA Tesla T4 GPUs with 16 GiB memory each.

Note: Hardware and training settings above describe the experimental configuration reported in the paper. They are not a requirement for using the scripts on another system. GPU availability and memory requirements depend on the selected model and dataset size.

Repository Structure

SipHash-Visual-ML-Differential-Distinguisher/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
│── genCSVsiphash_1blc.py
│── genCSVsiphash_2blc.py
│── genCSVsipR_InternalStates.py
│
│── gen_rp.py
│
│── DecisionTree.py
│── RandomForest.py
│── kNearestNeighbor.py
│── LogisticRegression.py
│── svmachines.py
│── MLP.py
│── XGBoost.py
│── LightGBM.py
│
├── resnet18.py
├── resnet34.py
├── resnet50.py
├── resnet101.py
├── resnet152.py
├── densenet121.py
├── densenet169.py
├── densenet201.py
└── densenet264.py

The files are kept in the repository root in the current release; the grouping above describes their functional roles.

Requirements

# The Python dependencies used by the source files are listed in:

requirements.txt

# Install the dependencies with:

pip install -r requirements.txt

For GPU-based experiments, install a PyTorch/torchvision build compatible with the CUDA version and GPU available on the target system. The generic dependency list should not be interpreted as fixing a particular CUDA build.

# General Experimental Workflow

A typical workflow is:

1. Generate SipHash differential datasets
              ↓
2. Generate recurrence-plot representations
              ↓
3. Train ML/deep-learning classifiers
              ↓
4. Evaluate accuracy and F1-score
              ↓
5. Compare performance across SipHash-c-d variants and rounds

Generated datasets, recurrence-plot images, trained model checkpoints, and experiment result text files are intentionally excluded from version control through .gitignore.

Results Summary

The paper reports that:

Very low-round configurations such as SipHash-1-0, SipHash-1-1, and SipHash-2-0 can exhibit near-perfect classification performance in the evaluated settings.

Classification performance generally decreases as additional compression/finalization rounds increase diffusion.

SipHash-2-2 approaches random-guessing accuracy in the reported 1-block tag and intermediate-state experiments.

The identified ML-assisted distinguishers for the reported configurations are (1,0), (1,1), (1,2), (2,0), (2,1), and (3,0).

For 2-block messages, the paper reports that differences introduced into the first message block can be largely masked by diffusion, whereas differences introduced later can retain more distinguishable structure in some reduced-round configurations.

These results concern the reduced-round variants evaluated in the paper and should not be interpreted as a cryptanalytic break of the standard full-round SipHash deployment configuration.

Reproducibility and Data Policy

The repository provides the source code required for the experimental pipeline. Large generated datasets, recurrence-plot image collections, trained model checkpoints, and raw experiment result files are not included in the Git repository.

The .gitignore file excludes generated artifacts such as:

siphash_datasets_1block_tag/
siphash_datasets_2blocks_tag/
siphash_internal_ml_dataset/
siphash_datasets_1block_tag_rp/
siphash_datasets_2blocks_tag_rp/
siphash_internal_ml_dataset_rp/
result/
*.pt
*.pth
*.ckpt

This keeps the source repository lightweight while allowing the datasets and results to be regenerated from the provided scripts.

Scope and Responsible Use

The implementation is intended for academic research, reproducibility, and security evaluation of reduced-round cryptographic constructions. The reported distinguisher should be understood as evidence of statistical distinguishability under the controlled experimental conditions described in the paper; it is not, by itself, a practical forgery or key-recovery attack against full-round SipHash.

Citation

If you use this repository or build upon the implementation, please cite:

@inproceedings{gautam2026siphash,
  author    = {Virendra Kumar Gautam and Dipesh Makwane and Anurag Dutta and Rajat Subhra Chakraborty},
  title     = {Evaluating the Trust and Provenance Guarantees of Multimedia Authentication Codes: a Visualization-Driven Deep Cryptanalytic Assessment of SipHash},
  booktitle = {Impact-aware Multimodal Persuasive Analysis and Contextual Trust (IMPACT '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3841453.3841498}
}

License

This repository is released under the license specified in LICENSE.

The associated paper states that the published work is licensed under a Creative Commons Attribution 4.0 International License.
