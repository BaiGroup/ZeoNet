# ZeoNet: Representations for predicting adsorption in nanoporous zeolites

## Introduction

This is the official code repository for the paper [Representation Learning for Long-Chain Hydrocarbon Adsorption in Zeolites](https://www.researchsquare.com/article/rs-6689839/v1) and [ZeoNet: 3D convolutional neural networks for predicting adsorption in nanoporous zeolites](https://pubs.rsc.org/en/content/articlelanding/2023/ta/d3ta01911j).

ZeoNet is a flexible and modular tool for predicting porous material (e.g. zeolites) adsorption properties using various representations, including ConvNets with 3D volumetric grids and 2D multi-view images, Vision Transformers with 3D volumetric grids, PointNet and EdgeConv with pointclouds of atomic coordinates and solvent-accessible surface, and graph-based neural networks (CGCNN, MEGNet, M3GNet, and MACE).

## Available Models

* 3D CNNs: AlexNet, VGG, ResNet, DenseNet (various depths)
* 2D CNNs: Multi-view ResNet (18, 50)
* ViTs
* Pointclouds: PointNet, EdgeConv
* GNNs: CGCNN, MEGNet, M3GNet, MACE

## Project Structure

1. Codebase

```
zeonet/
├── models/           # Model architectures (e.g., CNNs, GNNs, ViTs)
├── trainer/          # Training and evaluation logic (e.g., Trainer classes)
├── utils/            # Utility functions (e.g., config loading, data processing)
├── configs/          # YAML configuration files for models, datasets, and training
├── datasets/         # Custom PyTorch Dataset classes and data loading logic
└── train.py          # Main entry point for training ZeoNet  
```

2. Dataset

For each zeolite structure, the `.cif` file contains crystallographic information such as lattice parameters and atomic positions. The `.h5py` file contains distance grids (a 3D array of shape [x, y, z] in which each grid point is assigned the distance to the surface of its nearest atom) to encode porous structural features, computed by [Zeo++](https://www.zeoplusplus.org/examples.html#grid).

There are two datasets of zeolite structures used in this work, IZASC and PCOD. [IZASC](https://www.iza-structure.org/databases/) consists of experimentally validated zeolite structures approved by the Structure Commission of the International Zeolite Association, while [PCOD](https://pubs.rsc.org/en/content/articlelanding/2011/cp/c0cp02255a) contains computationally predicted zeolite-like materials publicly available in the Predicted Crystallography Open Database.

```
ML-Zeolites/
├── CIFs/                     
│   ├── IZASC/               # CIF files of experimentally validated zeolite structures from the IZA database
│   └── PCOD/                # CIF files of hypothetical zeolite structures from the PCOD database
├── distance-grids-h5/         
│   ├── IZASC/               # Preprocessed 3D distance grid data (in HDF5 format) for IZASC structures
│   └── PCOD/                # Preprocessed 3D distance grid data (in HDF5 format) for PCOD structures
└── C18-adsorption/         
    ├── each-zeolite-info.csv   # Metadata and adsorption properties (e.g., Henry constants) for each zeolite
    ├── train_set.txt           # List of training structure IDs
    ├── val_set.txt             # List of validation structure IDs
    ├── test_set.txt            # List of test structure IDs
    └── atom_init.json          # Atom embedding initialization for graph-based models
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
conda env create -f environment-linux64-cuda121.yml
```
Note: The provided environment file is for Linux (64-bit) systems with CUDA 12.1. Ensure that the DGL version you install is compatible with your PyTorch and CUDA versions. Please refer to the [official DGL installation guide](https://www.dgl.ai/pages/start.html) for proper installation instructions.

## Usage

1. Organize your dataset following the structure described above. `ML-Zeolites` is a small toy dataset we include to help you get started.
2. Configure your run: set dataset path, representation type, model architecture, and training hyperparameters in a YAML file under the configs/ directory.
3. Run training:
```bash
python train.py --config configs/your_config.yaml
```
