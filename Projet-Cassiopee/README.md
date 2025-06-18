# 🧠 Brain Tumor Segmentation with Mamba-based UNets (Cassiopée 2024–2025)

This repository contains the code and results of our **Cassiopée 2024–2025** project at **Télécom SudParis**, focused on **semantic segmentation of brain tumors** from multi-parametric MRI using **Mamba-based U-Net architectures**.

## Team

- Gustavo Melo Scheidt Paulino  
- Mariana Meirelles  
- Ayman Orkhis  
**Supervisor:** Nicolas Rougon (TSP / ARTEMIS)

---

## Project Overview

We explored new Mamba-based architectures for 3D medical image segmentation, including:

- **SegMamba**: Vision Mamba encoder + 3D residual U-Net decoder  
- **UMamba**: Hybrid convolutional + Mamba blocks  
- **VM-UNet 3D** *(in progress)*: Fully Mamba-based 3D architecture

These models were compared agaisn't:

- **MedNeXt** (CNN-based)
- **Swin-UNETR** (Transformer-based)

All models were trained and evaluated within the **nnU-Net** framework.

---

## Dataset

- **BraTS 2021**: 1265 multi-sequence 3D MRI exams  
- Labels: enhancing tumor, necrotic core, edema  
- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2021/)

---

## Components

- Custom Mamba-based modules integrated into `nnU-Net`
- Deep supervision for training stability
- Unified training scripts for all variants
- Basic web interface to visualize results (WIP)

---

## Metrics

- Dice Score  
- Hausdorff Distance  


