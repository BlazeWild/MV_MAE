# MV-MAE: Motion Vector Masked Autoencoder for UAV Action Recognition

MV-MAE is a hierarchical video architecture designed specifically for top-down UAV (Unmanned Aerial Vehicle) perspectives. It leverages Motion Vectors (MV) and I-frames from compressed bitstreams to achieve efficient and accurate action recognition.

## 🚀 Key Features

-   **Compressed Domain Processing**: Directly uses Motion Vectors, avoiding expensive optical flow computation or heavy 3D convolutions.
-   **2-Phase Supervised Fine-Tuning (SFT)**:
    -   **Phase 1 (Warmup)**: Freezes the backbone encoders to stabilize the randomly initialized Temporal Transformer and Head.
    -   **Phase 2 (End-to-End)**: Unfreezes the top layers of the backbone for task-specific adaptation.
-   **Precision Evaluation**: Logs Accuracy, Macro-F1, Precision, and Recall. Macro metrics provide a balanced view across the 155 UAV action classes.

## 🛣️ Project Pipeline

The MV-MAE workflow consists of three major stages:

### Step 1: Codebook Generation
Before pretraining, a discrete codebook must be generated from the Motion Vector distributions to enable the masked prediction task.
```bash
python pretraining/codebook/build_codebook.py
```

### Step 2: DMV-MAE Pretraining
Pretrain the Motion Encoder using the generated codebook. This stage teaches the model to reconstruct masked motion patches.
```bash
python pretraining/pretrain.py
```

### Step 3: Supervised Fine-Tuning (SFT)
The final stage where the Context Encoder (backbone) and Motion Encoder (pretrained) are fused and fine-tuned for action recognition using the 2-phase logic.
```bash
python train.py
```

## 🛠️ Project Structure

```text
MV_MAE/
├── mv_mae/
│   ├── models/
│   │   ├── mv_mae.py      # Main architecture (Context + Motion + Temporal)
│   │   ├── encoders.py    # I-frame and MV backbones
│   │   └── temporal.py    # Temporal Transformer aggregator
│   ├── data/
│   │   ├── dataset.py     # UAVHuman Dataset loader
│   │   └── video_loader.py # GOP-based video decoding
│   └── utils/
│       ├── checkpoint.py  # Phase-aware checkpointing
│       └── optimizer.py   # Multi-LR parameter group management
├── pretraining/           # Pretraining scripts and model definitions
├── model_zoo/             # Pretrained weights (CLIP, DMVMAE)
└── train.py               # 2-Phase SFT training pipeline
```

## 📈 Supervised Fine-Tuning (SFT) Details

### 1. Requirements
Install dependencies:
```bash
pip install torch timm scikit-learn tqdm av
```

### 2. Dataset Setup
Ensure your dataset is organized under `datasets/UAVHuman_480p_mp4/`. The `train.py` script expects splits at `datasets/UAVHuman_480p_mp4/train_split.txt`.

### 3. Running SFT
The `train.py` script automatically handles the 2-phase logic:
- **Phase 1 (Warmup)**: Encoders frozen (Epochs 0-5). Stabilizes the Temporal Transformer.
- **Phase 2 (End-to-End)**: Top-2 encoder layers unfrozen (Epochs 6-50) with 10x lower learning rate.

## 📊 Evaluation
The model logs Macro-F1 scores after every validation epoch to ensure performance is not skewed by majority classes in the UAVHuman dataset.

---
*Developed for Advanced Action Recognition from UAV Perspectives.*
