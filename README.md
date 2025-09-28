-----

## Getting Started: Setup and Installation

### 1\. Download the Repository

Since this project builds directly on the 3DGS structure, clone your repository into the desired workspace:

```bash
git clone [https://github.com/zcemcui/One-step-3DGS2NeRF.git](https://github.com/zcemcui/One-step-3DGS2NeRF.git)
cd One-step-3DGS2NeRF
```

### 2\. Prepare Data

To run the code, you must first download a dataset. This pipeline is designed to work with data from the **360 V2 dataset**.

Please ensure your downloaded data is placed in the expected directory structure before proceeding to the training step.

-----

##  Usage: Training and Evaluation

### 1\. Training and Distillation (3DGS to NeRF)

Use the following command to initiate the main pipeline. This script performs two primary functions:

1.  **Trains** a 3D Gaussian Splatting (3DGS) model from scratch using the downloaded 360 V2 data.
2.  **Converts** (distills) the resulting high-fidelity 3DGS model into a compact, faster **Instant-NGP style NeRF** model (leveraging TinyCUDANn).

<!-- end list -->

```bash
python 3DGS2NeRF.py
```

### 2\. Evaluation

After the distillation process is complete, run the evaluation script to determine the quality and fidelity of the converted NeRF model:

```bash
python 3DGS2NeRF_eval.py
```

This step generates standard metrics to assess how well the distilled NeRF model retained the visual quality of the original 3DGS representation.
