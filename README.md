# 🚀 One-Step 3DGS-to-NeRF Distillation and Evaluation Pipeline

This repository extends the original **3D Gaussian Splatting (3DGS)** framework by introducing a specialized pipeline. This pipeline's goal is convert a high-quality 3DGS model into a smaller, faster **Instant-NGP-style NeRF model** (using TinyCUDANn) and subsequently **evaluate** the distilled NeRF's fidelity.

---

## 📥 Getting Started: Setup and Installation

### 1. Download the Repository

Since this project builds directly on the 3DGS structure, clone your repository into the desired workspace:

```bash
git clone [https://github.com/zcemcui/One-step-3DGS2NeRF.git](https://github.com/zcemcui/One-step-3DGS2NeRF.git)
cd One-step-3DGS2NeRF
