# 3DFMcell_SR_bench

This repository contains the code, configurations, and scripts related to my ongoing research on **3D Fluorescence Microscopy De-noising and Super-Resolution**.  
The project is part of a paper currently in preparation, focusing on image restoration and analysis for high-resolution biological microscopy data.

---

## 📖 Project Overview
Fluorescence microscopy plays a vital role in studying cellular structures and dynamics.  
However, challenges such as:
- **low resolution** in the axial direction,  
- **noise interference**, and  
- **limited data acquisition**  

make accurate 3D reconstruction and quantitative analysis difficult.

This repository provides:
- Benchmark code for 3D fluorescence microscopy super-resolution and reconstruction tasks.
- Experiment configurations (`configs/`) and training scripts (`scripts/`).
- Preliminary implementations for data preprocessing, training, and evaluation.

---

## 🧪 Research Goals
- Develop robust **super-resolution methods** for 3D fluorescence microscopy (FM).
- Benchmark algorithms under realistic conditions (noise, limited samples).
- Provide reproducible experiments for future biological image analysis.

---

## 📂 Repository Structure
```
3DFMcell_SR_bench/
│
├── configs/ # Experiment configuration files
├── scripts/ # Training, testing, and evaluation scripts
├── mmagic/ # Submodule: OpenMMLab MMagic framework
├── data/ # (Not included) Placeholder for microscopy datasets
└── README.md # Project documentation
```

---

## ⚙️ Requirements
- Python 3.9+
- PyTorch >= 1.12
- [MMagic](https://github.com/open-mmlab/mmagic) (included as submodule)
- CUDA-compatible GPU (recommended)

Install dependencies:
```bash
pip install -r requirements.txt


## 🚀 Usage
1. Clone the repository
git clone --recursive https://github.com/<your-username>/3DFMcell_SR_bench.git
cd 3DFMcell_SR_bench


## 📊 Dataset

The live-cell 3D fluorescence microscopy datasets used in this project are **not publicly available for direct download** due to data sensitivity and usage restrictions.  

- To access the datasets, researchers are required to **submit an application form**.  
- After review and approval, a secure download link will be provided via email.  

📌 Please contact the maintainer for application details and data usage agreements.


## 👥 Contributors

- **Chenwei Wang**  
  Lead author and maintainer. Responsible for project design, methodology development (VTCD framework), and code implementation.  

- **Zhaoke Huang**  
  Co-author. Contributed to methodology discussion, experimental validation, and manuscript preparation.  

- **Zelin Li**  
  Co-author. Contributed to dataset preparation, experimental analysis, and result verification.  

- **Prof. Hong Yan**  
  Supervisor. Provided overall guidance, research supervision, and critical manuscript revision.  

---

💡 *If you are interested in contributing (e.g., extending models, improving code, or benchmarking on new datasets), please feel free to open an issue or contact the maintainer.*



