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

### 1. Clone the repository
```bash
git clone --recursive https://github.com/<your-username>/3DFMcell_SR_bench.git
cd 3DFMcell_SR_bench
```

### 2. Data Preprocessing

#### From NIfTI volumes (.nii.gz)
```bash
python scripts/data_pre_process/make_mmagic_dataset_from_nii.py \
  --input_dir /path/to/nii_volumes \
  --out_root ./data/cells_dataset \
  --normalize p1p99 \
  --xy_scales 4 \
  --yzxz_scales 2 4 8
```

#### Convert to SeeSR format
```bash
python scripts/data_pre_process/prepare_cells_for_seesr.py \
  --src-root /path/to/cells_xy_p1p99/train \
  --dest-root /path/to/seesr_dataset \
  --scales 2,4,8 --tile 512 --stride 512 --norm p1p99 --to-uint8
```

### 3. Training
```bash
python mmagic/tools/train.py configs/edsr_x_cells_xy_template.py \
  --work-dir results/work_dirs/edsr_x4_cells_xy \
  --cfg-options experiment_name=edsr_x4_cells_xy \
                data_root=/path/to/cells_xy_p1p99 \
                scale=4
```

### 4. Batch Evaluation

Evaluate multiple models on multiple datasets with comprehensive metrics.

#### Install evaluation dependencies
```bash
pip install lpips pytorch-msssim pyiqa pytorch-fid pandas
```

#### Metrics
| Metric | Type | Direction |
|--------|------|-----------|
| PSNR | Full-reference | ↑ Higher is better |
| SSIM | Full-reference | ↑ Higher is better |
| MS-SSIM | Full-reference | ↑ Higher is better |
| LPIPS | Full-reference | ↓ Lower is better |
| VIF | Full-reference | ↑ Higher is better |
| NIQE | No-reference | ↓ Lower is better |
| PIQE | No-reference | ↓ Lower is better |
| NRQM | No-reference | ↑ Higher is better |
| FID | Distribution | ↓ Lower is better |

#### Run evaluation
```bash
# Single model, single dataset
python scripts/eval/batch_eval_sr.py \
  --model-dirs results/work_dirs/edsr_x4_cells_xy \
  --data-roots /path/to/cells_xy_p1p99/val \
  --scales 4 \
  --output results/eval_results.csv

# Multiple models, multiple datasets
python scripts/eval/batch_eval_sr.py \
  --model-dirs \
    results/work_dirs/edsr_x4_cells_xy \
    results/work_dirs/srcnn_x4_cells_xy \
    results/work_dirs/swinir_x4_cells_xy \
  --data-roots \
    /path/to/singleS_30_highL/cells_xy_p1p99/val \
    /path/to/fastz_200_highL/cells_xy_p1p99/val \
  --scales 2 4 8 \
  --output results/eval_all.csv

# Skip inference (use existing SR results)
python scripts/eval/batch_eval_sr.py \
  --model-dirs results/work_dirs/edsr_x4_cells_xy \
  --data-roots /path/to/val \
  --scales 4 \
  --skip-inference \
  --output results/eval_results.csv
```

#### Output format
Results are saved as CSV:
```
Model     | Dataset        | Scale | PSNR   | SSIM   | MS-SSIM | LPIPS  | VIF    | NIQE   | PIQE   | NRQM   | FID
----------|----------------|-------|--------|--------|---------|--------|--------|--------|--------|--------|-------
edsr_x4   | cells_xy_p1p99 | X4    | 32.15  | 0.9234 | 0.9567  | 0.0423 | 0.4521 | 4.2341 | 23.45  | 7.234  | 45.67
srcnn_x4  | cells_xy_p1p99 | X4    | 30.23  | 0.8912 | 0.9234  | 0.0612 | 0.3892 | 5.1234 | 28.12  | 6.891  | 52.34
```

### 5. 3D NIfTI Super-Resolution

End-to-end super-resolution on 3D NIfTI volumes. The script slices the volume along XZ or YZ planes, applies 2D SR models, and reconstructs the 3D volume.

**Input:** Original nii.gz volume (X, Y, Z)
**Output:** Z-axis super-resolved nii.gz (X, Y, Z×scale)

#### Single model
```bash
python scripts/eval/sr_3d_nifti.py \
  --input-dir /path/to/nii_volumes \
  --output-dir results/sr_3d \
  --config configs/edsr_x_cells_xy_template.py \
  --checkpoint results/work_dirs/edsr_x4_cells_xz/best_PSNR_iter_xxx.pth \
  --scale 4 \
  --slice-axis xz
```

#### Multiple models comparison
```bash
python scripts/eval/sr_3d_nifti.py \
  --input-dir /path/to/nii_volumes \
  --output-dir results/sr_3d \
  --config \
    configs/edsr_x_cells_xy_template.py \
    configs/srcnn_x_cells_xy_template.py \
    configs/swinir_x_cells_xy_template.py \
  --checkpoint \
    results/work_dirs/edsr_x4/best.pth \
    results/work_dirs/srcnn_x4/best.pth \
    results/work_dirs/swinir_x4/best.pth \
  --scale 4 \
  --slice-axis xz
```

#### Parameters
| Parameter | Description |
|-----------|-------------|
| `--input-dir` | Directory containing original .nii.gz files |
| `--output-dir` | Output root directory |
| `--config` | MMagic config file(s) |
| `--checkpoint` | Model checkpoint(s), must match configs |
| `--scale` | Super-resolution scale factor |
| `--slice-axis` | `xz` (slice along Y) or `yz` (slice along X) |
| `--prefix` | Only process files with this prefix (e.g., `memb`) |

#### Output structure
```
results/sr_3d/
├── edsr_x4/
│   ├── memb01_sr_x4.nii.gz
│   ├── memb02_sr_x4.nii.gz
│   └── ...
├── srcnn_x4/
│   └── ...
└── swinir_x4/
    └── ...
```

---

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




