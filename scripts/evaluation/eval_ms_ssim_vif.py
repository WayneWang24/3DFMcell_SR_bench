#!/usr/bin/env python3
"""
Compute MS-SSIM and VIF between prediction and GT folders using PIQ.

Usage:
  python eval_ms_ssim_vif.py --pred /path/to/pred --gt /path/to/gt --ext tif

Requires:
  pip install piq pillow
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import piq
from tqdm import tqdm

def load_img(path):
    im = Image.open(path)
    try:
        im.seek(0)
    except Exception:
        pass
    if im.mode != "RGB":
        im = im.convert("RGB")
    x = torch.from_numpy(np.array(im)).permute(2,0,1).float() / 255.0
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ext", default="tif")
    args = ap.parse_args()

    pred_dir = Path(args.pred)
    gt_dir = Path(args.gt)

    suffs = [f".{args.ext.lower()}", f".{args.ext.upper()}", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    pred_files = sorted([p for p in pred_dir.rglob("*") if p.suffix.lower() in [s.lower() for s in suffs]])
    if not pred_files:
        print(f"No prediction images found in {pred_dir}.")
        return

    ms_list, vif_list = [], []
    for p in tqdm(pred_files, desc="Evaluating"):
        rel = p.relative_to(pred_dir)
        g = gt_dir / rel
        if not g.exists():
            print(f"GT not found for {rel}, skip.")
            continue
        x = load_img(p).unsqueeze(0)  # NCHW
        y = load_img(g).unsqueeze(0)

        _,_,H,W = y.shape
        x = x[..., :H, :W]

        with torch.no_grad():
            ms = piq.multi_scale_ssim(x, y, data_range=1.0).item()
            vif = piq.vif_p(x, y, data_range=1.0).item()
        ms_list.append(ms)
        vif_list.append(vif)

    if ms_list:
        import numpy as np
        print(f"MS-SSIM: mean={np.mean(ms_list):.6f}, std={np.std(ms_list):.6f}, n={len(ms_list)}")
    if vif_list:
        import numpy as np
        print(f"VIF:     mean={np.mean(vif_list):.6f}, std={np.std(vif_list):.6f}, n={len(vif_list)}")

if __name__ == "__main__":
    main()
