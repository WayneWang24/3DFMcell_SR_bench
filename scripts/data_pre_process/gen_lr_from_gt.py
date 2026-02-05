#!/usr/bin/env python3
"""
Generate LR images (bicubic, anti-aliased) from a GT folder for x4 SR.

Usage:
  python gen_lr_from_gt.py --gt /path/to/gt --out_root /path/to/out --scale 4 --ext tif

This will create:
  /path/to/out/
    ├── gt/                  (symlink or copied GTs)
    └── LR_bicubic/X4/
"""
import argparse
from pathlib import Path
from PIL import Image
import shutil

def downsample_bicubic(im: Image.Image, scale: int) -> Image.Image:
    w, h = im.size
    nw, nh = w // scale, h // scale
    if nw < 1 or nh < 1:
        raise ValueError(f"Image too small ({w}x{h}) for scale {scale}.")
    return im.resize((nw, nh), resample=Image.BICUBIC)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT folder containing .tif/.png etc.")
    ap.add_argument("--out_root", required=True, help="Output root to place gt/ and LR_bicubic/X{scale}/")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--ext", type=str, default="tif", help="Extension to search (case-insensitive)." )
    ap.add_argument("--copy_gt", action="store_true", help="Copy GTs into out_root/gt (default: create symlinks if possible)." )
    args = ap.parse_args()

    gt_dir = Path(args.gt)
    out_root = Path(args.out_root)
    out_gt = out_root / "gt"
    out_lr = out_root / f"LR_bicubic/X{args.scale}"
    out_gt.mkdir(parents=True, exist_ok=True)
    out_lr.mkdir(parents=True, exist_ok=True)

    files = [p for p in gt_dir.rglob("*") if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]]
    if not files:
        print(f"No images found under {gt_dir}.")
        return

    for p in files:
        rel = p.relative_to(gt_dir)
        (out_gt / rel.parent).mkdir(parents=True, exist_ok=True)
        (out_lr / rel.parent).mkdir(parents=True, exist_ok=True)

        tgt_gt = out_gt / rel
        if args.copy_gt:
            shutil.copy2(p, tgt_gt)
        else:
            try:
                if tgt_gt.exists():
                    tgt_gt.unlink()
                tgt_gt.symlink_to(p.resolve())
            except Exception:
                shutil.copy2(p, tgt_gt)

        im = Image.open(p)
        try:
            im.seek(0)  # if multipage tiff, take first frame
        except Exception:
            pass

        # Convert to a stable mode for resizing
        if im.mode not in ["L", "RGB"]:
            # For simplicity, map to RGB
            im_for_resize = im.convert("RGB")
        else:
            im_for_resize = im

        lr = downsample_bicubic(im_for_resize, args.scale)
        tgt_lr = out_lr / rel
        lr.save(tgt_lr)

    print(f"Done. GT saved/linked at: {out_gt}")
    print(f"LR (bicubic X{args.scale}) saved at: {out_lr}")

if __name__ == "__main__":
    main()
