import os
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import imageio.v2 as iio
from skimage.transform import resize
from tqdm import tqdm
import random

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def percent_norm(vol, low=1, high=99):
    p0, p1 = np.percentile(vol, [low, high])
    vol = np.clip(vol, p0, p1)
    denom = max(1e-6, (p1 - p0))
    vol = (vol - p0) / denom
    return vol

def to_uint16(img):
    # img in [0,1] or arbitrary range
    if img.dtype.kind == "f":
        img = np.clip(img, 0.0, 1.0)
        return (img * 65535.0 + 0.5).astype(np.uint16)
    if img.dtype == np.uint16:
        return img
    if img.dtype == np.uint8:
        return (img.astype(np.uint16) * 257)  # 8bit→16bit
    # fallback
    img = img.astype(np.float32)
    img = img - img.min()
    m = img.max()
    if m > 0:
        img = img / m
    return (img * 65535.0 + 0.5).astype(np.uint16)

def bicubic(im, out_h, out_w):
    # skimage resize: (out_h, out_w); order=3 bicubic; anti_aliasing
    return resize(
        im, (out_h, out_w),
        order=3, preserve_range=True, anti_aliasing=True
    ).astype(im.dtype)

def save_xy_sets(vol, vid, split_dir, scales):
    # vol shape: (x, y, z); XY 切片沿 z
    x, y, z = vol.shape
    hr_dir = Path(split_dir, "HR"); hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dirs = {s: Path(split_dir, f"LR_bicubic/X{s}") for s in scales}
    for p in lr_dirs.values(): p.mkdir(parents=True, exist_ok=True)

    for k in range(z):
        img = vol[:, :, k]  # (x,y)
        name = f"{vid}_z{k:04d}.png"
        iio.imwrite(hr_dir / name, to_uint16(img))
        for s, d in lr_dirs.items():
            lr = bicubic(img, img.shape[0]//s, img.shape[1]//s)  # (x/s, y/s)
            iio.imwrite(d / name, to_uint16(lr))

def save_yz_sets(vol, vid, split_dir, scales):
    # YZ 切片：对每个 x 取 (y,z)，为了让 H=Z、W=Y，做转置 -> (z,y)
    x, y, z = vol.shape
    hr_dir = Path(split_dir, "HR"); hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dirs = {s: Path(split_dir, f"LR_bicubic/X{s}") for s in scales}
    for p in lr_dirs.values(): p.mkdir(parents=True, exist_ok=True)

    for i in range(x):
        yz = vol[i, :, :]            # (y,z)
        yz = yz.T                    # -> (z,y) ：保存为 H=Z, W=Y
        name = f"{vid}_x{i:04d}.png"
        iio.imwrite(hr_dir / name, to_uint16(yz))
        for s, d in lr_dirs.items():
            lr = bicubic(yz, yz.shape[0]//s, yz.shape[1]//s)  # H 保持Z，W=Y/s
            iio.imwrite(d / name, to_uint16(lr))

def save_xz_sets(vol, vid, split_dir, scales):
    # XZ 切片：对每个 y 取 (x,z)，同理转置 -> (z,x) 保存为 H=Z, W=X
    x, y, z = vol.shape
    hr_dir = Path(split_dir, "HR"); hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dirs = {s: Path(split_dir, f"LR_bicubic/X{s}") for s in scales}
    for p in lr_dirs.values(): p.mkdir(parents=True, exist_ok=True)

    for j in range(y):
        xz = vol[:, j, :]  # (x,z)
        xz = xz.T          # -> (z,x)
        name = f"{vid}_y{j:04d}.png"
        iio.imwrite(hr_dir / name, to_uint16(xz))
        for s, d in lr_dirs.items():
            lr = bicubic(xz, xz.shape[0]//s, xz.shape[1]//s)  # H 保持Z，W=X/s
            iio.imwrite(d / name, to_uint16(lr))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="包含 memb*.nii.gz 的目录")
    ap.add_argument("--out_root", required=True, help="输出根目录（会创建 cells_* 子目录）")
    ap.add_argument("--only_prefix", default="memb", help="仅处理此前缀开头的 nii.gz")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="体级验证集比例")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--xy_scales", type=int, nargs="+", default=[4], help="XY 生成的 LR 倍数，如 4 8")
    ap.add_argument("--yzxz_scales", type=int, nargs="+", default=[4], help="YZ/XZ 生成的 LR 倍数")
    ap.add_argument("--normalize", choices=["none","p1p99","minmax"], default="none",
                    help="强度归一-none / p1p99/ minmax")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_root = Path(args.out_root)
    random.seed(args.seed)

    files = sorted([p for p in in_dir.iterdir()
                    if p.is_file()
                    and p.name.lower().startswith(args.only_prefix.lower())
                    and p.suffixes[-2:]==['.nii', '.gz']])

    assert files, f"No nii.gz with prefix '{args.only_prefix}' in {in_dir}"

    # 体级划分
    random.shuffle(files)
    n_val = max(1, int(round(len(files)*args.val_ratio)))
    val_set = set(files[:n_val])
    tr_set = set(files[n_val:])

    # 记录一下划分
    ensure_dir(out_root)
    with open(out_root / "split.json", "w") as f:
        json.dump({
            "train": [p.name for p in tr_set],
            "val":   [p.name for p in val_set]
        }, f, indent=2)

    print(f"Found {len(files)} volumes. train={len(tr_set)}, val={len(val_set)}")
    print(f"XY scales: {args.xy_scales}; YZ/XZ scales: {args.yzxz_scales}")

    for p in tqdm(files, desc="processing"):
        split = "val" if p in val_set else "train"
        img = nib.load(str(p))
        vol = img.get_fdata().astype(np.float32)  # (x,y,z)
        if args.normalize == "p1p99":
            vol = percent_norm(vol, 1, 99)
        elif args.normalize == "minmax":
            vmin, vmax = float(np.min(vol)), float(np.max(vol))
            vol = (vol - vmin) / max(1e-6, (vmax - vmin))

        vid = p.name.replace(".nii.gz", "")

        # XY
        tag = args.normalize if args.normalize != "none" else "raw"
        # save_xy_sets(vol, vid, out_root / f"cells_xy_{tag}" / split, args.xy_scales)
        # YZ
        save_yz_sets(vol, vid, out_root / f"cells_yz_{tag}" / split, args.yzxz_scales)
        # XZ
        save_xz_sets(vol, vid, out_root / f"cells_xz_{tag}" / split, args.yzxz_scales)

    print("Done.")

if __name__ == "__main__":
    main()


