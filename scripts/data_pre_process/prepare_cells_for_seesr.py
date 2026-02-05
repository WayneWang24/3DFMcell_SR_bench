# tools/prepare_cells_for_seesr.py
import os, cv2, glob, argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def imread_any(p):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype.kind == "f":
        img = np.nan_to_num(img)
        lo, hi = np.percentile(img, [0.01, 99.99])
        if hi > lo:
            img = (np.clip(img, lo, hi) - lo) / (hi - lo)
        img = (img * 65535.0 + 0.5).astype(np.uint16)
    return img

def ensure_3ch_uint8(img, norm: str = "p1p99", to_uint8: bool = True) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.merge([img, img, img])
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.merge([img[:,:,0], img[:,:,0], img[:,:,0]])
    elif img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]

    if norm.lower() != "none":
        img = img.astype(np.float64)
        if norm.lower() == "minmax":
            lo, hi = img.min(), img.max()
        elif norm.lower() == "p1p99":
            lo, hi = np.percentile(img, [1, 99])
        else:
            raise ValueError("norm must be none|minmax|p1p99")
        hi = max(hi, lo + 1e-6)
        img = np.clip((img - lo) / (hi - lo), 0, 1)
        if to_uint8:
            img = (img * 255.0 + 0.5).astype(np.uint8)
        else:
            img = img.astype(np.float32)
    else:
        if to_uint8 and img.dtype != np.uint8:
            lo, hi = np.percentile(img.astype(np.float64), [0.1, 99.9])
            hi = max(hi, lo + 1e-6)
            img = np.clip((img - lo) / (hi - lo), 0, 1)
            img = (img * 255.0 + 0.5).astype(np.uint8)
    return img

def upsample_to_size(img: np.ndarray, size: Tuple[int, int], interp: str = "cubic") -> np.ndarray:
    h, w = size
    map_interp = {
        "nearest": cv2.INTER_NEAREST,
        "linear":  cv2.INTER_LINEAR,
        "cubic":   cv2.INTER_CUBIC,
        "area":    cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4
    }
    return cv2.resize(img, (w, h), interpolation=map_interp.get(interp, cv2.INTER_CUBIC))

def pad_to_min(img: np.ndarray, min_h: int, min_w: int, mode: str = "edge") -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return img
    top, bottom, left, right = 0, pad_h, 0, pad_w
    if mode == "edge":
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    elif mode == "reflect":
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    else:
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

def tile_pairs(hr, lr, tile=512, stride=512):
    H, W = hr.shape[:2]
    patches = []
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            hr_p = hr[y:y+tile, x:x+tile]
            lr_p = lr[y:y+tile, x:x+tile]
            if hr_p.shape[:2] == (tile, tile) and lr_p.shape[:2] == (tile, tile):
                patches.append(((y,x), hr_p, lr_p))
    return patches

def find_hr_dir(src_root: Path) -> Path:
    for name in ["HR", "hr"]:
        p = src_root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"未找到 HR 目录（尝试 HR/ 或 hr/）：{src_root}")

def find_lr_dir(src_root: Path, scale: int) -> Optional[Path]:
    """
    兼容：
      - LR_bicubic/x2|x4|x8
      - lr_x2 / x2 / lr
    """
    candidates = [
        src_root / "LR_bicubic" / f"X{scale}",
        src_root / f"lr_x{scale}",
        src_root / f"x{scale}",
        src_root / "lr"
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=False,
                    default='/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99/train',
                    help="来源 train 路径，例如 .../fastz_200_highL/cells_xz_p1p99/train（其下包含 HR/ 与 LR_bicubic/x*）")
    ap.add_argument("--dest-root", required=False,
                    default='/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/set_seesr/fastz_200_highL/cells_xy_p1p99',
                    help="目标 SeeSR 数据根，例如 preset/datasets/training_datasets/cells_xz_p1p99")
    ap.add_argument("--scales", default="2,4,8", help="处理倍率，逗号分隔，如 2,4,8")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--pad", default="edge", choices=["none","edge","reflect","constant"])
    ap.add_argument("--norm", default="p1p99", choices=["none","minmax","p1p99"])
    ap.add_argument("--to-uint8", action="store_true")
    ap.add_argument("--lr-up-interp", default="cubic",
                    choices=["nearest","linear","cubic","area","lanczos"])
    ap.add_argument("--exts", default=".png,.jpg,.jpeg,.tif,.tiff")
    ap.add_argument("--tag-mode", default="empty", choices=["empty","constant"])
    ap.add_argument("--tag-text", default="cell slice")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    hr_dir = find_hr_dir(src_root)

    dest_root = Path(args.dest_root)
    (dest_root / "gt").mkdir(parents=True, exist_ok=True)
    (dest_root / "lr").mkdir(parents=True, exist_ok=True)
    (dest_root / "tag").mkdir(parents=True, exist_ok=True)

    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])

    hr_map = {Path(p).stem: p for p in glob.glob(str(hr_dir / "*")) if Path(p).suffix.lower() in exts}
    if not hr_map:
        raise RuntimeError(f"HR 目录无可用图像：{hr_dir}")

    total_written = 0
    for scale in scales:
        lr_dir = find_lr_dir(src_root, scale)
        if lr_dir is None:
            print(f"[警告] 未找到倍率 x{scale} 的 LR 目录（期望 LR_bicubic/x{scale} 等），跳过。")
            continue

        lr_map = {Path(p).stem: p for p in glob.glob(str(lr_dir / "*")) if Path(p).suffix.lower() in exts}
        common = sorted(set(hr_map.keys()) & set(lr_map.keys()))
        if not common:
            print(f"[警告] 倍率 x{scale} 下 HR/LR 同名对为空，跳过。")
            continue

        print(f"== 处理倍率 x{scale}，成对图像数：{len(common)}，LR目录：{lr_dir}")
        for stem in common:
            hr_img = imread_any(hr_map[stem])
            lr_img = imread_any(lr_map[stem])
            if hr_img is None or lr_img is None:
                print(f"[跳过] 读取失败：{stem}")
                continue

            hr_img = ensure_3ch_uint8(hr_img, norm=args.norm, to_uint8=args.to_uint8)
            lr_img = ensure_3ch_uint8(lr_img, norm=args.norm, to_uint8=args.to_uint8)

            # 将 LR 上采样到 HR 尺寸对齐
            hH, wH = hr_img.shape[:2]
            lr_up = upsample_to_size(lr_img, (hH, wH), interp=args.lr_up_interp)

            if args.pad != "none":
                hr_img = pad_to_min(hr_img, args.tile, args.tile, mode=args.pad)
                lr_up = pad_to_min(lr_up, args.tile, args.tile, mode=args.pad)

            patches = tile_pairs(hr_img, lr_up, tile=args.tile, stride=args.stride)
            if not patches:
                print(f"[跳过] 无法切出 {args.tile}x{args.tile}：{stem}")
                continue

            for (y,x), h, l in patches:
                save_stem = f"{stem}_s{scale}_y{y}_x{x}"
                cv2.imwrite(str(dest_root / "gt" / f"{save_stem}.png"), h)
                cv2.imwrite(str(dest_root / "lr" / f"{save_stem}.png"), l)
                tag_path = dest_root / "tag" / f"{save_stem}.txt"
                tag_text = "" if args.tag_mode == "empty" else args.tag_text
                tag_path.write_text(tag_text, encoding="utf-8")
                total_written += 1

    print(f"\n✅ 完成！共写出对齐 patch：{total_written} 对（gt/lr/tag）")
    print(f"输出目录：{dest_root}")

if __name__ == "__main__":
    main()