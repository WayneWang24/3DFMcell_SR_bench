import os
import glob
import argparse
import random
import math
import numpy as np
import tifffile as tiff
import imageio.v2 as iio
import cv2
from skimage.transform import resize

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_volume(path):
    """读取 3D TIF，返回 shape=(Z,H,W)，保留原始 dtype（常见 uint16）"""
    vol = tiff.imread(path)
    vol = np.asarray(vol)
    if vol.ndim == 2:
        # 单张也当成 (1,H,W)
        vol = vol[None, ...]
    elif vol.ndim == 3:
        pass
    else:
        raise ValueError(f'Unsupported ndim {vol.ndim} for {path}')

    # 期望两边为 512，另一边为 Z（94/60/30...）
    shp = vol.shape
    # 常见两种： (Z, H, W) 或 (H, W, Z)
    if shp[1] == 512 and shp[2] == 512:
        # already (Z,512,512) 或 (H=512,W=512, Z=? 不可能出现在 [1]==512且[2]==512）
        zhw = vol
    elif shp[0] == 512 and shp[1] == 512:
        # (512,512,Z) -> 转成 (Z,512,512)
        zhw = np.transpose(vol, (2,0,1))
    else:
        # 尝试自动识别：找到非 512 的维度作为 Z
        dims = list(shp)
        if dims.count(512) >= 2:
            z_axis = [i for i,d in enumerate(dims) if d != 512][0]
            if z_axis == 0:
                zhw = vol
            elif z_axis == 2:
                zhw = np.transpose(vol, (2,0,1))
            else:
                # (H,Z,W) -> (Z,H,W)
                zhw = np.transpose(vol, (1,0,2))
        else:
            raise ValueError(f'Cannot infer axes for {path}, shape={shp}')
    return zhw

def imwrite16(path, img):
    """以 16-bit（或原 dtype）保存 PNG/TIF；自动裁剪到数据范围"""
    img = np.asarray(img)
    if img.dtype == np.float32 or img.dtype == np.float64:
        # 假设在 [0,1] 或原范围，做简单归一
        m0, m1 = img.min(), img.max()
        if m1 > 1.0:
            # 直接裁剪到 [0, 65535]
            img = np.clip(img, 0, 65535).astype(np.uint16)
        else:
            img = (np.clip(img, 0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    elif img.dtype == np.uint8:
        # 也可以直接保存
        pass
    elif img.dtype == np.uint16:
        pass
    else:
        # 其他整型转 16-bit
        img = np.clip(img, 0, 65535).astype(np.uint16)
    # 用 imageio 写 16bit PNG/TIF 都可；PNG 更通用
    iio.imwrite(path, img)

def bicubic_resize(image, out_w, out_h):
    """双三次 + 抗锯齿，保持 16-bit 动态范围"""
    # OpenCV 的 INTER_CUBIC 不带 anti_aliasing，skimage 有 anti_aliasing，但相对慢
    # 这里用 skimage，确保高质量
    return resize(image, (out_h, out_w), order=3, preserve_range=True, anti_aliasing=True).astype(image.dtype)

def process_xy(vol, vid, out_root, split, xy_scale=4):
    """XY：每个 z-slice -> HR=512x512；LR_bicubic/X4=128x128"""
    Z, H, W = vol.shape
    assert H == 512 and W == 512, 'XY 期望 H=W=512'
    base = os.path.join(out_root, 'cells_xy', split)
    hr_dir = os.path.join(base, 'HR')
    lr_dir = os.path.join(base, f'LR_bicubic/X{xy_scale}')
    ensure_dir(hr_dir); ensure_dir(lr_dir)

    for z in range(Z):
        img = vol[z, :, :]
        hr_name = f'{vid}_z{z:04d}.png'
        lr_name = hr_name
        imwrite16(os.path.join(hr_dir, hr_name), img)
        lr = bicubic_resize(img, W//xy_scale, H//xy_scale)
        imwrite16(os.path.join(lr_dir, lr_name), lr)

def process_yz(vol, vid, out_root, split, yzxz_scales=(4,), target_z=None):
    """YZ：对每个 x -> 图像 shape (Z,H)= (Z,512)。保存 HR；可选生成 Xs 和 matched LR"""
    Z, H, W = vol.shape
    base = os.path.join(out_root, 'cells_yz', split)
    hr_dir = os.path.join(base, 'HR')
    ensure_dir(hr_dir)
    lr_dirs_s = {}
    for s in yzxz_scales:
        d = os.path.join(base, f'LR_bicubic/X{s}')
        ensure_dir(d); lr_dirs_s[s] = d
    lr_matched_dir = None
    w_lr_matched = None
    if target_z is not None:
        f = target_z / Z
        w_lr_matched = max(1, int(round(H / f)))
        lr_matched_dir = os.path.join(base, 'LR_matched')
        ensure_dir(lr_matched_dir)

    for x in range(W):
        yz = vol[:, :, x]      # (Z,H)
        name = f'{vid}_x{x:04d}.png'
        imwrite16(os.path.join(hr_dir, name), yz)
        # 固定倍率 LR
        for s, d in lr_dirs_s.items():
            lr = bicubic_resize(yz, H//s, Z)
            imwrite16(os.path.join(d, name), lr)
        # 匹配倍数 LR
        if lr_matched_dir is not None:
            lr = bicubic_resize(yz, w_lr_matched, Z)
            imwrite16(os.path.join(lr_matched_dir, name), lr)

def process_xz(vol, vid, out_root, split, yzxz_scales=(4,), target_z=None):
    """XZ：对每个 y -> 图像 shape (Z,W)= (Z,512)。保存 HR；可选生成 Xs 和 matched LR"""
    Z, H, W = vol.shape
    base = os.path.join(out_root, 'cells_xz', split)
    hr_dir = os.path.join(base, 'HR')
    ensure_dir(hr_dir)
    lr_dirs_s = {}
    for s in yzxz_scales:
        d = os.path.join(base, f'LR_bicubic/X{s}')
        ensure_dir(d); lr_dirs_s[s] = d
    lr_matched_dir = None
    w_lr_matched = None
    if target_z is not None:
        f = target_z / Z
        w_lr_matched = max(1, int(round(W / f)))
        lr_matched_dir = os.path.join(base, 'LR_matched')
        ensure_dir(lr_matched_dir)

    for y in range(H):
        xz = vol[:, y, :]      # (Z,W)
        name = f'{vid}_y{y:04d}.png'
        imwrite16(os.path.join(hr_dir, name), xz)
        # 固定倍率 LR
        for s, d in lr_dirs_s.items():
            lr = bicubic_resize(xz, W//s, Z)
            imwrite16(os.path.join(d, name), lr)
        # 匹配倍数 LR
        if lr_matched_dir is not None:
            lr = bicubic_resize(xz, w_lr_matched, Z)
            imwrite16(os.path.join(lr_matched_dir, name), lr)

def main(args):
    input_dir = args.input_dir
    out_root = args.out_root
    xy_scale = args.xy_scale
    yzxz_scales = tuple(args.yzxz_scales) if args.yzxz_scales else ()
    target_z = args.target_z

    vols = sorted(glob.glob(os.path.join(input_dir, '*.tif'))) + \
           sorted(glob.glob(os.path.join(input_dir, '*.tiff')))
    assert len(vols) > 0, f'No tif found in {input_dir}'

    random.seed(args.seed)
    random.shuffle(vols)
    n_val = max(1, int(round(len(vols) * args.val_ratio)))
    val_set = set(vols[:n_val])
    train_set = set(vols[n_val:])

    print(f'Found {len(vols)} volumes; train={len(train_set)}, val={len(val_set)}')
    print(f'XY scale: X{xy_scale}; YZ/XZ scales: {yzxz_scales}; target_z: {target_z}')

    for path in vols:
        vid = os.path.splitext(os.path.basename(path))[0]
        split = 'val' if path in val_set else 'train'
        vol = read_volume(path)  # (Z,512,512)
        Z, H, W = vol.shape
        if H != 512 or W != 512:
            raise ValueError(f'{vid} has non-512 XY plane: {vol.shape}')
        # （可选）强度预处理：百分位截断再拉伸 —— 如需开启，请取消注释
        # p1, p99 = np.percentile(vol, [1, 99])
        # vol = np.clip(vol, p1, p99)
        # vol = ((vol - p1) / max(1e-6, (p99 - p1)) * 65535.0 + 0.5).astype(np.uint16)

        # XY
        process_xy(vol, vid, out_root, split, xy_scale=xy_scale)
        # YZ / XZ
        process_yz(vol, vid, out_root, split, yzxz_scales=yzxz_scales, target_z=target_z)
        process_xz(vol, vid, out_root, split, yzxz_scales=yzxz_scales, target_z=target_z)

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='包含 3D tif 的文件夹')
    parser.add_argument('--out_root', required=True, help='输出数据根目录（会创建 cells_xy/yz/xz 子目录）')
    parser.add_argument('--xy_scale', type=int, default=4, help='XY 的下采样倍数（默认 X4）')
    parser.add_argument('--yzxz_scales', type=int, nargs='*', default=[4], help='为 YZ/XZ 生成的固定倍率 LR 列表，如 2 4 8')
    parser.add_argument('--target_z', type=int, default=None, help='目标 Z（如 448）。设置后会额外生成 LR_matched（宽度=round(512/(target_z/Z)))')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='按体划分的验证集比例')
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    main(args)
