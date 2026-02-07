#!/usr/bin/env python
"""
批量超分辨率模型评测脚本

指标：
- Full-reference: PSNR ↑, SSIM ↑, MS-SSIM ↑, LPIPS ↓, VIF ↑
- No-reference: NIQE ↓, PIQE ↓, NRQM ↑
- Distribution-based: FID ↓

使用方法：
python scripts/eval/batch_eval_sr.py \
  --model-dirs results/work_dirs/edsr_x4_singleS_xy_p1p99 results/work_dirs/srcnn_x4_singleS_xy_p1p99 \
  --data-roots /path/to/dataset1/val /path/to/dataset2/val \
  --scales 4 \
  --output results/eval_results.csv
"""

import os
import argparse
import glob
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# 注册 mmagic 模型到 mmengine registry（需先修复 Adafactor 冲突，见 run_all_eval.sh）
import mmagic.models  # noqa: F401

# 图像质量指标
import torch
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

# 可选依赖
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[警告] lpips 未安装，LPIPS 指标将跳过。安装: pip install lpips")

try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False
    print("[警告] pytorch_msssim 未安装，MS-SSIM 指标将跳过。安装: pip install pytorch-msssim")

try:
    import pyiqa
    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False
    print("[警告] pyiqa 未安装，VIF/NIQE/PIQE/NRQM 指标将跳过。安装: pip install pyiqa")

try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("[警告] pytorch-fid 未安装，FID 指标将跳过。安装: pip install pytorch-fid")


def imread_rgb(path):
    """读取图像，返回 RGB uint8"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # 处理 16-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    # 灰度转 RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def to_tensor(img, device='cuda'):
    """numpy HWC uint8 -> torch BCHW float [0,1]"""
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return t.unsqueeze(0).to(device)


class MetricCalculator:
    """指标计算器"""

    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        # LPIPS
        self.lpips_fn = None
        if HAS_LPIPS:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_fn.eval()

        # PyIQA metrics
        self.vif_fn = None
        self.niqe_fn = None
        self.piqe_fn = None
        self.nrqm_fn = None
        if HAS_PYIQA:
            try:
                self.vif_fn = pyiqa.create_metric('vif', device=self.device)
            except:
                print("[警告] VIF 初始化失败")
            try:
                self.niqe_fn = pyiqa.create_metric('niqe', device=self.device)
            except:
                print("[警告] NIQE 初始化失败")
            try:
                self.piqe_fn = pyiqa.create_metric('piqe', device=self.device)
            except:
                print("[警告] PIQE 初始化失败")
            try:
                self.nrqm_fn = pyiqa.create_metric('nrqm', device=self.device)
            except:
                print("[警告] NRQM 初始化失败")

    def calc_psnr(self, sr, hr):
        """PSNR (越高越好)"""
        return calc_psnr(hr, sr, data_range=255)

    def calc_ssim(self, sr, hr):
        """SSIM (越高越好)"""
        return calc_ssim(hr, sr, channel_axis=2, data_range=255)

    def calc_msssim(self, sr, hr):
        """MS-SSIM (越高越好)"""
        if not HAS_MSSSIM:
            return np.nan
        sr_t = to_tensor(sr, self.device)
        hr_t = to_tensor(hr, self.device)
        # MS-SSIM 需要至少 160x160
        if sr_t.shape[2] < 160 or sr_t.shape[3] < 160:
            return np.nan
        with torch.no_grad():
            return ms_ssim(sr_t, hr_t, data_range=1.0).item()

    def calc_lpips(self, sr, hr):
        """LPIPS (越低越好)"""
        if self.lpips_fn is None:
            return np.nan
        sr_t = to_tensor(sr, self.device) * 2 - 1  # [-1, 1]
        hr_t = to_tensor(hr, self.device) * 2 - 1
        with torch.no_grad():
            return self.lpips_fn(sr_t, hr_t).item()

    def calc_vif(self, sr, hr):
        """VIF (越高越好)"""
        if self.vif_fn is None:
            return np.nan
        sr_t = to_tensor(sr, self.device)
        hr_t = to_tensor(hr, self.device)
        with torch.no_grad():
            return self.vif_fn(sr_t, hr_t).item()

    def calc_niqe(self, sr):
        """NIQE (越低越好) - No-reference"""
        if self.niqe_fn is None:
            return np.nan
        sr_t = to_tensor(sr, self.device)
        with torch.no_grad():
            return self.niqe_fn(sr_t).item()

    def calc_piqe(self, sr):
        """PIQE (越低越好) - No-reference"""
        if self.piqe_fn is None:
            return np.nan
        sr_t = to_tensor(sr, self.device)
        with torch.no_grad():
            return self.piqe_fn(sr_t).item()

    def calc_nrqm(self, sr):
        """NRQM (越高越好) - No-reference"""
        if self.nrqm_fn is None:
            return np.nan
        sr_t = to_tensor(sr, self.device)
        with torch.no_grad():
            return self.nrqm_fn(sr_t).item()

    def calc_all(self, sr, hr):
        """计算所有指标"""
        results = {
            'PSNR': self.calc_psnr(sr, hr),
            'SSIM': self.calc_ssim(sr, hr),
            'MS-SSIM': self.calc_msssim(sr, hr),
            'LPIPS': self.calc_lpips(sr, hr),
            'VIF': self.calc_vif(sr, hr),
            'NIQE': self.calc_niqe(sr),
            'PIQE': self.calc_piqe(sr),
            'NRQM': self.calc_nrqm(sr),
        }
        return results


def find_checkpoint(model_dir):
    """找到最佳或最新的 checkpoint"""
    model_dir = Path(model_dir)

    # 优先找 best
    best_ckpts = list(model_dir.glob('best_*.pth')) + list(model_dir.glob('**/best_*.pth'))
    if best_ckpts:
        return str(best_ckpts[0])

    # 其次找 iter_*.pth 中最大的
    iter_ckpts = list(model_dir.glob('iter_*.pth')) + list(model_dir.glob('**/iter_*.pth'))
    if iter_ckpts:
        iter_ckpts.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(iter_ckpts[-1])

    # 最后找 last.pth
    last_ckpts = list(model_dir.glob('last.pth')) + list(model_dir.glob('**/last.pth'))
    if last_ckpts:
        return str(last_ckpts[0])

    return None


def find_config(model_dir):
    """找到配置文件"""
    model_dir = Path(model_dir)
    configs = list(model_dir.glob('*.py')) + list(model_dir.glob('**/*.py'))
    if configs:
        return str(configs[0])
    return None


def run_inference_mmagic(config_path, checkpoint_path, lr_dir, output_dir):
    """使用 mmagic 进行推理"""
    from mmagic.apis import MMagicInferencer

    os.makedirs(output_dir, exist_ok=True)

    inferencer = MMagicInferencer(model_name=None, model_config=config_path,
                                   model_ckpt=checkpoint_path, device='cuda')

    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*')))
    for lr_path in tqdm(lr_files, desc='Inference'):
        result = inferencer(lr_path)
        sr_img = result[0]

        out_path = os.path.join(output_dir, os.path.basename(lr_path))
        cv2.imwrite(out_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))


def run_inference_simple(config_path, checkpoint_path, lr_dir, output_dir, scale=4):
    """简单推理（不依赖完整 mmagic API）"""
    import torch
    from mmengine.config import Config
    from mmengine.registry import MODELS

    os.makedirs(output_dir, exist_ok=True)

    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model.generator)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = {k.replace('generator.', ''): v for k, v in ckpt['state_dict'].items()
                      if k.startswith('generator.')}
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*')))
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    lr_files = [f for f in lr_files if Path(f).suffix.lower() in exts]

    for lr_path in tqdm(lr_files, desc='Inference'):
        img = imread_rgb(lr_path)
        if img is None:
            continue

        img_t = to_tensor(img, 'cuda')
        with torch.no_grad():
            sr_t = model(img_t)

        sr = (sr_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        out_path = os.path.join(output_dir, Path(lr_path).stem + '.png')
        cv2.imwrite(out_path, cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))


def eval_pair_dirs(sr_dir, hr_dir, calc):
    """评估一对 SR/HR 目录"""
    sr_files = {Path(f).stem: f for f in glob.glob(os.path.join(sr_dir, '*'))}
    hr_files = {Path(f).stem: f for f in glob.glob(os.path.join(hr_dir, '*'))}

    common = sorted(set(sr_files.keys()) & set(hr_files.keys()))
    if not common:
        print(f"[警告] 无匹配文件对: SR={sr_dir}, HR={hr_dir}")
        return {}

    all_metrics = defaultdict(list)
    for stem in tqdm(common, desc='Evaluating'):
        sr = imread_rgb(sr_files[stem])
        hr = imread_rgb(hr_files[stem])
        if sr is None or hr is None:
            continue

        # 确保尺寸一致
        if sr.shape != hr.shape:
            hr = cv2.resize(hr, (sr.shape[1], sr.shape[0]))

        metrics = calc.calc_all(sr, hr)
        for k, v in metrics.items():
            if not np.isnan(v):
                all_metrics[k].append(v)

    # 计算均值
    return {k: np.mean(v) for k, v in all_metrics.items()}


def calc_fid(sr_dir, hr_dir, device='cuda'):
    """计算 FID"""
    if not HAS_FID:
        return np.nan
    try:
        fid = fid_score.calculate_fid_given_paths(
            [sr_dir, hr_dir],
            batch_size=50,
            device=device,
            dims=2048
        )
        return fid
    except Exception as e:
        print(f"[警告] FID 计算失败: {e}")
        return np.nan


def main():
    parser = argparse.ArgumentParser(description='批量超分辨率模型评测')
    parser.add_argument('--model-dirs', nargs='+', required=True,
                        help='模型目录列表，如 results/work_dirs/model1 results/work_dirs/model2')
    parser.add_argument('--data-roots', nargs='+', required=True,
                        help='数据集根目录列表（包含 HR 和 LR_bicubic/X* 子目录）')
    parser.add_argument('--scales', type=int, nargs='+', default=[4],
                        help='超分倍率列表')
    parser.add_argument('--output', type=str, default='results/eval_results.csv',
                        help='输出 CSV 路径')
    parser.add_argument('--sr-subdir', type=str, default=None,
                        help='如果已有 SR 结果，指定子目录名（跳过推理）')
    parser.add_argument('--skip-inference', action='store_true',
                        help='跳过推理，直接评估已有结果')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    calc = MetricCalculator(args.device)
    results = []

    for model_dir in args.model_dirs:
        model_name = Path(model_dir).name

        # 找 checkpoint 和 config
        ckpt = find_checkpoint(model_dir)
        config = find_config(model_dir)

        if ckpt is None:
            print(f"[跳过] 未找到 checkpoint: {model_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"Checkpoint: {ckpt}")
        print(f"Config: {config}")

        for data_root in args.data_roots:
            data_name = Path(data_root).parent.name + '/' + Path(data_root).name
            hr_dir = os.path.join(data_root, 'HR')

            if not os.path.isdir(hr_dir):
                print(f"[跳过] HR 目录不存在: {hr_dir}")
                continue

            for scale in args.scales:
                lr_dir = os.path.join(data_root, f'LR_bicubic/X{scale}')
                if not os.path.isdir(lr_dir):
                    print(f"[跳过] LR 目录不存在: {lr_dir}")
                    continue

                print(f"\n数据集: {data_name}, Scale: X{scale}")

                # SR 输出目录
                sr_dir = os.path.join(model_dir, 'sr_outputs',
                                      Path(data_root).parent.name,
                                      Path(data_root).name, f'X{scale}')

                # 推理
                if not args.skip_inference:
                    if config is not None:
                        print(f"运行推理...")
                        try:
                            run_inference_simple(config, ckpt, lr_dir, sr_dir, scale)
                        except Exception as e:
                            print(f"[错误] 推理失败: {e}")
                            continue
                    else:
                        print(f"[跳过] 无配置文件，无法推理")
                        continue

                if not os.path.isdir(sr_dir):
                    print(f"[跳过] SR 目录不存在: {sr_dir}")
                    continue

                # 评估
                print(f"评估指标...")
                metrics = eval_pair_dirs(sr_dir, hr_dir, calc)

                # FID
                print(f"计算 FID...")
                metrics['FID'] = calc_fid(sr_dir, hr_dir, args.device)

                # 记录结果
                row = {
                    'Model': model_name,
                    'Dataset': data_name,
                    'Scale': f'X{scale}',
                    **metrics
                }
                results.append(row)

                # 打印
                print(f"结果: PSNR={metrics.get('PSNR', 'N/A'):.4f}, "
                      f"SSIM={metrics.get('SSIM', 'N/A'):.4f}, "
                      f"LPIPS={metrics.get('LPIPS', 'N/A'):.4f}")

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        # 调整列顺序
        cols = ['Model', 'Dataset', 'Scale', 'PSNR', 'SSIM', 'MS-SSIM',
                'LPIPS', 'VIF', 'NIQE', 'PIQE', 'NRQM', 'FID']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df.to_csv(args.output, index=False, float_format='%.4f')
        print(f"\n结果已保存到: {args.output}")
        print(df.to_string(index=False))
    else:
        print("\n无评测结果")


if __name__ == '__main__':
    main()
