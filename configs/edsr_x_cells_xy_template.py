# configs/edsr_x_cells_xy_template.py
_base_ = ['./_base_/default_runtime.py']

# ===== 可被命令行覆盖的变量（默认值，命令行 --cfg-options 可改）=====
experiment_name = 'edsr_x4_cells_xy_p1p99'
data_root = '/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99'
scale = 4
img_size = 64  # LR patch；GT patch = img_size * scale

work_dir = f'./results/work_dirs/{experiment_name}'
save_dir = './results/work_dirs/'

# ========== 模型 ==========
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='EDSRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale,      # ← 跟随 scale
        res_scale=0.1,             # 更稳的 EDSR 残差缩放；若想与旧实验对齐可改回 1.0
        rgb_mean=[0.0, 0.0, 0.0],
        rgb_std=[1.0, 1.0, 1.0],
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR', 'SSIM'], crop_border=scale),  # ← 跟随 scale
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[1., 1., 1.],   # 与 SRCNN/SwinIR 统一到 0–1 口径
    )
)

# ========== 数据 Pipeline ==========
train_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='LoadImageFromFile', key='gt',  color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=img_size * scale),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='LoadImageFromFile', key='gt',  color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='PackInputs')
]
test_pipeline = val_pipeline

# ========== 数据集（随 scale 自动切 X{scale}）==========
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    batch_size=24,
    num_workers=4,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + '/train',
        data_prefix=dict(img=f'LR_bicubic/X{scale}', gt='HR'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + '/val',
        data_prefix=dict(img=f'LR_bicubic/X{scale}', gt='HR'),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# ========== 训练/验证/测试循环 ==========
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=5000)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        by_epoch=False,
        save_best='PSNR',
        rule='greater',
        out_dir=save_dir,
        save_last=True
    ),
    logger=dict(type='LoggerHook', interval=100),
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),  # 让下面的 LR 调度生效
)

# ========== 验证指标 ==========
val_evaluator  = [dict(type='PSNR', input_order='CHW'),
                  dict(type='SSIM', input_order='CHW')]
test_evaluator = val_evaluator

# ========== 优化器/学习率 ==========
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
)
param_scheduler = dict(type='MultiStepLR', by_epoch=False, milestones=[80000], gamma=0.5)

# 复现性（可选）
randomness = dict(seed=2025, deterministic=False)
