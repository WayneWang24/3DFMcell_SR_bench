# configs/edsr_x4_cells_xy.py
_base_ = ['./_base_/default_runtime.py']

experiment_name = 'edsr_x4_cells_xy_p1p99'
work_dir = f'./results/work_dirs/{experiment_name}'
save_dir = './results/work_dirs/'

# ===== 核心一致性：与 Condition A 对齐 =====
scale = 4
img_size = 64   # 训练裁片的“LR 尺寸边长”，GT 裁片会是 img_size*scale

# ========== 模型 ==========
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='EDSRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale,
        res_scale=1,
        # 对齐你的细胞数据：关闭 DIV2K 的均值偏移
        rgb_mean=[0.0, 0.0, 0.0],
        rgb_std=[1.0, 1.0, 1.0],
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR', 'SSIM'], crop_border=scale),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[1., 1., 1.],     # 与 SwinIR/SRCNN 统一到 0~1 口径
        # rgb_to_bgr=False
    )
)

# ========== 数据 Pipeline（RGB 三通道、与前两法一致）==========
train_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='LoadImageFromFile', key='gt',  color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=img_size * scale),  # 这里=256，若显存吃紧可把 img_size=48
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

# ========== 数据集（与 SwinIR/SRCNN 同路径结构）==========
dataset_type = 'BasicImageDataset'
data_root = '/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99'
# 如果你这组实验想统一到 singleS_30_highL，就把上面 data_root 改成那条路径即可

train_dataloader = dict(
    # _delete_=True,
    batch_size=24,                 # 与前面两法统一；显存不够就降到 16/12
    num_workers=4,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + '/train',
        data_prefix=dict(img='LR_bicubic/X4', gt='HR'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    # _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root + '/val',
        data_prefix=dict(img='LR_bicubic/X4', gt='HR'),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# ========== 训练循环/Hook（与前两法一致）==========
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(

    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        by_epoch=False,
        save_best='PSNR',
        rule='greater',
        out_dir=save_dir
    ),
    logger=dict(type='LoggerHook', interval=100),
    timer=dict(type='IterTimerHook')
)

# ========== 验证指标（与前两法一致；更多指标用统一离线脚本补）==========
val_evaluator = [
    dict(type='PSNR', input_order='CHW'),
    dict(type='SSIM', input_order='CHW'),
]
test_evaluator = val_evaluator

# ========== 优化器/学习率（保持简单稳定）==========
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
)
param_scheduler = dict(type='MultiStepLR', by_epoch=False, milestones=[80000], gamma=0.5)
