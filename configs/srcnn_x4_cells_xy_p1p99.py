
# python ../mmagic/tools/train.py ../configs/swinir_x4_cells_xy.py --work-dir results/work_dirs/srcnn_x4_single_XY
_base_ = ['../_base_/default_runtime.py']

experiment_name = 'srcnn_x4_cells_xy_p1p99'
work_dir = f'./results/work_dirs/{experiment_name}'
save_dir = './results/work_dirs/'

scale = 4
img_size = 64

# =========================
#   模型 (SRCNN)
# =========================
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='SRCNNNet',
        channels=(3, 64, 32, 3),
        kernel_sizes=(9, 1, 5),
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR', 'SSIM'], crop_border=scale),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        rgb_to_bgr=False
    )
)

# =========================
#   数据 Pipeline
# =========================
train_pipeline = _base_.train_pipeline
for step in train_pipeline:
    if step.get('type') == 'LoadImageFromFile':
        step['color_type'] = 'color'
        step['channel_order'] = 'rgb'

if hasattr(_base_, 'test_pipeline'):
    val_pipeline = _base_.test_pipeline
else:
    val_pipeline = _base_.val_pipeline
for step in val_pipeline:
    if step.get('type') == 'LoadImageFromFile':
        step['color_type'] = 'color'
        step['channel_order'] = 'rgb'

test_pipeline = _base_.test_pipeline
for step in test_pipeline:
    if step.get('type') == 'LoadImageFromFile':
        step['color_type'] = 'color'
        step['channel_order'] = 'rgb'

# =========================
#   数据集 Dataloader
# =========================
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    _delete_=True,
    batch_size=24,
    num_workers=4,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root='/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xy_p1p99/train',
        data_prefix=dict(
            img='LR_bicubic/X4',
            gt='HR'
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root='/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xy_p1p99/val',
        data_prefix=dict(
            img='LR_bicubic/X4',
            gt='HR'
        ),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# =========================
#   训练设置
# =========================
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=100000,
    val_interval=5000
)

default_hooks = dict(
    _delete_=True,
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        by_epoch=False,
        save_best='PSNR',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=100),
    timer=dict(type='IterTimerHook')
)

# =========================
#   验证指标
# =========================
val_evaluator = [
    dict(type='PSNR', input_order='CHW'),
    dict(type='SSIM', input_order='CHW'),
    # dict(type='NIQE', convert_to='gray', input_order='CHW'),
]
test_evaluator = val_evaluator