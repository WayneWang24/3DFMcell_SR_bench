

# 继承官方 SwinIR ×4 配置
_base_ = ['swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py']

experiment_name = 'swinir_x4_cells_xy_p1p99'
data_root = '/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99'

work_dir = f'./results/work_dirs/{experiment_name}'
save_dir = './results/work_dirs/'

scale = 4
img_size = 64

# ========== 模型设置 ==========
model = dict(
    generator=dict(
        img_size=img_size,
        in_chans=3,   # 灰度输入
        out_chans=3,  # 灰度输出
        upscale=scale,
    )
)

data_preprocessor = dict(
    type='DataPreprocessor',
    mean=[0.0, 0.0, 0.0],   # 三通道
    std=[1.0, 1.0, 1.0],    # 三通道
    rgb_to_bgr=False        # 保持 RGB
)

# ========== 数据管道 ==========
# train_pipeline = _base_.train_pipeline
# train_pipeline[0]['color_type'] = 'grayscale'  # lq
# train_pipeline[1]['color_type'] = 'grayscale'  # gt
# train_pipeline[3]['gt_patch_size'] = img_size * scale

# val_pipeline = _base_.test_pipeline if hasattr(_base_, 'test_pipeline') else _base_.val_pipeline
# val_pipeline[0]['color_type'] = 'grayscale'
# val_pipeline[1]['color_type'] = 'grayscale'

train_pipeline = _base_.train_pipeline
for step in train_pipeline:
    if step.get('type') == 'LoadImageFromFile':
        step['color_type'] = 'color'      # 从 grayscale → color
        step['channel_order'] = 'rgb'     # 明确 RGB 顺序

# 验证 pipeline
if hasattr(_base_, 'test_pipeline'):
    val_pipeline = _base_.test_pipeline
else:
    val_pipeline = _base_.val_pipeline
for step in val_pipeline:
    if step.get('type') == 'LoadImageFromFile':
        step['color_type'] = 'color'
        step['channel_order'] = 'rgb'

# 测试 pipeline（通常和 val 一致）
test_pipeline = _base_.test_pipeline
for step in test_pipeline:
    if step.get('type') == 'LoadImageFromFile':
        step['color_type'] = 'color'
        step['channel_order'] = 'rgb'

# ========== 数据集 ==========
dataset_type = 'BasicImageDataset'

# train_dataloader = dict(
#     num_workers=4,
#     batch_size=4,
#     drop_last=True,
#     persistent_workers=False,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root='/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xy_p1p99/train',
#         input_folder='LR_bicubic/X4',
#         gt_folder='HR',
#         pipeline=train_pipeline
#     )
# )

# val_dataloader = dict(
#     num_workers=2,
#     batch_size=1,
#     persistent_workers=False,
#     dataset=dict(
#         type=dataset_type,
#         data_root='/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xy_p1p99/val',
#         input_folder='LR_bicubic/X4',
#         gt_folder='HR',
#         pipeline=val_pipeline
#     )
# )

# test_dataloader = dict(
#     _delete_=True,   # 加这一行！
#     num_workers=2,
#     batch_size=1,
#     persistent_workers=False,
#     dataset=dict(
#         type='BasicImageDataset',
#         data_root='/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99/val',
#         input_folder='LR_bicubic/X4',
#         gt_folder='HR',
#         pipeline=val_pipeline
#     )
# )

train_dataloader = dict(
    _delete_=True,
    batch_size=24,
    num_workers=4,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BasicImageDataset',
        data_root=data_root + '/train',
        data_prefix=dict(img=f'LR_bicubic/X{scale}', gt='HR'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type='BasicImageDataset',
        data_root=data_root + '/val',
        data_prefix=dict(img=f'LR_bicubic/X{scale}', gt='HR'),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# ========== 训练循环 ==========
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)

# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True, save_best='PSNR')
# )
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=100000,
    val_interval=5000
)

# default_hooks = dict(
#     _delete_=True,
#     checkpoint=dict(type='CheckpointHook', interval=5000, by_epoch=False, save_best='PSNR')
# )
default_hooks = dict(
    _delete_=True,
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,          # 按 epoch 存；若按 iter 存就改 by_epoch=False, interval=5000
        by_epoch=False,
        save_best='PSNR',    # 想用哪一个指标作最佳就填它
        rule='greater'       # PSNR / SSIM 是“越大越好”
    ),
    logger=dict(type='LoggerHook', interval=100),
    timer=dict(type='IterTimerHook')
)

# ========== 验证指标 ==========
# ========== 验证指标 ==========
val_evaluator = [
    dict(type='PSNR', input_order='CHW'),
    dict(type='SSIM', input_order='CHW'),
    # dict(type='NIQE', convert_to='gray', input_order='CHW'),
]

test_evaluator = val_evaluator

