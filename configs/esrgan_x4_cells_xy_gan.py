# ESRGAN (GAN finetune) for cells Condition A
_base_ = ['./_base_/default_runtime.py']
ckpt_cfg = dict(path=None)
experiment_name = 'esrgan_x4_cells_xy_gan_p1p99'
work_dir = f'./results/work_dirs/{experiment_name}'
save_dir = './results/work_dirs/'
scale = 4
# pretrain_generator_ckpt = 'results/work_dirs/esrgan_psnr_x4_cells_xy_p1p99/iter_100000.pth'
img_size = 32  # LR patch; GT patch=256

# 把这个路径改成你阶段A的 best.pth


model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')  # 分布式时更稳

model = dict(
    type='ESRGAN',
    generator=dict(
        type='RRDBNet',
        in_channels=3, out_channels=3,
        mid_channels=64, num_blocks=23, growth_channels=32,
        upscale_factor=scale,
        init_cfg=dict(type='Pretrained', checkpoint=None, prefix='generator.')
    ),
    discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=64),
    pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'34': 1.0},
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False
    ),
    gan_loss=dict(
        type='GANLoss', gan_type='vanilla',
        loss_weight=5e-3, real_label_val=1.0, fake_label_val=0.0
    ),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR','SSIM'], crop_border=scale),
    data_preprocessor=dict(type='DataPreprocessor', mean=[0.,0.,0.], std=[1.,1.,1.])
)

train_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='LoadImageFromFile', key='gt',  color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=img_size*scale),  # ESRGAN 经典 128~192 GT patch；这里 256 如显存紧张可调小
    dict(type='Flip', keys=['img','gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img','gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img','gt'], transpose_ratio=0.5),
    dict(type='PackInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='LoadImageFromFile', key='gt',  color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='PackInputs')
]
test_pipeline = val_pipeline

dataset_type = 'BasicImageDataset'
data_root = '/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99'

train_dataloader = dict(
    batch_size=12,  # GAN会更吃显存，先设 12；不够再降
    num_workers=4, drop_last=True, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root+'/train',
        data_prefix=dict(img='LR_bicubic/X4', gt='HR'),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root+'/val',
        data_prefix=dict(img='LR_bicubic/X4', gt='HR'),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=5000)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5000, by_epoch=False,
                    save_best='PSNR', rule='greater', out_dir=save_dir, save_last=True),
    logger=dict(type='LoggerHook', interval=100),
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
)

val_evaluator  = [dict(type='PSNR', input_order='CHW'), dict(type='SSIM', input_order='CHW')]
test_evaluator = val_evaluator

# 双优化器（G和D）
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
    ),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
    ),
)

# 分段衰减；GAN阶段建议更保守
param_scheduler = dict(
    type='MultiStepLR', by_epoch=False,
    milestones=[50000, 80000], gamma=0.5
)

randomness = dict(seed=2025, deterministic=False)
