# configs/glean_x_cells_xy_template.py
# 继承你的 base（包含 DDP、loop、optim、scheduler、hooks、evaluator）
_base_ = './_base_/models/base_glean.py'

# ===== 可被 --cfg-options 覆盖的变量（默认值仅作兜底）=====
experiment_name = 'glean_x4_cells_xy_p1p99'  # ★ 可覆盖
data_root = '/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99'  # ★ 可覆盖
scale = 4              # ★ 可覆盖：2/4/8
img_size_lr = 64      # ★ 可覆盖：LR 裁片（GLEAN 输入尺寸）
use_pretrained = False # ★ 可覆盖：是否加载 StyleGAN2 FFHQ 先验（跨域不一定有效）
pretrain_url = (
    'http://download.openmmlab.com/mmediting/stylegan2/official_weights/'
    'stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth'
)

work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

# ===== 模型（GLEAN-StyleGANv2）=====
in_size = img_size_lr             # GLEAN 输入固定尺寸
out_size = img_size_lr * scale    # 输出固定尺寸（必须成倍）

model = dict(
    type='SRGAN',
    generator=dict(
        type='GLEANStyleGANv2',
        in_size=in_size,
        out_size=out_size,
        style_channels=512,
        init_cfg=(dict(
            type='Pretrained',
            checkpoint=pretrain_url,
            prefix='generator_ema'
        ) if use_pretrained else None)
    ),
    discriminator=dict(
        type='StyleGANv2Discriminator',
        in_size=out_size,
        init_cfg=(dict(
            type='Pretrained',
            checkpoint=pretrain_url,
            prefix='discriminator'
        ) if use_pretrained else None)
    ),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'21': 1.0},   # vgg16 relu3_3
        vgg_type='vgg16',
        perceptual_weight=2e-3,
        style_weight=0,
        norm_img=True,
        criterion='mse',
        pretrained='torchvision://vgg16'
    ),
    gan_loss=dict(
        type='GANLoss', gan_type='vanilla',
        loss_weight=2e-3, real_label_val=1.0, fake_label_val=0.0
    ),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR', 'SSIM'], crop_border=scale),  # 与其他方法对齐
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[1., 1., 1.],  # 与你其它基线保持 0–1 口径
    ),
)

# ===== 数据管线：配对 LR/HR（不再做人脸退化）=====
train_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='LoadImageFromFile', key='gt',  color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=out_size),  # GT 裁片 = out_size；对应 LR = in_size
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

# ===== 数据集（随 scale 自动切 X{scale}）=====
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    batch_size=4,  # GLEAN+GAN 更吃显存；不够可降到 2
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

# 由于 base_glean.py 里启用了 MultiTestLoop，因此需要提供 test_dataloader
test_dataloader = val_dataloader  # 复用
