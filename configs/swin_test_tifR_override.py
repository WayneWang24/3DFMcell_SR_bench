_base_ = ['swinir_x4_cells_xy.py']


# 自定义一个最小可用的测试 pipeline（与训练时的RGB/3通道设置一致）
test_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt',  color_type='color', channel_order='rgb'),
    dict(type='PackInputs')
]

# 定义 test_dataloader，指向你的 LR / GT 文件夹
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type='BasicImageDataset',
        data_root='',
        data_prefix=dict(
            img='/root/autodl-tmp/datasets/raw_cell_datasets/beforedeconv/tifR/LR_bicubic/X4',
            gt ='/root/autodl-tmp/datasets/raw_cell_datasets/beforedeconv/tifR/gt',
        ),
        pipeline=test_pipeline,
    ),
)

# 指标：PSNR / SSIM / LPIPS（首次需要 pip install lpips）
test_evaluator = [
    dict(type='PSNR', input_order='CHW'),
    dict(type='SSIM', input_order='CHW')
]