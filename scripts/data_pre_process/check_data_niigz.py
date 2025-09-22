import os
import nibabel as nib
from collections import Counter
import matplotlib.pyplot as plt

# 数据目录
data_dir = "/root/autodl-tmp/datasets/fastz_volumes"

shapes = {}

for fname in os.listdir(data_dir):
    if fname.lower().endswith(".nii.gz"):
        fpath = os.path.join(data_dir, fname)
        try:
            img = nib.load(fpath)
            arr = img.get_fdata()
            shapes[fname] = arr.shape
        except Exception as e:
            print(f"❌ 读取失败: {fname}, 错误: {e}")

# 统计 shape
shape_counts = Counter(shapes.values())
print("📊 shape 统计：")
for shp, count in shape_counts.items():
    print(f"{shp}: {count} 个文件")

# 保存直方图
labels = [str(k) for k in shape_counts.keys()]
values = list(shape_counts.values())

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.ylabel("文件数")
plt.title("不同 NIfTI shape 分布")
plt.tight_layout()

out_path = "nii_shape_distribution.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"✅ 直方图已保存到 {out_path}")

# 输出异常文件
if len(shape_counts) > 1:
    main_shape = shape_counts.most_common(1)[0][0]  # 出现最多的 shape
    bad_files = [f"{k}\t{v}" for k, v in shapes.items() if v != main_shape]
    with open("bad_shapes.txt", "w") as f:
        f.write("\n".join(bad_files))
    print(f"⚠️ 异常 shape 文件已写入 bad_shapes.txt")
