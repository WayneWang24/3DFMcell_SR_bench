import os
import tifffile
import matplotlib.pyplot as plt
from collections import Counter

# 数据目录
data_dir = "/root/autodl-tmp/datasets/beforedeconv/tifR/"

shapes = []
for fname in os.listdir(data_dir):
    if fname.lower().endswith((".tif", ".tiff")):
        fpath = os.path.join(data_dir, fname)
        try:
            arr = tifffile.imread(fpath)
            shapes.append(arr.shape)
        except Exception as e:
            print(f"读取失败: {fname}, 错误: {e}")

# 统计不同 shape 的数量
shape_counts = Counter(shapes)

print("shape 统计：")
for shp, count in shape_counts.items():
    print(f"{shp}: {count} 个文件")

# 画直方图（x轴是 shape，y轴是数量）
labels = [str(k) for k in shape_counts.keys()]
values = list(shape_counts.values())

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.ylabel("文件数")
plt.title("不同 tiff shape 的分布")
plt.tight_layout()
# plt.show()
out_path = "shape_distribution.png"
plt.savefig(out_path, dpi=300)   # 保存成高清图
plt.close()

print(f"✅ 直方图已保存到 {out_path}")