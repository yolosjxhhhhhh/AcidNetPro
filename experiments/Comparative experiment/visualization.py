import numpy as np
import matplotlib.pyplot as plt

# 指标标签
labels = ['ACC', 'PRE', 'REC(Sn)', 'SP', 'F1', 'MCC', 'AUC', 'AUPRC']
num_vars = len(labels)

# 4组指标数据，顺序对应labels
data_100 = [0.9112, 0.9637, 0.9095, 0.9154, 0.9358, 0.7957, 0.9635, 0.9853]
data_200 = [0.9025, 0.9561, 0.9046, 0.8973, 0.9296, 0.7745, 0.9576, 0.9791]
data_300 = [0.9225, 0.9483, 0.9425, 0.8731, 0.9454, 0.8120, 0.9685, 0.9869]  # 最佳
data_400 = [0.8982, 0.9168, 0.9425, 0.7885, 0.9295, 0.7474, 0.9477, 0.9772]

# 角度坐标，闭合环形雷达图
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 为了闭合图形，补充第一个点
def close_data(data):
    return data + data[:1]

data_100 = close_data(data_100)
data_200 = close_data(data_200)
data_300 = close_data(data_300)
data_400 = close_data(data_400)

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制4条曲线
ax.plot(angles, data_100, color='blue', linewidth=1, linestyle='-', label='100 epoch')
ax.fill(angles, data_100, color='blue', alpha=0.1)

ax.plot(angles, data_200, color='green', linewidth=1, linestyle='--', label='200 epoch')
ax.fill(angles, data_200, color='green', alpha=0.1)

# 突出300epoch，颜色加粗、点大、填充深
ax.plot(angles, data_300, color='red', linewidth=2.5, linestyle='-', label='300 epoch (Best)')
ax.fill(angles, data_300, color='red', alpha=0.25)

ax.plot(angles, data_400, color='purple', linewidth=1, linestyle='-.', label='400 epoch')
ax.fill(angles, data_400, color='purple', alpha=0.1)

# 设置雷达图的标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)

# 设置径向刻度
ax.set_rlabel_position(30)
ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.7', '0.8', '0.9', '1.0'], fontsize=10)
ax.set_ylim(0.7, 1.0)

# 图例
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)

plt.title('Performance Metrics Radar Chart\n(Highlight 300 epoch)', fontsize=16, fontweight='bold', y=1.1)

plt.tight_layout()
plt.show()
plt.savefig()