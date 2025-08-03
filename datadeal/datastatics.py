import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 路径
csv_path = r"E:\模型\star\first_data\acidoSplit.csv"

# 读取数据
df = pd.read_csv(csv_path)

# 添加序列长度列
df['seq_len'] = df['sequence'].apply(len)

# 输出统计量
print("====== 序列长度分布统计 ======")
print(df['seq_len'].describe())
print("\n90% 的序列长度小于：", df['seq_len'].quantile(0.90))
print("95% 的序列长度小于：", df['seq_len'].quantile(0.95))
print("99% 的序列长度小于：", df['seq_len'].quantile(0.99))

# 绘图，限制横坐标范围
plt.figure(figsize=(10, 6))
sns.histplot(df['seq_len'], bins=100, kde=True, color='skyblue')
plt.title("Sequence Length Distribution (zoomed in)")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.xlim(0, 1500)  # <- 只显示0~1500的序列长度
plt.grid(True)
plt.tight_layout()
plt.savefig(csv_path.replace("acidoSplit.csv", "seq_length_zoomed.png"))
plt.show()

# ===============================
# 2. 样本数量统计
# ===============================
total_counts = df['label'].value_counts().rename({1: 'Positive', 0: 'Negative'})
set_label_counts = df.groupby(['set', 'label']).size().unstack(fill_value=0)
set_label_counts.columns = ['Negative', 'Positive']

print("====== 样本总数（全部） ======")
print(total_counts)
print("\n====== 各数据集中正负样本数量 ======")
print(set_label_counts)

# ===============================
# 3. 输出统计摘要
# ===============================
print("\n====== 序列长度统计信息 ======")
print(df['seq_len'].describe())
