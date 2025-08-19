import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_path = r".\acidoSplit.csv"

df = pd.read_csv(csv_path)


df['seq_len'] = df['sequence'].apply(len)


print("====== Sequence length distribution statistics ======")
print(df['seq_len'].describe())
print("\n90% The length of the sequence is less than：", df['seq_len'].quantile(0.90))
print("95% The length of the sequence is less than：", df['seq_len'].quantile(0.95))
print("99% The length of the sequence is less than：", df['seq_len'].quantile(0.99))


plt.figure(figsize=(10, 6))
sns.histplot(df['seq_len'], bins=100, kde=True, color='skyblue')
plt.title("Sequence Length Distribution (zoomed in)")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.xlim(0, 1500)  
plt.grid(True)
plt.tight_layout()
plt.savefig(csv_path.replace("acidoSplit.csv", "seq_length_zoomed.png"))
plt.show()


total_counts = df['label'].value_counts().rename({1: 'Positive', 0: 'Negative'})
set_label_counts = df.groupby(['set', 'label']).size().unstack(fill_value=0)
set_label_counts.columns = ['Negative', 'Positive']

print("====== Total sample count (all) ======")
print(total_counts)
print("\n====== Positive and negative sample counts in each dataset ======")
print(set_label_counts)

print("\n====== Sequence length statistics ======")
print(df['seq_len'].describe())
