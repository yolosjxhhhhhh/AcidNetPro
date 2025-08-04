# AcidNetPro - 酸性蛋白质预测深度学习框架

## 项目简介

AcidNetPro 是一个基于深度学习的酸性蛋白质预测框架，集成了多种先进的机器学习技术，包括：

- **Transformer架构**：基于注意力机制的序列建模
- **专家混合模型(MoE)**：提高模型容量和专业化能力
- **生成对抗网络(GAN)**：数据增强和特征学习
- **ESM预训练嵌入**：利用蛋白质语言模型的先验知识

## 项目结构

```
e:\模型\star - 副本 - 副本/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包
├── environment.yml              # Conda环境配置
├── .gitignore                   # Git忽略文件配置
├── model.ipynb                  # 主要模型训练笔记本
├── dataselect.ipynb            # 数据选择和预处理
├── DCGAN-GP.py                 # DCGAN梯度惩罚实现
│
├── datadeal/                   # 数据处理模块
│   ├── datastatics.py         # 数据统计分析
│   └── ESMC.py                # ESM嵌入处理
│
├── Raw Data/                   # 原始数据
│   ├── acidoSplit.csv         # 酸性蛋白质数据集
│   ├── positive_train.fasta   # 正样本训练集
│   ├── positive_test.fasta    # 正样本测试集
│   ├── negative_train.fasta   # 负样本训练集
│   └── negative_test.fasta    # 负样本测试集
│
└── experiments/               # 实验相关代码
    ├── moe_anlysisi/         # MoE模型分析
    ├── gan_anlysis/          # GAN模型分析
    ├── qianruxuanze/         # 嵌入选择实验
    ├── add_position/         # 位置编码实验
    ├── Comparative experiment/ # 对比实验
    ├── xiaorongshiyan/       # 消融实验
    └── zhongjianceng_t-sne/  # 中间层t-SNE可视化
```

## 核心功能

### 1. 模型架构

- **TransformerMoE**: 基于Transformer的专家混合模型
- **多头注意力机制**: 捕获序列中的长距离依赖关系

### 2. 数据处理

- **FASTA格式支持**: 标准蛋白质序列格式
- **ESM嵌入生成**: 利用预训练蛋白质语言模型
- **序列长度标准化**: 支持可变长度序列处理
- **数据增强**: 基于GAN的负样本生成

### 3. 分析工具

- **注意力可视化**: 理解模型关注的关键区域
- **专家使用分析**: MoE模型中各专家的激活模式
- **氨基酸相关性分析**: 不同氨基酸间的关联性研究
- **桑基图可视化**: 信息流动路径可视化

## 快速开始

### 环境配置

```bash
# 使用conda创建环境
conda env create -f environment.yml
conda activate acidnetpro

# 或使用pip安装依赖
pip install -r requirements.txt
```

### 数据准备

确保您的数据文件路径正确设置：

```python
# 训练数据路径
train_pos = '/path/to/positive_train_embedding.npy'  # 正样本训练集
train_neg = '/path/to/negative_train_combined.npy'   # 负样本训练集(含GAN生成)

# 测试数据路径  
test_pos = '/path/to/positive_test_embedding.npy'    # 正样本测试集
test_neg = '/path/to/negative_test_embedding.npy'    # 负样本测试集
```

### 使用方法

#### 1. 直接使用Jupyter Notebook

推荐使用项目根目录下的 `model.ipynb` 进行模型训练和测试

按顺序执行所有cells：
1. **导入库和设置随机种子**
2. **定义数据集类和模型架构**
3. **训练模型** (可选择标准训练或10折交叉验证)
4. **模型评估**

#### 2. 标准训练模式

```python
# 在notebook中执行以下代码块进行标准训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据加载器
train_dataset = ProteinNPYDataset(train_pos, train_neg)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 初始化模型
model = TransformerMoE(
    d_model=1152, nhead=8, d_ff=2048, num_layers=4, 
    num_experts=30, k=3, dropout=0.1, noisy_std=1.0, num_classes=2
).to(device)

# 训练配置
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()  # 混合精度训练

# 训练10个epochs
epochs = 10
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
```

#### 3. 10折交叉验证

#### 4. 模型测试和评估

```python
# 加载训练好的模型
model = TransformerMoE(
    d_model=1152, nhead=8, d_ff=2048, num_layers=4, 
    num_experts=30, k=3, dropout=0.1, noisy_std=1.0, num_classes=2
).to(device)

model.load_state_dict(torch.load('path/to/your/model.pth', map_location=device))

# 创建测试数据加载器
test_dataset = ProteinNPYDataset(test_pos, test_neg)  
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# 扩展评估 (包含更多指标)
eval_model_extended(model, test_loader, device)
```

### 模型参数说明

| 参数           | 默认值 | 说明                        |
| -------------- | ------ | --------------------------- |
| `d_model`      | 1152   | 模型维度，匹配ESM嵌入维度   |
| `nhead`        | 8      | 多头注意力头数              |
| `d_ff`         | 2048   | 前馈网络隐藏层维度          |
| `num_layers`   | 4      | Transformer层数             |
| `num_experts`  | 30     | MoE专家数量                 |
| `k`            | 3      | Top-k路由，每次选择的专家数 |
| `dropout`      | 0.1    | Dropout概率                 |
| `noisy_std`    | 1.0    | 门控噪声标准差              |
| `batch_size`   | 64     | 批处理大小                  |
| `lr`           | 2e-4   | 学习率                      |
| `weight_decay` | 1e-4   | 权重衰减                    |

### 评估指标

模型提供完整的二分类评估指标：

- **ACC**: 准确率
- **PRE**: 精确率  
- **REC/Sn**: 召回率/敏感性
- **Sp**: 特异性
- **F1**: F1分数
- **MCC**: 马修斯相关系数
- **AUC**: ROC曲线下面积
- **AUPRC**: PR曲线下面积
