<<<<<<< HEAD
# 模型测试代码使用说明

## 文件说明

1. **`test_model.py`** - 命令行版本的测试脚本
2. **`test_model_notebook.py`** - Notebook版本的测试代码
3. **`README_test.md`** - 使用说明文档

## 使用方法

### 方法1：命令行版本

```bash
python test_model.py \
    --model_path /path/to/your/model.pth \
    --pos_test_path /path/to/positive_test.npy \
    --neg_test_path /path/to/negative_test.npy \
    --batch_size 32 \
    --save_predictions \
    --predictions_path test_predictions.npy \
    --model_type efficient_window_moe
```

### 方法2：Notebook版本

1. 将 `test_model_notebook.py` 的内容复制到你的 notebook cell 中
2. 修改配置参数：
   ```python
   # 修改这些路径为你的实际路径
   MODEL_PATH = '/path/to/your/model.pth'
   POS_TEST_PATH = '/path/to/positive_test.npy'
   NEG_TEST_PATH = '/path/to/negative_test.npy'
   ```
3. 确保你的模型类（如 `EfficientWindowMoEProteinClassifier`）已经定义
4. 运行 `test_model()` 函数

## 需要修改的地方

### 1. 模型类导入
在测试代码中，你需要导入你的模型类：

```python
# 如果你有单独的模型文件，添加导入语句
from your_model_file import EfficientWindowMoEProteinClassifier
# 或者
from your_model_file import ProteinClassifier
```

### 2. 模型创建
根据你使用的模型类型，修改模型创建部分：

```python
# 对于 EfficientWindowMoEProteinClassifier
model = EfficientWindowMoEProteinClassifier(
    embed_dim=1152, hidden_dim=256, num_classes=2, 
    window_size=30, num_windows=6, dropout=0.5, load_balance_weight=0.01
).to(device)

# 对于 ProteinClassifier
model = ProteinClassifier().to(device)
```

### 3. 数据路径
修改为你的实际数据路径：

```python
MODEL_PATH = '/exp_data/sjx/star/efficient_window_moe_checkpoints/best_moe_model.pth'
POS_TEST_PATH = '/exp_data/sjx/star/first_data/ESM-embedding/positive_test_embedding.npy'
NEG_TEST_PATH = '/exp_data/sjx/star/gan_data/negative_test_all_combined.npy'
```

## 输出结果

测试完成后会输出：
- **控制台**：ACC、PRE、REC、F1、AUC、MCC 等指标
- **文本文件**：`test_results.txt` 包含详细结果
- **预测文件**：`test_predictions.npy`（如果启用）包含预测结果、标签和概率

## 注意事项

1. **确保模型架构一致**：测试时使用的模型架构必须与训练时完全一致
2. **数据格式**：确保测试数据的格式与训练数据一致
3. **内存使用**：代码使用内存映射加载数据，适合大文件
4. **设备兼容**：代码会自动检测并使用可用的 GPU/CPU

## 常见问题

### Q: 模型加载失败怎么办？
A: 检查模型路径是否正确，确保模型架构与训练时一致

### Q: 如何修改批次大小？
A: 修改 `BATCH_SIZE` 参数或使用 `--batch_size` 命令行参数

### Q: 如何保存预测结果？
A: 设置 `SAVE_PREDICTIONS = True` 或使用 `--save_predictions` 参数 
=======
# AcidNetPro
>>>>>>> 670007e5b0ec88f9355cb492d0e67791b4ba3f7f
