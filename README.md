# AcidNetPro - Deep Learning Framework for Acidophilic Protein Prediction



## Project Overview



**AcidNetPro** is a deep learning-based framework for predicting acidophilic proteins. It integrates multiple advanced machine learning technologies, including:

- **Transformer Architecture**: Sequence modeling based on attention mechanisms
- **Mixture of Experts (MoE)**: Enhances model capacity and specialization
- **Generative Adversarial Network (GAN)**: Data augmentation and feature learning
- **ESM Pretrained Embeddings**: Leverages prior knowledge from protein language models

## Project Structure



```
├── README.md                    # Project description
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment configuration
├── .gitignore                   # Git ignore settings
├── model.ipynb                  # Main model training notebook
├── dataselect.ipynb             # Data selection and preprocessing
├── DCGAN-GP.py                  # DCGAN with gradient penalty
│
├── datadeal/                    # Data processing module
│   ├── datastatics.py           # Data statistics analysis
│   └── ESMC.py                  # ESM embedding processing
│
├── Raw Data/                    # Raw data
│   ├── acidoSplit.csv           # Acidophilic protein dataset
│   ├── positive_train.fasta     # Positive training set
│   ├── positive_test.fasta      # Positive test set
│   ├── negative_train.fasta     # Negative training set
│   └── negative_test.fasta      # Negative test set
│
└── experiments/                 # Experiment-related code
    ├── moe_anlysis/             # MoE model analysis
    ├── gan_anlysis/             # GAN model analysis
    ├── qianruxuanze/            # Embedding selection
    ├── Comparative experiment/  # Comparative experiments
    ├── xiaorongshiyan/          # Ablation experiments
    └── zhongjianceng_t-sne/     # t-SNE visualization of intermediate layers
```



## Core Features



### 1. Model Architecture



- **TransformerMoE**: Transformer-based mixture-of-experts model
- **Multi-head Attention**: Captures long-range dependencies in sequences

### 2. Data Processing



- **FASTA Format Support**: Standard format for protein sequences
- **ESM Embedding Generation**: Using pretrained protein language models
- **Sequence Length Normalization**: Handles variable-length sequences
- **Data Augmentation**: GAN-based negative sample generation

### 3. Analysis Tools



- **Attention Visualization**: Understand important regions in sequences
- **Expert Utilization Analysis**: MoE expert activation patterns
- **Amino Acid Correlation Analysis**: Study correlations between residues
- **Sankey Diagrams**: Visualize information flow

## Quick Start



### Environment Setup



```
bash
# Using conda
conda env create -f environment.yml
conda activate acidnetpro

# Or using pip
pip install -r requirements.txt
```



### Data Preparation



Make sure your data paths are correctly set:

```
# Training data paths
train_pos = '/path/to/positive_train_embedding.npy'  # Positive training samples
train_neg = '/path/to/negative_train_combined.npy'   # Negative training samples (including GAN-generated)

# Testing data paths
test_pos = '/path/to/positive_test_embedding.npy'    # Positive test samples
test_neg = '/path/to/negative_test_embedding.npy'    # Negative test samples
```



### Usage



#### 1. Use Jupyter Notebook



We recommend using `model.ipynb` in the project root for training and testing.

Execute all cells in order:

1. Import libraries and set random seed
2. Define dataset class and model architecture
3. Train model(standard or 10-fold cross-validation)
4. Evaluate model

#### 2. Standard Training



```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ProteinNPYDataset(train_pos, train_neg)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

model = TransformerMoE(
    d_model=1152, nhead=8, d_ff=2048, num_layers=4,
    num_experts=30, k=3, dropout=0.1, noisy_std=1.0, num_classes=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()  # Mixed precision training

epochs = 10
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
```



#### 3. 10-Fold Cross Validation



#### 4. Model Testing and Evaluation



```
model = TransformerMoE(
    d_model=1152, nhead=8, d_ff=2048, num_layers=4,
    num_experts=30, k=3, dropout=0.1, noisy_std=1.0, num_classes=2
).to(device)

model.load_state_dict(torch.load('path/to/your/model.pth', map_location=device))

test_dataset = ProteinNPYDataset(test_pos, test_neg)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

eval_model_extended(model, test_loader, device)
```



### Model Parameter Description



| Parameter      | Default | Description                              |
| -------------- | ------- | ---------------------------------------- |
| `d_model`      | 1152    | Model dimension (matches ESM embeddings) |
| `nhead`        | 8       | Number of attention heads                |
| `d_ff`         | 2048    | Feed-forward hidden layer size           |
| `num_layers`   | 4       | Number of Transformer layers             |
| `num_experts`  | 30      | Number of MoE experts                    |
| `k`            | 3       | Top-k routing (number of experts used)   |
| `dropout`      | 0.1     | Dropout rate                             |
| `noisy_std`    | 1.0     | Standard deviation of noisy gating       |
| `batch_size`   | 64      | Batch size                               |
| `lr`           | 2e-4    | Learning rate                            |
| `weight_decay` | 1e-4    | Weight decay for regularization          |

### Evaluation Metrics



The model supports full binary classification metrics:

- **ACC**: Accuracy – proportion of correct predictions
- **PRE**: Precision – proportion of true positives among predicted positives
- **REC/Sn**: Recall / Sensitivity – proportion of true positives found
- **Sp**: Specificity – proportion of true negatives found
- **F1**: Harmonic mean of precision and recal
- **MCC**: Matthews Correlation Coefficient – measures correlation between predictions and ground truth, range [-1, 1]
- **AUC**: Area Under ROC Curve – evaluates overall classification ability
- **AUPRC**: Area Under Precision-Recall Curve
