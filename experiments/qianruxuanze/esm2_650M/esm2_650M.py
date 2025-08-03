import os
import numpy as np
from Bio import SeqIO
import torch
from transformers import AutoTokenizer, AutoModel
model_dir = "/new/LLM/ESM2-650M/"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def read_fasta(file_path):
    records = list(SeqIO.parse(file_path, "fasta"))
    return records
def save_embeddings(embeddings, output_path):
    np.save(output_path, embeddings)
    print(f"✅ Saved embeddings to: {output_path}")
def pad_embedding(emb, max_len, emb_dim):
    padded = np.zeros((max_len, emb_dim), dtype=np.float32)
    length = emb.shape[0]
    padded[:length, :] = emb
    return padded
def process_fasta_with_hf_esm2(model, tokenizer, fasta_path, output_path, max_aa_len=300, embedding_dim=1280, device="cuda"):
    records = read_fasta(fasta_path)
    all_embeddings = []
    for i, record in enumerate(records):
        seq = str(record.seq).replace(" ", "").replace("\n", "")
        if len(seq) < max_aa_len:
            seq = seq + "X" * (max_aa_len - len(seq))
        else:
            seq = seq[:max_aa_len]
        # Tokenize
        inputs = tokenizer(seq, return_tensors="pt", padding="max_length", max_length=max_aa_len, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # outputs.last_hidden_state: (1, max_aa_len, hidden_dim)
        emb = outputs.last_hidden_state[0].cpu().numpy()
        emb_padded = pad_embedding(emb, max_aa_len, embedding_dim)
        assert emb_padded.shape == (max_aa_len, embedding_dim), f"Embedding padded shape {emb_padded.shape} 不符合预期"
        all_embeddings.append(emb_padded)
        print(f"[{i+1}/{len(records)}] {os.path.basename(fasta_path)} - emb shape: {emb.shape}, Padded emb shape: {emb_padded.shape}")
    all_embeddings_array = np.array(all_embeddings)
    save_embeddings(all_embeddings_array, output_path)
    print(f"最终保存的shape: {all_embeddings_array.shape}")
input_map = {
    "positive_train": "/exp_data/sjx/star/first_data/shisuandanbai/positive_train.fasta",
    "positive_test": "/exp_data/sjx/star/first_data/shisuandanbai/positive_test.fasta",
    "negative_train": "/exp_data/sjx/star/first_data/shisuandanbai/negative_train.fasta",
    "negative_test": "/exp_data/sjx/star/first_data/shisuandanbai/negative_test.fasta",
}
output_dir = "/exp_data/sjx/star/experiments/qianruxuanze/esm2_650M/data/"
os.makedirs(output_dir, exist_ok=True)
max_aa_len = 300
embedding_dim = model.config.hidden_size  # 自动获取模型输出维度
for name, fasta_path in input_map.items():
    output_path = os.path.join(output_dir, f"{name}.npy")
    if os.path.exists(output_path):
        print(f"Skipping {name} because {output_path} already exists.")
        continue
    print(f"Processing {name} ...")
    process_fasta_with_hf_esm2(model, tokenizer, fasta_path, output_path, max_aa_len=max_aa_len, embedding_dim=embedding_dim, device=device)