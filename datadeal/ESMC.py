import os
import numpy as np
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein

def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def save_embeddings(embeddings, output_path):
    np.save(output_path, embeddings)
    print(f"✅ Saved embeddings to: {output_path}")
def pad_embedding(emb, max_len, emb_dim):
    padded = np.zeros((max_len, emb_dim), dtype=np.float32)
    length = emb.shape[0]
    padded[:length, :] = emb
    return padded
def process_fasta_with_model(model: ESMC, fasta_path: str, output_path: str, max_aa_len: int = 300, embedding_dim: int = 1152):
    sequences = read_fasta(fasta_path)
    all_embeddings = []

    max_token_len = max_aa_len + 2  # 预测特殊token数

    for i, seq in enumerate(sequences):
        seq = seq[:max_aa_len]

        protein = ESMProtein(sequence=seq)
        input_ids = model._tokenize([protein.sequence])

        if input_ids.shape[1] > max_token_len:
            input_ids = input_ids[:, :max_token_len]

        output = model(input_ids)
        emb = output.embeddings.float().detach().cpu().numpy()[0]

        emb = emb[1:-1, :]  # 去起始和终止token

        emb_padded = pad_embedding(emb, max_aa_len, embedding_dim)

        assert emb_padded.shape == (max_aa_len, embedding_dim), f"Embedding padded shape {emb_padded.shape} 不符合预期"

        all_embeddings.append(emb_padded)

        print(f"[{i+1}/{len(sequences)}] {os.path.basename(fasta_path)} - Original emb shape: {emb.shape}, Padded emb shape: {emb_padded.shape}")

    all_embeddings_array = np.array(all_embeddings)
    save_embeddings(all_embeddings_array, output_path)

def main():
    input_dir = "/exp_data/sjx/star/first_data/shisuandanbai/"
    output_dir = "/exp_data/sjx/star/first_data/ESM-embedding/"
    max_aa_len = 300
    embedding_dim = 1152

    os.makedirs(output_dir, exist_ok=True)

    model = ESMC.from_pretrained("esmc_600m").to("cuda")  # 或者 .to("cpu")

    for filename in os.listdir(input_dir):
        if filename.endswith(".fasta"):
            fasta_path = os.path.join(input_dir, filename)
            out_name = os.path.splitext(filename)[0] + "_embedding.npy"
            output_path = os.path.join(output_dir, out_name)

            if os.path.exists(output_path):
                print(f"Skipping {filename} because {out_name} already exists.")
                continue  # 文件存在，跳过

            print(f"Processing {filename} ...")
            process_fasta_with_model(model, fasta_path, output_path, max_aa_len=max_aa_len, embedding_dim=embedding_dim)

if __name__ == "__main__":
    main()
