# build_faiss.py
from sentence_transformers import SentenceTransformer
import faiss
import json

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load text and chunk
def split_text(text, max_len=500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_len:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

with open("tuyensinh.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

chunks = split_text(raw_text)
embeddings = model.encode(chunks)

# Save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "tuyensinh.index")

# Lưu text lại để map
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
