import json
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 重新建立與儲存時相同的 embedding 模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

# 載入資料庫（資料夾名稱與你當初 save_local 的一致）
db = FAISS.load_local("./vector_db/fpc_esg", embedding_model, allow_dangerous_deserialization=True)


# === 5. 查詢向量資料庫 ===
queries = [
    "PCR 材料使用量與 CO2 減排的關聯",
    "PCR 再生塑料對碳排的影響",
    "回收材料對 ESG 減碳目標貢獻",
    "再生原料在減碳方面的績效"
]

results = []
for q in queries:
    results.extend(db.similarity_search(q, k=2))

# === 去除重複內容 ===
def dedup_documents(documents):
    seen_hashes = set()
    unique_docs = []
    for doc in documents:
        content_hash = hashlib.md5(doc.page_content.strip().encode("utf-8")).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    return unique_docs

unique_results = dedup_documents(results)

# === 6. 串接 bge-reranker-base 進行 rerank ===
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")

def rerank_bge_reranker(query, docs, top_k=5):
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = reranker_tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.view(-1)
    scores = scores.numpy()
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

reranked = rerank_bge_reranker("PCR 材料使用量與 CO2 減排的關聯", unique_results, top_k=5)

# === 儲存結果 ===
results_data = [
    {
        "content": doc.page_content,
        "page": doc.metadata.get("page"),
        "page_label": doc.metadata.get("page_label"),
        "score": float(score)
    }
    for doc, score in reranked
]

output_path = "bge_reranked_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)