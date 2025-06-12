import json
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np

# 重新建立與儲存時相同的 embedding 模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

# 載入資料庫（資料夾名稱與你當初 save_local 的一致）
db = FAISS.load_local("./vector_db/fpc_esg", embedding_model, allow_dangerous_deserialization=True)


# 5. 查詢向量資料庫
queries = [
    "PCR 材料使用量與 CO2 減排的關聯",
    "PCR 再生塑料對碳排的影響",
    "回收材料對 ESG 減碳目標貢獻",
    "再生原料在減碳方面的績效"
]

results = []
for q in queries:
    results.extend(db.similarity_search(q, k=2))

def dedup_documents(documents):
    seen_hashes = set()
    unique_docs = []
    for doc in documents:
        content_hash = hashlib.md5(doc.page_content.strip().encode("utf-8")).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    return unique_docs

def rerank_documents(query: str, documents: list[Document], embedder) -> list[tuple[Document, float]]:
    query_embedding = embedder.embed_query(query)
    doc_embeddings = [embedder.embed_query(doc.page_content) for doc in documents]

    scores = [
        np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embeddings
    ]

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked


# 去除重複內容
unique_results = dedup_documents(results)

# 重新用第一個 query rerank（你可以改成用綜合 query 向量）
reranked = rerank_documents("PCR 材料使用量與 CO2 減排的關聯", unique_results, embedding_model)

# 顯示前 5 筆

results_data = [
    {
        "content": doc.page_content,
        "page": doc.metadata.get("page"),
        "page_label": doc.metadata.get("page_label"),
        "score": float(score)  # 確保 score 可序列化
    }
    for doc, score in reranked
]

with open("reranked_results.json", "w", encoding="utf-8") as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)
