from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np

# 1. 載入 PDF
loader = PyPDFLoader("./data/FPC_2023_ESG-CH.pdf")
pages = loader.load()

# 2. 斷句設定
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "。", "，", " ", ""]
)
chunks = text_splitter.split_documents(pages)

# 3. 使用 Hugging Face 的 embedding 模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
# ✅ 這個模型支援中英文，多語言效果穩定

# 4. 建立向量資料庫
db = FAISS.from_documents(chunks, embedding_model)
db.save_local("./vector_db/fpc_esg")