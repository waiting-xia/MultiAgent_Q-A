import chromadb
import jieba
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# ==========================================
# ChromaDB 初始化 & 种子数据
# ==========================================

chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="company_knowledge",
    embedding_function=default_ef,
)

SEED_DOCUMENTS = [
    "员工每年有 15 天带薪年假，病假无限期，且支持远程办公。",
    "项目X目前处于UAT测试阶段，预计下个月上线，目前核心Bug已修复。",
    "公司差旅报销标准：国内每日补助 200 元，住宿上限 500 元。",
]

if collection.count() == 0:
    collection.add(
        documents=SEED_DOCUMENTS,
        metadatas=[{"type": "HR"}, {"type": "Project"}, {"type": "Finance"}],
        ids=["doc_001", "doc_002", "doc_003"],
    )


# ==========================================
# BM25 索引（基于 jieba 中文分词）
# ==========================================

def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))


def _build_bm25_index():
    """从 collection 中读取全部文档，构建 BM25 索引。"""
    all_docs = collection.get(include=["documents"])
    documents = all_docs["documents"] if all_docs and all_docs["documents"] else []
    corpus_tokens = [_tokenize(doc) for doc in documents]
    return documents, BM25Okapi(corpus_tokens) if corpus_tokens else None


_all_documents, bm25_index = _build_bm25_index()


def get_collection():
    """返回原始 ChromaDB collection，供外部使用。"""
    return collection


def hybrid_search(query: str, k: int = 2, bm25_weight: float = 0.3, semantic_weight: float = 0.7) -> list[str]:
    """
    混合检索：BM25 语义检索 + ChromaDB 向量检索。
    返回融合排序后的 top-k 文档内容列表。
    """
    n_docs = len(_all_documents)
    if n_docs == 0:
        return []

    # --- BM25 检索 ---
    bm25_scores = [0.0] * n_docs
    if bm25_index is not None:
        query_tokens = _tokenize(query)
        raw_scores = bm25_index.get_scores(query_tokens)
        max_bm25 = max(raw_scores) if max(raw_scores) > 0 else 1.0
        bm25_norm = [s / max_bm25 for s in raw_scores]
    else:
        bm25_norm = bm25_scores

    # --- ChromaDB 向量检索 ---
    results = collection.query(query_texts=[query], n_results=min(k, n_docs), include=["distances", "documents"])

    semantic_scores = [0.0] * n_docs
    retrieved = results.get("documents")
    distances = results.get("distances")
    if retrieved and retrieved[0] and distances and distances[0]:
        for doc, dist in zip(retrieved[0], distances[0]):
            for i, seed_doc in enumerate(_all_documents):
                if doc == seed_doc:
                    semantic_scores[i] = 1.0 / (1.0 + dist)

    # --- 加权融合 ---
    final_scores = []
    for i in range(n_docs):
        score = bm25_weight * bm25_norm[i] + semantic_weight * semantic_scores[i]
        final_scores.append((score, i))

    final_scores.sort(key=lambda x: x[0], reverse=True)

    return [_all_documents[idx] for _, idx in final_scores[:k]]
