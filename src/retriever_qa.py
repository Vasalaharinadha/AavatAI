import faiss
import sqlite3
import numpy as np
from config import INDEX_PATH, SQLITE_DB, TOP_K, LOCAL_GEN_MODEL, MAX_GEN_TOKENS, DATA_DIR
from embedder import embed_texts
from transformers import pipeline
import textwrap

# load faiss index
def load_index():
    idx = faiss.read_index(str(INDEX_PATH))
    return idx

def query_faiss(query, k=TOP_K):
    q_emb = embed_texts([query]).astype("float32")
    idx = load_index()
    D, I = idx.search(q_emb, k)
    scores = D[0].tolist()
    ids = I[0].tolist()
    # map indices -> metadata via sqlite (we used consecutive c0..cN ids)
    conn = sqlite3.connect(str(SQLITE_DB))
    cur = conn.cursor()
    results = []
    for score, vec_idx in zip(scores, ids):
        if vec_idx < 0:
            continue
        chunk_id = f"c{vec_idx}"
        cur.execute("SELECT text, source, page, chunk_index FROM chunks WHERE chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        if row:
            text, source, page, chunk_index = row
            results.append({"text": text, "source": source, "page": page, "score": float(score)})
    conn.close()
    return results

# local generator (Flan-T5)
_gen = None
def get_generator():
    global _gen
    if _gen is None:
        _gen = pipeline("text2text-generation", model=LOCAL_GEN_MODEL, max_length=MAX_GEN_TOKENS, truncation=True)
    return _gen

def generate_answer(query, retrieved):
    # create prompt with numbered citations
    pieces = []
    for i, r in enumerate(retrieved):
        pieces.append(f"[{i+1}] {r['source']}::page-{r['page']}\n{r['text']}")
    context = "\n\n---\n\n".join(pieces)
    prompt = textwrap.dedent(f"""
    You are an assistant. Use the context below and answer the question. Cite sources using bracket numbers like [1].
    If the answer isn't contained in the context, say "I don't know from the provided documents."

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """).strip()
    gen = get_generator()
    out = gen(prompt, max_length=MAX_GEN_TOKENS, do_sample=False)
    return out[0]["generated_text"]

def answer(query, k=TOP_K):
    retrieved = query_faiss(query, k=k)
    answer_text = generate_answer(query, retrieved)
    return {"answer": answer_text, "retrieved": retrieved}
