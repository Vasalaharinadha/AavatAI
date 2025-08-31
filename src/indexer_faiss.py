import os
from pathlib import Path
import faiss
import sqlite3
import json
from config import DATA_DIR, PDF_FILENAME, PERSIST_DIR, INDEX_PATH, SQLITE_DB, CHUNK_SIZE, CHUNK_OVERLAP
from utils import extract_pdf_pages, chunk_text
from embedder import embed_texts
import numpy as np

PERSIST_DIR.mkdir(parents=True, exist_ok=True)

def init_sqlite(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            text TEXT,
            source TEXT,
            page INTEGER,
            chunk_index INTEGER
        )""")
    conn.commit()
    return conn

def build_index():
    pdf_path = DATA_DIR / PDF_FILENAME
    pages = extract_pdf_pages(pdf_path)
    # prepare chunks list
    metadatas = []
    texts = []
    for p in pages:
        page_no = p["page_no"]
        page_text = p["text"]
        page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(page_chunks):
            texts.append(chunk)
            metadatas.append({"source": pdf_path.name, "page": page_no, "chunk_index": idx})
    if len(texts) == 0:
        print("No text extracted. Check PDF or OCR.")
        return

    # embeddings
    embs = embed_texts(texts).astype("float32")
    dim = embs.shape[1]

    # initialize FAISS index (cosine via inner product on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[index] Stored FAISS index at {INDEX_PATH} with {index.ntotal} vectors")

    # persist metadata in sqlite
    conn = init_sqlite(SQLITE_DB)
    cur = conn.cursor()
    for i, meta in enumerate(metadatas):
        chunk_id = f"c{i}"
        text = texts[i]
        cur.execute("INSERT OR REPLACE INTO chunks (chunk_id, text, source, page, chunk_index) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, text, meta["source"], meta["page"], meta["chunk_index"]))
    conn.commit()
    conn.close()
    print(f"[index] Stored metadata for {len(metadatas)} chunks in {SQLITE_DB}")

if __name__ == "__main__":
    build_index()
