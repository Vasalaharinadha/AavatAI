from pathlib import Path

# paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_FILENAME = "policy.pdf"   # change to your file inside data/
PERSIST_DIR = ROOT / "faiss_data"
INDEX_PATH = PERSIST_DIR / "faiss.index"
SQLITE_DB = PERSIST_DIR / "metadata.db"

# chunking
CHUNK_SIZE = 500      # words per chunk (tune)
CHUNK_OVERLAP = 100

# embeddings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# retrieval / gen
TOP_K = 5
LOCAL_GEN_MODEL = "google/flan-t5-base"  
MAX_GEN_TOKENS = 256
