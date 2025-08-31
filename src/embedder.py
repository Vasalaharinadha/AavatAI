from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBED_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_texts(texts):
    """
    Returns L2-normalized embeddings as numpy array (N, D)
    """
    model = get_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-12)
    return embs
