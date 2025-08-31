import fitz
from typing import List, Dict
from pathlib import Path
from config import DATA_DIR, PDF_FILENAME
from PIL import Image
import io

def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    """Return list of pages: {'page_no', 'text', 'blocks'}"""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        blocks_raw = page.get_text("blocks")
        blocks = []
        for b in blocks_raw:
            # Handle different PyMuPDF versions
            try:
                if len(b) >= 6:  # For older versions with block_no
                    x0, y0, x1, y1, btext, block_no = b[:6]
                else:  # For newer versions
                    x0, y0, x1, y1, btext = b[:5]
                blocks.append({"bbox": [x0, y0, x1, y1], "text": btext})
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not process block: {b}")
                continue
                
        pages.append({"page_no": i+1, "text": text, "blocks": blocks})
    return pages

def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    n = len(words)
    if n == 0:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, n, step):
        chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= n:
            break
    return chunks

def render_page_image(pdf_path: Path, page_no: int, zoom: float = 2.0) -> Image.Image:
    """Return PIL Image of a PDF page for UI highlighting."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_no - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img
