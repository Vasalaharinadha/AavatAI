import streamlit as st
import fitz  # PyMuPDF
from config import DATA_DIR, PDF_FILENAME
from retriever_qa import answer
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw

def highlight_text_on_page(pdf_path, page_num, text_to_highlight):
    """Highlight text on a PDF page and return as PIL Image"""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]  # pages are 0-indexed
    
    # Search for the text
    text_instances = page.search_for(text_to_highlight[:50])  # Search first 50 chars to avoid long searches
    
    # Create a highlight for each instance found
    for inst in text_instances:
        highlight = page.add_highlight_annot(inst)
        highlight.update()
    
    # Convert to image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Draw red rectangle around the text
    draw = ImageDraw.Draw(img)
    for inst in text_instances:
        # Convert PDF coordinates to image coordinates
        x0, y0 = inst.tl * 2  # Scale by 2 because of the matrix above
        x1, y1 = inst.br * 2
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    
    return img

st.set_page_config(page_title="FAISS RAG Chat", layout="wide")
st.title("Retrieval-Augmented Generation System for PDF Q&A")

# # st.sidebar.markdown("**Document**")
# st.sidebar.write(PDF_FILENAME)

query = st.text_input("Ask a question about the document:")
if st.button("Search") and query.strip():
    with st.spinner("Retrieving and generating..."):
        out = answer(query)
    
    # st.subheader("Answer")
    # st.write(out["answer"])
    
    st.subheader("Top retrieved chunks (evidence)")
    for i, r in enumerate(out["retrieved"], start=1):
        st.markdown(f"**[{i}] {r['source']} — page {r['page']} — score {r['score']:.3f}**")
        st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
        
        # show page image (small)
        pdf_path = Path(DATA_DIR) / PDF_FILENAME
        try:
            # Highlight the relevant text on the page
            img = highlight_text_on_page(pdf_path, r['page'], r["text"][:200])  # Use first 200 chars for matching
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.image(buf)
        except Exception as e:
            st.write("(Could not render page image)", e)
