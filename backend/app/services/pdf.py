#extract text from pdf bytes


# app/services/pdf.py
import io
from ..config import settings
import fitz # PyMuPDF
# Fallback: pdfminer.six (robust)
from pdfminer.high_level import extract_text as pm_extract


def extract_text(data: bytes) -> str:
    use_pymupdf = settings.USE_PYMUPDF
    if use_pymupdf:
        try:
             # PyMuPDF
            doc = fitz.open(stream=data, filetype="pdf")
            # limit pages to avoid pathological files
            max_pages = int(os.getenv("PDF_MAX_PAGES", "200"))
            texts = []
            for i, page in enumerate(doc):
                if i >= max_pages: break
                texts.append(page.get_text("text"))
            return "\n".join(texts)
        except Exception as e:
            # fallback below
            pass

    # fallback: pdfminer.six (slower but robust)
    try:
        #pdfminer
        return pm_extract(io.BytesIO(data))
    except Exception as e:
        raise RuntimeError(f"PDF parse failed: {e}")
