"""Extract FMR PDFs into page-aware chunks and store them in ChromaDB."""

from __future__ import annotations

""" 
1_ Extract page number, Page text (one page each)
2_ Clean page page text, for better performance of chromadb and Ollama
3_ Chunking/Cutting give texts (overlapping chunks for retrieval) into smaller pieces for better indexing performance.
 _ As AI performance improves, otherwise sending full text pages to AI create noise.
 _ CHUNK_SIZE = 1400	Approximate size of one piece.
 _ CHUNK_OVERLAP = 180	Approximate overlap size between chunks.
 _ These settings can be adjusted based on the desired granularity of text chunks.
 _ start = end - overlap


"""

import argparse
import re
from pathlib import Path

import chromadb
import fitz #  The PyMuPDF library, imported as fitz
from dotenv import load_dotenv

from embeddings import get_embedding_function

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "fmr_documents"
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180


def report_metadata(path: Path) -> tuple[str, str, int]:
    """Read the report category, month, and year from a PDF filename."""
    name = path.name.lower()
    category = "Islamic" if "islamic" in name else "Conventional"
    month_match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)", name)
    year_match = re.search(r"20\d{2}", name)
    return category, month_match.group(1).title() if month_match else "Unknown", int(year_match.group()) if year_match else 0


def infer_section(text: str) -> str:
    """Assign a simple section label by matching known report headings."""
    lowered = text.lower()
    for keyword, section in (
        ("asset allocation", "asset_allocation"),
        ("top holdings", "top_holdings"),
        ("fund manager commentary", "fund_manager_commentary"),
        ("performance", "performance"),
        ("capital market review", "capital_market_review"),
        ("ceo's write-up", "ceo_write_up"),
    ):
        if keyword in lowered:
            return section
    return "fund_report_page"


def chunks(text: str) -> list[str]:
    """Normalize page text and split it into overlapping retrieval chunks."""
    cleaned = re.sub(r"\s+", " ", text).strip()     # clean pdf text
    if not cleaned:
        return []
    result: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + CHUNK_SIZE, len(cleaned))
        result.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = end - CHUNK_OVERLAP
    return result


def ingest(reset: bool = False) -> int:
    """Extract all project PDFs and upsert their chunks into ChromaDB."""
    load_dotenv(BASE_DIR / ".env")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=get_embedding_function())

    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    ids: list[str] = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        category, month, year = report_metadata(pdf_path)
        with fitz.open(pdf_path) as pdf:                        # opens the PDF file for reading
            for page_number, page in enumerate(pdf, start=1):   # loop through pages in the PDF
                                                                # page 11 becomes something like page_number = 11 and page_text = "NBP ISLAMIC ENERGY FUND..."
                page_text = page.get_text("text")               # get pdf text from the current page
                for chunk_number, chunk in enumerate(chunks(page_text)):
                    ids.append(f"{pdf_path.stem}:{page_number}:{chunk_number}")
                    documents.append(chunk)
                    metadatas.append(                        # To keep the track of where in the PDF this chunk comes from
                        {                                                                
                            "fund_name": "See page text",    # The fund name is extracted from the page text itself
                            "category": category,            # islamic/ conventional
                            "report_month": month,           # july, august, etc.
                            "report_year": year,              # 2024, 2025, etc.
                            "section": infer_section(chunk),  # inferred section of the report
                            "source_page": page_number,       # page number in the PDF
                            "source_file": pdf_path.name,
                        }
                    )

    if not documents:
        raise RuntimeError(f"No PDF content found in {PDF_DIR}")
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(documents)} chunks from {len(list(PDF_DIR.glob('*.pdf')))} PDFs into {CHROMA_DIR}")
    return len(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Replace the existing collection")
    args = parser.parse_args()
    ingest(reset=args.reset)
