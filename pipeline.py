
import os
import re
from pathlib import Path
from urllib.parse import urlparse
import pymupdf
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_DIR = Path("data/pdfs")
INDEX_DIR = "faiss_index"


def extract_pdfs():
    documents = []
    urls = set()

    for pdf_path in PDF_DIR.glob("*.pdf"):
        print(f"Reading: {pdf_path.name}")

        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page_number,
                            "source_url": "",
                            "content_type": "pdf_text"
                        }
                    ))

                for link in page.get_links():
                    uri = link.get("uri")
                    if uri and urlparse(uri).scheme in ("http", "https"):
                        urls.add(uri)

                urls.update(
                    re.findall(r'https?://[^\s<>"\']+', text)
                )

    return documents, urls


def scrape_url(url):
    try:
        parsed = urlparse(url)

        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            return None

        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "TN-Agri-RAG/1.0"},
            allow_redirects=False
        )
        response.raise_for_status()

        if "text/html" not in response.headers.get(
            "Content-Type", ""
        ).lower():
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(
            ["script", "style", "nav", "footer", "header"]
        ):
            tag.decompose()

        text = soup.get_text(" ", strip=True)

        if len(text) < 100:
            return None

        return Document(
            page_content=text,
            metadata={
                "source_file": parsed.netloc,
                "page": 0,
                "source_url": url,
                "content_type": "webpage"
            }
        )

    except Exception as exc:
        print(f"Could not fetch {url}: {exc}")
        return None


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY in your .env file."
        )

    if not any(PDF_DIR.glob("*.pdf")):
        raise RuntimeError(
            "Add PDF files to data/pdfs before ingestion."
        )

    documents, urls = extract_pdfs()

    print(f"Extracted {len(documents)} PDF pages.")
    print(f"Found {len(urls)} unique URLs.")

    for url in sorted(urls):
        page_doc = scrape_url(url)

        if page_doc:
            documents.append(page_doc)
            print(f"Indexed webpage: {url}")

    if not documents:
        raise RuntimeError(
            "No readable content found. Scanned PDFs need OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save the FAISS index and its document metadata locally.
    vector_store.save_local(INDEX_DIR)

    print(f"Successfully indexed {len(chunks)} chunks.")
    print(f"FAISS index saved to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
