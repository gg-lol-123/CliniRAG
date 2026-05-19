# src/document_processor.py

import os
import re
import json
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Handles:
    1. PDF loading
    2. Text cleaning
    3. Smart chunking
    4. Metadata attachment
    5. Caching processed chunks
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        cache_path: str = "data/processed/cleaned_chunks.json"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_path = cache_path

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def clean_text(self, text: str) -> str:

        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\s+", " ", text)

        # Remove page numbers
        text = re.sub(
            r"Page\s+\d+",
            "",
            text,
            flags=re.IGNORECASE
        )

        # Remove standalone page digits
        text = re.sub(
            r"\b\d+\b(?=\s*$)",
            "",
            text
        )

        # Remove repeated guideline headers
        text = re.sub(
            r"ICMR Guidelines for Management of Type 2 Diabetes 2018",
            "",
            text,
            flags=re.IGNORECASE
        )

        # Remove isolated roman numerals
        text = re.sub(
            r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b",
            "",
            text,
            flags=re.IGNORECASE
        )

        return text.strip()

    def is_valid_chunk(self, chunk: str) -> bool:

        chunk = chunk.strip()

        # Too short
        if len(chunk) < 100:
            return False

        # Table of contents / useless chunks
        bad_patterns = [
            "table of contents",
            "contents page",
            "acknowledgement",
            "foreword",
            "preface"
        ]

        lower_chunk = chunk.lower()

        if any(pattern in lower_chunk for pattern in bad_patterns):
            return False

        return True

    def load_pdf(self, file_path: str):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = PyPDFLoader(file_path)

        return loader.load()

    def save_chunks_to_cache(
        self,
        chunks: List[Dict[str, Any]]
    ):

        os.makedirs(
            os.path.dirname(self.cache_path),
            exist_ok=True
        )

        with open(
            self.cache_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"Cached {len(chunks)} chunks to {self.cache_path}"
        )

    def load_chunks_from_cache(self):

        if not os.path.exists(self.cache_path):
            return None

        with open(
            self.cache_path,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from cache")

        return chunks

    def process_single_pdf(
        self,
        file_path: str
    ) -> List[Dict[str, Any]]:

        raw_docs = self.load_pdf(file_path)

        processed_chunks = []

        file_name = os.path.basename(file_path)

        for page_doc in raw_docs:

            page_number = page_doc.metadata.get(
                "page",
                -1
            )

            cleaned_text = self.clean_text(
                page_doc.page_content
            )

            if not cleaned_text:
                continue

            chunks = self.text_splitter.split_text(
                cleaned_text
            )

            for idx, chunk in enumerate(chunks):

                # Skip noisy/useless chunks
                if not self.is_valid_chunk(chunk):
                    continue

                processed_chunks.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "source": file_name,
                            "page": page_number,
                            "chunk_id": (
                                f"{file_name}"
                                f"_page_{page_number}"
                                f"_chunk_{idx}"
                            )
                        }
                    }
                )

        return processed_chunks

    def process_directory(
        self,
        folder_path: str,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        If cache exists → load from cache
        Else → process PDFs + save cache
        """

        if use_cache:

            cached_chunks = self.load_chunks_from_cache()

            if cached_chunks:
                return cached_chunks

        if not os.path.exists(folder_path):

            raise FileNotFoundError(
                f"Folder not found: {folder_path}"
            )

        all_chunks = []

        pdf_files = [
            file for file in os.listdir(folder_path)
            if file.lower().endswith(".pdf")
        ]

        print(f"Found {len(pdf_files)} PDF files.\n")

        for pdf_file in pdf_files:

            file_path = os.path.join(
                folder_path,
                pdf_file
            )

            print(f"Processing: {pdf_file}")

            try:

                chunks = self.process_single_pdf(
                    file_path
                )

                all_chunks.extend(chunks)

                print(
                    f"Created {len(chunks)} chunks "
                    f"from {pdf_file}\n"
                )

            except Exception as e:

                print(
                    f"Error processing {pdf_file}: "
                    f"{str(e)}\n"
                )

        print(f"Total chunks created: {len(all_chunks)}")

        self.save_chunks_to_cache(all_chunks)

        return all_chunks


if __name__ == "__main__":

    processor = DocumentProcessor()

    chunks = processor.process_directory(
        "data/raw_docs",
        use_cache=False
    )

    print("\nSample Chunk:\n")

    if chunks:
        print(chunks[0])