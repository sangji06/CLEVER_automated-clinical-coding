"""Build a FAISS index for KCD code retrieval using SapBERT embeddings.

This script is a local/VS Code-friendly version of the Colab vector store
generation notebook used for CLEVER Phase 2 (Code Retriever).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def build_documents(
    kcd_csv: Path,
    code_column: str = "code",
    description_column: str = "description",
) -> list[Document]:
    """Load KCD rows from CSV and convert them into LangChain documents."""
    df = pd.read_csv(kcd_csv, encoding="utf-8")

    missing_columns = {
        column
        for column in (code_column, description_column)
        if column not in df.columns
    }
    if missing_columns:
        raise ValueError(
            f"Missing required column(s) in {kcd_csv}: "
            f"{', '.join(sorted(missing_columns))}"
        )

    documents: list[Document] = []
    for _, row in df.iterrows():
        description = row[description_column]
        code = row[code_column]

        if pd.isna(description):
            continue

        description_text = str(description).strip()
        if not description_text:
            continue

        code_text = "" if pd.isna(code) else str(code).strip()
        documents.append(
            Document(
                page_content=description_text,
                metadata={"code": code_text},
            )
        )

    if not documents:
        raise ValueError(f"No valid documents were created from {kcd_csv}")

    return documents


def build_faiss_index(
    kcd_csv: Path,
    output_index: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    batch_size: int = 32,
    code_column: str = "code",
    description_column: str = "description",
) -> None:
    """Build and save a FAISS vector store."""
    documents = build_documents(
        kcd_csv=kcd_csv,
        code_column=code_column,
        description_column=description_column,
    )

    model_kwargs = {}
    encode_kwargs = {"batch_size": batch_size}
    if device != "auto":
        model_kwargs["device"] = device
        encode_kwargs["device"] = device

    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings_model,
    )

    output_index.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_index))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SapBERT FAISS index for CLEVER Phase 2."
    )
    parser.add_argument(
        "--kcd-csv",
        required=True,
        type=Path,
        help="Path to a KCD CSV with code and description columns.",
    )
    parser.add_argument(
        "--output-index",
        required=True,
        type=Path,
        help="Directory where the FAISS index will be saved.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace embedding model name.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Embedding device. Use cuda when a GPU is available.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--code-column",
        default="code",
        help="CSV column containing KCD codes.",
    )
    parser.add_argument(
        "--description-column",
        default="description",
        help="CSV column containing curated KCD descriptions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_faiss_index(
        kcd_csv=args.kcd_csv,
        output_index=args.output_index,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        code_column=args.code_column,
        description_column=args.description_column,
    )
    print(f"Saved FAISS index to {args.output_index}")


if __name__ == "__main__":
    main()
