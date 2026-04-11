"""Ingest one or many PDF/TXT files, chunk them, and index them in FAISS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_input_file(input_path: Path):
	"""Load supported input files into LangChain documents."""
	extension = input_path.suffix.lower()
	if extension == ".pdf":
		loader = PyPDFLoader(str(input_path))
		return loader.load()
	if extension == ".txt":
		loader = TextLoader(str(input_path), encoding="utf-8")
		return loader.load()
	raise ValueError(f"Unsupported file type: {extension}. Use .pdf or .txt")


def resolve_input_files(input_path: Path) -> list[Path]:
	"""Resolve a single file path or collect all supported files from a directory."""
	if not input_path.exists():
		raise FileNotFoundError(f"Input path not found: {input_path}")

	if input_path.is_file():
		if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
			raise ValueError("Unsupported file type. Use .pdf or .txt")
		return [input_path]

	files = sorted(
		path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
	)
	if not files:
		raise ValueError(
			f"No supported files found in directory: {input_path}. Expected .pdf or .txt"
		)
	return files


def split_documents(documents):
	"""Split documents into chunks of roughly 512 tokens."""
	splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
		chunk_size=512,
		chunk_overlap=64,
	)
	return splitter.split_documents(documents)


def build_faiss_index(documents, output_dir: Path):
	"""Embed chunks and persist them to a local FAISS index."""
	if not os.getenv("OPENAI_API_KEY"):
		raise EnvironmentError("OPENAI_API_KEY is not set")

	embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
	vector_store = FAISS.from_documents(documents, embeddings)
	vector_store.save_local(str(output_dir))
	return vector_store


def save_chunks_json(chunks, output_dir: Path, file_name: str = "chunks.json") -> Path:
	"""Persist chunk text and metadata to JSON for inspection/debugging."""
	chunk_rows = []
	for index, chunk in enumerate(chunks):
		chunk_rows.append(
			{
				"chunk_id": index,
				"content": chunk.page_content,
				"metadata": chunk.metadata,
			}
		)

	output_path = output_dir / file_name
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(chunk_rows, handle, ensure_ascii=False, indent=2, default=str)

	return output_path


def ingest_inputs(input_path: Path, output_dir: Path):
	"""End-to-end ingestion pipeline for one file or many files in a directory."""
	files = resolve_input_files(input_path)

	output_dir.mkdir(parents=True, exist_ok=True)

	documents = []
	for file_path in files:
		documents.extend(load_input_file(file_path))

	chunks = split_documents(documents)
	build_faiss_index(chunks, output_dir)
	save_chunks_json(chunks, output_dir)

	return len(files), len(chunks)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Ingest one PDF/TXT file or all PDF/TXT files in a directory."
	)
	parser.add_argument(
		"input_positional",
		nargs="?",
		help="Backward-compatible positional input path",
	)
	parser.add_argument(
		"output_positional",
		nargs="?",
		help="Backward-compatible positional output directory",
	)
	parser.add_argument(
		"--input",
		required=False,
		help="Path to input file or directory (.pdf/.txt supported)",
	)
	parser.add_argument(
		"--pdf",
		required=False,
		help="Deprecated alias for --input (PDF path)",
	)
	parser.add_argument(
		"--output_dir",
		default="faiss_index",
		help="Directory where the FAISS index will be saved",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_arg = args.input or args.pdf or args.input_positional
	if not input_arg:
		raise ValueError("Provide --input (or --pdf for backward compatibility)")
	input_path = Path(input_arg)
	output_dir = Path(args.output_positional or args.output_dir)

	file_count, chunk_count = ingest_inputs(input_path, output_dir)
	print(f"Indexed {chunk_count} chunks from {file_count} file(s) into {output_dir.resolve()}")


if __name__ == "__main__":
	main()
