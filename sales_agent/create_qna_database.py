"""
Script to build the ChromaDB RAG database for the AI Ultimate Course sales agent.

Reads a plain-text Q&A file (Q: ... / A: ... pairs separated by blank lines)
and indexes each pair into a Chroma collection so the sales agent can look up
answers by semantic similarity.
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


def get_embedding_function():
    """
    Use OpenAI's embeddings API instead of Chroma's default local ONNX model.
    The default model (all-MiniLM-L6-v2 via onnxruntime) needs to be downloaded
    and loaded into memory to embed documents — on Render's free tier (512MB),
    that alone was enough to OOM the whole app on every fresh deploy. Calling
    OpenAI's API instead avoids running any model locally.
    """
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )


def parse_qna_file(txt_path: str) -> list[dict]:
    """
    Parse a Q&A text file into a list of {"question": ..., "answer": ...} dicts.

    Expected format, one pair per block, blocks separated by a blank line:
        Q: <question>
        A: <answer that can span multiple lines>
    """
    text = Path(txt_path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    pairs = []
    for block in blocks:
        if not block.startswith("Q:"):
            continue

        q_part, _, a_part = block.partition("\nA:")
        question = q_part[2:].strip()
        answer = a_part.strip()

        if question and answer:
            pairs.append({"question": question, "answer": answer})

    return pairs


def setup_qna_chromadb(
    txt_path: str,
    chroma_path: str,
    collection_name: str = "course_qna_db",
    embedding_function=None,
):
    """
    Create and populate a ChromaDB collection with course Q&A pairs.
    """
    pairs = parse_qna_file(txt_path)
    if not pairs:
        raise ValueError(f"No Q&A pairs found in {txt_path}. Expected 'Q: ...' / 'A: ...' blocks.")

    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "AI Ultimate Course sales Q&A knowledge base"},
        embedding_function=embedding_function or get_embedding_function(),
    )

    documents = [f"Q: {p['question']}\nA: {p['answer']}" for p in pairs]
    metadatas = [{"question": p["question"], "answer": p["answer"]} for p in pairs]
    ids = [f"qna_{i}" for i in range(len(pairs))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Added {len(pairs)} Q&A pairs to ChromaDB collection '{collection_name}'")
    return collection


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    script_dir = Path(__file__).parent
    txt_path = script_dir.parent / "data" / "course_qna.txt"
    chroma_path = script_dir.parent / "chroma"

    setup_qna_chromadb(str(txt_path), str(chroma_path))
