"""Example script to test source code embeddings using Qwen3-Embedding model.

This script composes a query instruction from a task description and a
question about a source code file. It splits the file into logical code
chunks using LangChain's ``LanguageParser`` and then embeds the query with
the most relevant chunk to print their cosine similarity.

The script reads a list of task names from ``tasks.txt`` and a list of
queries from ``queries.txt``. Each line of the two files forms a pair. For
every pair it prints the cosine similarity score between the prompt and the
document embedding.

Example::

    python test_code_embedding.py --document path/to/source_chunk.py
"""

import argparse
import json
from pathlib import Path

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

TASK_FILE = "task_prompts.json"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool embeddings using the last non-padding token."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lengths]


def get_prompt(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def load_task_description(task_name: str) -> str:
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    desc = tasks.get(task_name)
    if isinstance(desc, dict):
        desc = desc.get("query") or next(iter(desc.values()))
    if not desc:
        raise ValueError(f"Task '{task_name}' not found in {TASK_FILE}")
    return desc


def load_code_chunks(path: Path, language: str = "python") -> list[str]:
    """Split a source file into code chunks using LanguageParser."""
    loader = GenericLoader.from_filesystem(str(path), parser=LanguageParser(language=language))
    docs = list(loader.lazy_load())
    return [doc.page_content for doc in docs]


def embed_texts(texts, model_name: str, max_length: int = 8192) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(torch.float32)
    model.eval()

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch.to(model.device)
    with torch.no_grad():
        outputs = model(**batch)
    embeddings = last_token_pool(outputs.last_hidden_state.float(), batch["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def extract_snippet(
    task_description: str,
    query: str,
    code_chunks: list[str],
    model_name: str,
    max_length: int,
    top_k: int = 5,
) -> str:
    """Select code chunks most relevant to the task and query using embeddings."""
    prompt = get_prompt(task_description, query)

    if not code_chunks:
        return ""

    embeddings = embed_texts([prompt] + code_chunks, model_name, max_length=max_length)
    prompt_emb = embeddings[0]
    chunk_embs = embeddings[1:]
    scores = torch.mv(chunk_embs, prompt_emb)
    k = min(top_k, len(code_chunks))
    top_idx = torch.topk(scores, k=k).indices.tolist()
    snippet_chunks = [code_chunks[i] for i in sorted(top_idx)]
    return "\n".join(snippet_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Qwen3-Embedding on source code snippets"
    )
    parser.add_argument(
        "--tasks_file",
        type=Path,
        default="tasks.txt",
        help="File containing task names (one per line)",
    )
    parser.add_argument(
        "--queries_file",
        type=Path,
        default="queries.txt",
        help="File containing queries (one per line)",
    )
    parser.add_argument(
        "--document",
        required=True,
        type=Path,
        help="Path to code snippet (~1k tokens)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model to use")
    parser.add_argument("--max_length", type=int, default=8192)
    args = parser.parse_args()

    tasks = [ln.strip() for ln in args.tasks_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    queries = [ln.strip() for ln in args.queries_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(tasks) != len(queries):
        raise ValueError("tasks_file and queries_file must have the same number of lines")

    code_chunks = load_code_chunks(args.document)

    snippet_embs = []

    for task_name, query in zip(tasks, queries):
        task_desc = load_task_description(task_name)
        prompt = get_prompt(task_desc, query)
        snippet = extract_snippet(
            task_desc, query, code_chunks, args.model, args.max_length
        )
        embeddings = embed_texts([prompt, snippet], args.model, max_length=args.max_length)
        score = float(embeddings[0] @ embeddings[1:].T)
        snippet_embs.append(embeddings[1])
        print(f"{task_name!s}: {score:.4f}")

    if len(snippet_embs) > 1:
        ref = snippet_embs[0]
        for idx, emb in enumerate(snippet_embs[1:], 1):
            cos_delta = float(ref @ emb)
            dist_delta = float(torch.dist(ref, emb))
            print(f"Δ task0-task{idx} | Cosine: {cos_delta:.4f} | Dist: {dist_delta:.4f}")


if __name__ == "__main__":
    main()
