#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_cosine_eval.py
Test every INDEX / SEARCH prompt pair across multiple formatting variants.

Data format  (JSONL) — one pair per line:
{
  "key":    "extract_func_headers",
  "query":  "def fetch_user_profile(uid: str) -> UserProfile",
  "doc":    "<full code block…>",
  "id":     "user_service.py"
}
"""

import json, argparse, sys, os
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ────────────────── 1. BASE INSTRUCTIONS  ────────────────────────────────── #
INDEXING_INSTR = {
    "extract_func_headers": (
        "From the given source code, identify and return ONLY the function "
        "definition headers (name, parameters, return type)."
    ),
    "extract_md_annotations": (
        "From the given Markdown, extract explanatory sections that describe "
        "function purpose, parameters and return value."
    ),
    "extract_config_keys": (
        "From the given YAML/JSON/INI snippet, return configuration keys with "
        "their default values."
    ),
    "extract_test_cases": (
        "From the given test file, return declarations of test cases."
    ),
}

SEARCH_INSTR = {
    "extract_func_headers": (
        "You are given a function definition header. Retrieve code fragments "
        "whose function headers semantically match this header."
    ),
    "extract_md_annotations": (
        "The query is a Markdown explanation of a function or API. Retrieve "
        "matching explanatory sections."
    ),
    "extract_config_keys": (
        "Given a configuration key (and optional default), retrieve config "
        "fragments declaring the same key."
    ),
    "extract_test_cases": (
        "The query is a test case declaration. Retrieve files declaring an "
        "equivalent test case."
    ),
}

# ────────────────── 2. FORMATTING VARIANTS  ───────────────────────────────── #
FORMATTING_VARIANTS = {
    "standard":     lambda inst: f"Instruct: {inst}",
    "caps_key":     lambda inst: f"Instruct: {inst.replace('function', 'FUNCTION').replace('declarations', 'DECLARATIONS')}",
    "with_colon":   lambda inst: f"Instruct: {inst}:",
    "with_marker":  lambda inst: f"Instruct: [FOCUS] {inst}",
}
# ── add/remove variants here — скрипт автоматически их подхватит ────────────#


def load_dataset(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def encode_texts(model, texts, instruction):
    return model.encode(
        list(texts),
        instruction=instruction,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def evaluate(model: SentenceTransformer, data, top_k=5):
    """
    Возвращает: dict[variant][key] → метрики.
    """
    # группируем пару query–doc по key
    grouped = defaultdict(list)
    for row in data:
        grouped[row["key"]].append(row)

    results = defaultdict(dict)

    for var_name, fmt in FORMATTING_VARIANTS.items():
        for key, rows in grouped.items():
            idx_instr = fmt(INDEXING_INSTR[key])
            qry_instr = fmt(SEARCH_INSTR[key])

            # индексация документов
            doc_vecs = encode_texts(
                model, [r["doc"] for r in rows], instruction=idx_instr
            )
            doc_vecs = np.vstack(doc_vecs).astype("float32")
            doc_ids  = [r["id"] for r in rows]

            # запросы
            query_vecs = encode_texts(
                model, [r["query"] for r in rows], instruction=qry_instr
            )

            # метрики
            cos_all, hits_all, mrr_all = [], [], []
            for q_vec, row in zip(query_vecs, rows):
                sims = cosine_similarity(q_vec[None], doc_vecs)[0]
                ranked = np.argsort(-sims)             # по убыванию
                cos_all.append(float(sims[doc_ids.index(row["id"])]))

                # Recall@k
                hits_all.append(row["id"] in np.array(doc_ids)[ranked[:top_k]])
                # MRR@k
                rank = np.where(np.array(doc_ids)[ranked] == row["id"])[0][0] + 1
                mrr_all.append(1 / rank)

            results[var_name][key] = {
                "mean_cosine":  np.mean(cos_all),
                f"recall@{top_k}": np.mean(hits_all),
                f"mrr@{top_k}":    np.mean(mrr_all),
                "n": len(rows),
            }

    return results


def print_report(res, top_k):
    header = f"{'variant':<11} | {'key':<22} | cosine  | recall |  mrr  | n"
    print("\n" + header + "\n" + "-" * len(header))
    for var, per_key in res.items():
        for key, metr in sorted(per_key.items(), key=lambda x: -x[1]["mean_cosine"]):
            print(
                f"{var:<11} | {key:<22} | "
                f"{metr['mean_cosine']:.4f} | "
                f"{metr[f'recall@{top_k}']:.2%} | "
                f"{metr[f'mrr@{top_k}']:.3f} | {metr['n']}"
            )


def main():
    ap = argparse.ArgumentParser(description="Evaluate formatting variants for prompts")
    ap.add_argument("--data", required=True, help="JSONL with query-doc pairs")
    ap.add_argument("--model-id", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer(args.model_id, device=args.device)

    data = load_dataset(Path(args.data))
    res  = evaluate(model, data, top_k=args.top_k)

    print_report(res, args.top_k)


if __name__ == "__main__":
    sys.exit(main())
