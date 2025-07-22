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

import json
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd

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


def compare_instruction_similarity(model, instructions: dict, title: str) -> None:
    names = list(instructions.keys())
    vecs = encode_texts(model, instructions.values(), instruction="")
    sims = cosine_similarity(vecs)
    df = pd.DataFrame(sims, index=names, columns=names)
    print(f"\n=== {title} ===")
    print(df.round(3).to_string())


def evaluate(model: SentenceTransformer, data, top_k=5):
    """Return DataFrame with metrics for every formatting variant and key."""

    grouped = defaultdict(list)
    for row in data:
        grouped[row["key"]].append(row)

    rows_out = []

    for var_name, fmt in FORMATTING_VARIANTS.items():
        for key, pairs in grouped.items():
            idx_instr = fmt(INDEXING_INSTR[key])
            qry_instr = fmt(SEARCH_INSTR[key])

            doc_vecs = encode_texts(
                model,
                [r["doc"] for r in pairs],
                instruction=idx_instr,
            )
            doc_vecs = np.vstack(doc_vecs).astype("float32")
            doc_ids = [r["id"] for r in pairs]

            query_vecs = encode_texts(
                model,
                [r["query"] for r in pairs],
                instruction=qry_instr,
            )

            cos_all, hits_all, mrr_all, margin_all = [], [], [], []
            for q_vec, pair in zip(query_vecs, pairs):
                sims = cosine_similarity(q_vec[None], doc_vecs)[0]
                ranked = np.argsort(-sims)
                correct_idx = doc_ids.index(pair["id"])
                cos_val = float(sims[correct_idx])
                cos_all.append(cos_val)
                hits_all.append(pair["id"] in np.array(doc_ids)[ranked[:top_k]])
                rank = np.where(np.array(doc_ids)[ranked] == pair["id"])[0][0] + 1
                mrr_all.append(1 / rank)

                # margin between relevant and best non-relevant
                other = np.delete(sims, correct_idx)
                margin_all.append(float(cos_val - np.max(other)))

            mean_cos = np.mean(cos_all)
            results_row = {
                "variant": var_name,
                "key": key,
                "instruction": f"{var_name}/{key}",
                "index_instr": idx_instr,
                "search_instr": qry_instr,
                "precision@{k}".format(k=top_k): np.mean(hits_all),
                "recall@{k}".format(k=top_k): np.mean(hits_all),
                "MRR": np.mean(mrr_all),
                "avg_greet": mean_cos,
                "min_greet": float(np.min(cos_all)),
                "margin": float(np.mean(margin_all)),
                "stability": float(1.0 / (1.0 + np.std(cos_all))),
            }
            rows_out.append(results_row)

    return pd.DataFrame(rows_out)


def print_report(df: pd.DataFrame, top_k: int) -> None:
    df_base = df.copy()

    print("\n=== TOP BY PRECISION@K ===")
    prec_col = f"precision@{top_k}"
    recall_col = f"recall@{top_k}"
    top_prec = df_base.nlargest(5, prec_col)[["instruction", prec_col, recall_col, "MRR"]]
    print(top_prec.to_string(index=False))

    print("\n=== TOP 10 BY MARGIN ===")
    top_margin = df_base.nlargest(10, "margin")[["instruction", "margin", "avg_greet", "min_greet", "stability"]]
    print(top_margin.to_string(index=False))

    print("\n=== TOP 10 BY AVERAGE (for stability) ===")
    top_avg = df_base.nlargest(10, "avg_greet")[["instruction", "avg_greet", "margin", "stability"]]
    print(top_avg.to_string(index=False))

    print("\n=== TOP 5 BY STABILITY (consistent across all code types) ===")
    good_margin = df_base[df_base["margin"] > 0.15]
    top_stable = good_margin.nlargest(5, "stability")[["instruction", "stability", "margin", "min_greet"]]
    print(top_stable.to_string(index=False))

    required_cols = ["margin", "avg_greet", "stability", "MRR"]
    for col in required_cols:
        if col not in df_base.columns:
            df_base[col] = 0.0

    df_base["composite_score"] = (
        0.35 * df_base["margin"] +
        0.30 * df_base["avg_greet"] +
        0.35 * df_base["stability"]
    )

    print("\n=== TOP 5 BY COMPOSITE SCORE ===")
    top_comp = df_base.nlargest(5, "composite_score")[["instruction", "composite_score", "margin", "avg_greet", "stability"]]
    print(top_comp.to_string(index=False))


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
    df_res = evaluate(model, data, top_k=args.top_k)

    print_report(df_res, args.top_k)

    compare_instruction_similarity(model, INDEXING_INSTR, "Index_Instruct vs Index_Instruct")
    for name, fmt in FORMATTING_VARIANTS.items():
        formatted = {k: fmt(v) for k, v in SEARCH_INSTR.items()}
        compare_instruction_similarity(model, formatted, f"Search_Instruct ({name}) vs Search_Instruct")


if __name__ == "__main__":
    sys.exit(main())
