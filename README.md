# Qwen3-Embedding

This repository demonstrates how to test source code embeddings using
the `Qwen3-Embedding` model.

## Usage

1. Create `tasks.txt` containing task names (one per line) and
   `queries.txt` containing matching queries.
2. Place the code snippet you want to compare in a file and run:

    ```bash
    python test_code_embedding.py --document path/to/code.py
    ```

    The script casts the model to `float32` automatically when running on
    CPU to avoid NaN values in the computed embeddings.

The script will read the task descriptions from `task_prompts.json`,
compose prompts from the tasks and queries and print the cosine
similarity between each prompt and the code snippet.

## JavaScript reranker example

The repository also includes `qwen3_reranker_onnx.js` which demonstrates how to
use [transformers.js](https://huggingface.co/docs/transformers.js/index) with the
ONNX version of **Qwen3-Reranker**. The class exposes an `infer()` method that
mirrors the Python example and returns probabilities for "yes" and "no" answers.
