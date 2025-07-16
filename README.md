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
