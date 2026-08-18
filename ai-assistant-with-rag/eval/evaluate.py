"""
Run as a module from the repo root (same reason as src/ingest.py):
    python -m eval.evaluate
"""
import json  # Parse the ground-truth evaluation file.
from pathlib import Path  # Build a path independent of the current working directory.

from sentence_transformers import SentenceTransformer, util  # Compute semantic similarity.

from src.config import DEVICE, EMBEDDING_MODEL  # Reuse the application embedding settings.
from src.pipeline import run  # Evaluate the same path used by the CLI.

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"  # Locate expected Q&A pairs.


def keyword_overlap(gold: str, pred: str) -> float:
    gold_words = set(gold.lower().split())  # Normalize expected-answer words.
    pred_words = set(pred.lower().split())  # Normalize generated-answer words.
    if not gold_words:  # Avoid division by zero for an empty expected answer.
        return 0.0  # Define empty-reference overlap as zero.
    return len(gold_words & pred_words) / len(gold_words)  # Measure shared reference vocabulary.


def main():
    with open(EVAL_SET_PATH) as f:  # Open the fixed evaluation dataset.
        eval_set = json.load(f)  # Decode its question/answer objects.

    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)  # Load the similarity encoder.

    total_sim, total_overlap = 0.0, 0.0  # Accumulate metrics for the final averages.
    for item in eval_set:  # Evaluate every reference question.
        question, gold = item["question"], item["answer"]  # Extract expected inputs and answers.
        result = run(question)  # Generate an answer through the production pipeline.
        pred = result["answer"]  # Read the answer from the pipeline result.

        emb_gold = embedder.encode(gold, convert_to_tensor=True)  # Encode the reference answer.
        emb_pred = embedder.encode(pred, convert_to_tensor=True)  # Encode the generated answer.
        sim = util.cos_sim(emb_gold, emb_pred).item()  # Compare meaning with cosine similarity.
        overlap = keyword_overlap(gold, pred)  # Compare shared exact words.

        total_sim += sim  # Add this answer's semantic score to the total.
        total_overlap += overlap  # Add this answer's keyword score to the total.

        print(f"Q: {question}")
        print(f"  Gold: {gold}")
        print(f"  Pred: {pred}")
        print(f"  similarity={sim:.3f}  keyword_overlap={overlap:.3f}\n")

    n = len(eval_set)  # Count cases for averaging.
    if n:  # Avoid division by zero if the evaluation file is empty.
        print(f"Average similarity: {total_sim / n:.3f}")  # Report semantic quality.
        print(f"Average keyword_overlap: {total_overlap / n:.3f}")  # Report lexical quality.
    # Report both: cosine similarity alone penalizes correct but differently-phrased
    # answers against terse gold answers, and keyword_overlap alone misses paraphrase.


if __name__ == "__main__":  # Support `python -m eval.evaluate` execution.
    main()  # Start the evaluation run.
