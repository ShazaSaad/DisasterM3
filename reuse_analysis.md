# Reuse Analysis — MONITRS

**Task 6 | Internship Preparatory Submission**  
**Selected framework:** MONITRS  
**Repository:** https://github.com/ShreelekhaR/MONITRS

---

## 1. Overview of MONITRS

MONITRS is a benchmark for temporal disaster monitoring from satellite image sequences. Unlike DisasterM3, which evaluates models on individual pre/post image pairs, MONITRS focuses on reasoning across sequences of images taken at different dates (tracking how a disaster unfolds over time).

The repository serves two purposes: a dataset creation pipeline (scraping news articles, querying satellite imagery, generating QA pairs via Gemini) and an evaluation module. For the purposes of this analysis, only the evaluation module is relevant, as the framework being built is inference and evaluation only.

---

## 2. Identified Reusable Design Pattern: The Evaluator Interface in `Evaluate/eval.py`

The single most reusable component in MONITRS is the evaluation module at `Evaluate/eval.py`. It implements a clean, self-contained interface for computing multiple NLP metrics from a single standardised input format.

### What it does

The module exposes two primary functions:

**`calculate_accuracy_mcq(answers_json)`**
Reads a JSON dict of predictions and ground truths, computes overall accuracy for multiple-choice questions, and returns both the scalar score and a per-sample boolean array.

**`calculate_nlp_metrics(answers_json)`**
Reads the same JSON format and computes six NLP metrics in one call: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, and ROUGE-L.

Both functions share the same input contract:

```python
# Input format; a dict keyed by sample ID:
answers_json = {
    "sample_0": {
        "predicted":    "C",
        "ground_truth": "A",
        "task":         "multiple_choice"
    },
    "sample_1": {
        "predicted":    "The flood destroyed the bridge.",
        "ground_truth": "Bridge infrastructure was severely damaged.",
        "task":         "open_ended"
    }
}

# Usage:
accuracy, correct = calculate_accuracy_mcq(answers_json)
scores = calculate_nlp_metrics(answers_json)
# scores = {"bleu_1": 0.72, "bleu_2": 0.61, ..., "rouge_l": 0.68}
```

The pattern is: **one standardised input format → multiple metrics out**. The caller does not need to know how BLEU or ROUGE-L are computed — it just passes predictions and ground truths.

### Why this pattern is reusable

Three properties make this pattern worth carrying forward:

**1. Input/output contract is dataset-agnostic.** The functions operate on a plain Python dict of `{id: {predicted, ground_truth, task}}`. They have no knowledge of DisasterM3, MONITRS, or any other dataset. Any dataset adapter that produces predictions in this format can use the same evaluator.

**2. Multiple metrics from a single call.** Rather than calling separate functions for BLEU, METEOR, and ROUGE-L individually, `calculate_nlp_metrics` returns all of them together. This is efficient and ensures consistent tokenisation across metrics.

**3. Task-type awareness.** The dict includes a `"task"` field, which allows the evaluator to route multiple-choice predictions to `calculate_accuracy_mcq` and open-ended predictions to `calculate_nlp_metrics` without the caller having to branch manually.

MONITRS also implements `mcnemars_test(correct_preds_a, correct_preds_b)` for statistical comparison between two models. A valuable addition for the experiment tracking layer that does not exist in DisasterM3 or EarthVQA.

---

## 3. How This Pattern Fits Into the Proposed Framework

In the proposed framework, this pattern maps directly onto the `evaluation/` module:

```
framework/
└── evaluation/
    ├── base.py          ← abstract BaseEvaluator
    └── vqa.py           ← adapted from MONITRS Evaluate/eval.py
```

### Adaptation into `evaluation/vqa.py`

The MONITRS functions are adapted into a class-based evaluator that matches the framework's accumulate-then-compute pattern (inspired also by EarthVQA's `VQA_OA_Metric`):

```python
# evaluation/vqa.py
class VQAEvaluator(BaseEvaluator):
    """
    Evaluator for VQA tasks.
    Handles both multiple-choice (accuracy) and open-ended (BLEU/ROUGE-L/METEOR).
    Metric functions adapted from MONITRS Evaluate/eval.py.
    """

    def update(self, prediction: str, ground_truth: str, task_type: str) -> None:
        # accumulate one prediction at a time
        self.predictions[self._current_id] = {
            "predicted":    prediction,
            "ground_truth": ground_truth,
            "task":         task_type
        }

    def compute(self) -> dict:
        # compute all metrics at once using adapted MONITRS functions
        results = {}
        results["accuracy"] = calculate_accuracy_mcq(self.predictions)
        results.update(calculate_nlp_metrics(self.predictions))
        return results
```

### What is reused vs. adapted

| Component | Original (MONITRS) | Adapted (Framework) |
|---|---|---|
| `calculate_accuracy_mcq()` | Standalone function in `eval.py` | Called inside `VQAEvaluator.compute()` |
| `calculate_nlp_metrics()` | Standalone function in `eval.py` | Called inside `VQAEvaluator.compute()` |
| Input format `{id: {predicted, ground_truth, task}}` | Hardcoded JSON file format | Produced by `VQAEvaluator.update()` in memory |
| `mcnemars_test()` | Standalone function in `eval.py` | Called by `experiments/tracker.py` for model comparison |
| LLM-as-judge (`LLM_eval.py`) | Separate script, requires Gemini API key | Optional evaluator — future extension |

The core change is structural: instead of reading from a JSON file on disk, the framework builds the same dict in memory as the experiment runner loops through predictions. The metric computation logic itself is reused with minimal modification.

---

## 4. What MONITRS Does Not Contribute

For completeness, the following parts of MONITRS are explicitly not reused:

- **Dataset creation pipeline** (`MONITRS/` folder): scraping, geocoding, and QA generation scripts are dataset construction tools, not evaluation tools
- **Dataset loader**: MONITRS has no dataset class; data loading must be written from scratch for the MONITRS adapter
- **Model runner**: MONITRS does not include any VLM inference code
- **Configuration system**: no config files exist in MONITRS

---

## 5. Summary

The reusable design pattern identified in MONITRS is the **dataset-agnostic evaluator interface** in `Evaluate/eval.py`: a standardised input dict of predictions and ground truths maps cleanly to a full suite of NLP metrics in a single call. This pattern is directly adapted into the framework's `evaluation/vqa.py` evaluator class, providing accuracy, BLEU-1 through 4, METEOR, ROUGE-L, and McNemar's statistical significance testing (which doesn't exist anywhere in the original DisasterM3 codebase).