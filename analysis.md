# DisasterM3 — Repository Analysis

**Task 3 | Internship Preparatory Submission**

---

## 1. Current Repository Structure

```
DisasterM3/
├── __init__.py
├── README.md
├── models/
│   └── __init__.py       ← All model logic (base class + QwenVL, InternVL, Llava)
└── pyscripts/
    ├── __init__.py
    └── run_vllm.py       ← entry-point (data loading + inference + saving)
```

The repository contains **two meaningful files**. There are no dedicated folders for datasets, evaluation, configuration, or experiment tracking. The entire pipeline (reading data, formatting prompts, running models, and saving results is implemented inside a single script `run_vllm.py`).

---

### 1.1 Code Organization Analysis

#### `pyscripts/run_vllm.py`

This is the only executable script. It performs five distinct responsibilities in sequence:

1. **Prompt template definition** — A `prompt_libs` dictionary at the top of the file hardcodes all 8 task prompt templates as Python strings
2. **Data loading** — Reads a JSON file from `{PROJECT_ROOT}/data/{subset}.json` inline in `__main__`
3. **Message construction** — `get_messages_from_data()` builds the multimodal message list per sample, with per-subset branching via if/elif chains
4. **Batch inference** — `create_batch_inputs()` and the main loop call the vLLM engine and collect outputs
5. **Result saving** — Writes predictions to `{PROJECT_ROOT}/results/{subset}/{model_id}/finished.jsonl`

There is no evaluation step. The script ends after saving raw model predictions.

#### `models/__init__.py`

This file is comparatively better structured. It defines:

- `ModelConfig` — an abstract base class with one abstract method: `get_prompt_from_question(messages)`
- `QwenVL`, `InternVL`, `Llava` — concrete subclasses, one per supported model family
- `build_model_config(model_id)` — a factory function that dispatches by checking substrings in the model ID string (e.g. `"qwen"`, `"intern"`, `"llava"`)

This model abstraction is the strongest design pattern in the repo. However, the file also contains several unrelated image and video processing utilities (`build_transform`, `dynamic_preprocess`, `load_video`, `load_video_frame_np`) that have no place in a model configuration module and should live in a `utils/` folder instead.

#### Summary

The two files serve entirely different levels of design quality. `models/__init__.py` shows awareness of abstraction; there is a base class, concrete implementations, and a factory. `run_vllm.py` shows none of this. It is a monolithic procedural script where every concern is mixed together. The limitations below stem directly from this imbalance.

---

## 2. Identified Limitations

### L1: No Dataset Abstraction

Dataset loading is not behind any interface. There is no `Dataset` class, no `load()` method, no way to swap datasets without touching the main script. The data path is hardcoded relative to `PROJECT_ROOT`:

```python
subset_json = join(f"{PROJECT_ROOT}/data", f"{args.subset}.json")
```

And the message construction function branches only on DisasterM3's own subset names:

```python
if subset in ["bearing_body", "building_damage_counting", ...]:
    ...
elif subset in ["landuse", "relational_reasoning_qa"]:
    ...
else:
    raise ValueError('Unknown subset {}'.format(subset))
```

Any dataset other than DisasterM3 raises a `ValueError` immediately. Adding a new dataset means editing `run_vllm.py` in multiple places simultaneously (the path logic, the branching logic, the prompt templates, and the image field names).

### L2: No Evaluation Module

The script saves raw model predictions to disk as `.jsonl` but computes no metrics. There is no accuracy calculation, no BLEU, no ROUGE-L, no F1. A user who runs the script receives a file of generated text strings with no way to know how well the model actually performed without writing their own evaluation code from scratch.

### L3: No Configuration System

All parameters are either command-line arguments or values hardcoded directly in Python:

- Prompt templates → hardcoded `prompt_libs` dict in `run_vllm.py`
- Dataset path → hardcoded relative to `PROJECT_ROOT`
- Model GPU settings → hardcoded per model size inside each model class (e.g. `"7b" → tensor_parallel_size=1`, `"72b" → 4`)

There is no YAML or JSON config file. Reproducing an experiment exactly requires either a carefully saved shell command or manually inspecting the source code to find what was hardcoded at the time.

### L4: No Experiment Tracking

Results are written to a directory tree: `results/{subset}/{model_id}/finished.jsonl`. There is no logging of hyperparameters, runtime, dataset metadata, or metric summaries. Comparing results across multiple models or subsets means manually navigating output folders and parsing raw JSON (no dashboard, no run history, no structured record of what was run).

### L5: Tight Coupling of Concerns

`run_vllm.py` bundles data loading, prompt construction, image preprocessing, batch inference, and file I/O into a single 200 line script approximately with no separation of concerns. Changing any one aspect (for example, how images are resized) requires carefully navigating the script to find the right place without accidentally breaking the parts around it. This makes the codebase fragile to modify and hard to test in isolation.

### L6: Model Extension Requires Source Edits

Adding a new model family requires modifying `models/__init__.py` directly: creating a new subclass of `ModelConfig` and adding a new `elif` branch inside `build_model_config()`. There is no plugin system or registry. A clean framework would allow adding a new model by creating a new file alone, with zero changes to existing code.

### L7: Unrelated Utilities Mixed Into the Model Module

`models/__init__.py` contains low-level image and video processing functions (`build_transform`, `dynamic_preprocess`, `load_video`, `load_video_frame_np`) alongside the model abstraction classes. These utilities are not part of the model interface — they are preprocessing helpers — and their presence in the model module makes the file harder to read, harder to test, and misleading about what the module is responsible for.