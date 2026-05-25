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

---

## 4. Is the Framework Tied to a Specific Dataset?

**Yes.** The coupling is not in one place but spread across four separate points in `run_vllm.py`, meaning there is no single clean extension point. Each one must be touched independently to support a new dataset.

### Coupling Point 1: Hardcoded Data Path

```python
PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
subset_json = join(f"{PROJECT_ROOT}/data", f"{args.subset}.json")
```

The data directory is derived by walking three levels up from the script's location. Any dataset must be placed inside this specific folder structure. There is no `--data_dir` argument or configurable root path.

### Coupling Point 2: Hardcoded Subset Names

```python
if subset in ["bearing_body", "building_damage_counting", "disaster_type", "road_damage_counting"]:
    ...
elif subset in ["landuse", "relational_reasoning_qa"]:
    ...
elif subset in ["caption", "recovery"]:
    ...
else:
    raise ValueError('Unknown subset {}'.format(subset))
```

The message construction function `get_messages_from_data()` only recognises DisasterM3's 8 subset names. Passing any other dataset name raises `ValueError` immediately — the script does not fail gracefully or suggest alternatives.

### Coupling Point 3: Hardcoded Prompt Templates

All 8 task prompts are defined as a Python dictionary at the top of the script:

```python
prompt_libs = {
    "bearing_body": "...<DisasterM3-specific instruction>...",
    "building_damage_counting": "...",
    ...
}
```

These prompts are written specifically for DisasterM3 tasks and assume DisasterM3 answer options. A different dataset with different task instructions and different answer choices has no mechanism to inject its own prompts without modifying this dictionary directly.

### Coupling Point 4: Hardcoded Image Field Names

The image paths are extracted by looking for specific JSON keys that only exist in DisasterM3's annotation format:

```python
pre_image  = item["pre_image_path"]
post_image = item["post_image_path"]
# or for single-image subsets:
image      = item["image_path"]
```

A dataset like MONITRS (which uses sequences of images referenced differently) or EarthVQA (which pairs images with segmentation masks) would produce `KeyError` on these field names.

### What should I do to Add MONITRS??

To add MONITRS support to the current codebase without any redesign, a developer would need to make **five separate edits** across `run_vllm.py`:

1. Add a new `elif` branch in `get_messages_from_data()` for each MONITRS task type
2. Add new MONITRS prompt strings to the `prompt_libs` dictionary
3. Add MONITRS-specific image field name handling to the image loading block
4. Place MONITRS JSON files inside `{PROJECT_ROOT}/data/` alongside DisasterM3 files
5. Verify the output path logic produces sensible filenames for MONITRS results

Every one of these edits modifies existing shared code rather than adding new isolated code. Any mistake risks breaking DisasterM3 at the same time. This is precisely the problem that a proper dataset abstraction layer solves.

---

## 5. Proposed Modular Redesign

The redesign follows the pipeline principle from the internship brief:

```
Dataset → Model Runner → Evaluator → Experiment Tracker
```

Each stage is an isolated module that communicates with the others only through well-defined interfaces. Adding a new dataset, model, or evaluator means adding a new file.

### Target Directory Structure

```
framework/
├── configs/
│   ├── disasterm3_qwen.yaml       ← one config file per experiment
│   └── monitrs_internvl.yaml
├── datasets/
│   ├── base.py                    ← abstract BaseDataset class
│   ├── disasterm3.py              ← DisasterM3 adapter
│   ├── monitrs.py                 ← MONITRS adapter
│   └── earthvqa.py                ← EarthVQA adapter
├── models/
│   ├── base.py                    ← abstract BaseModelRunner
│   ├── qwen_runner.py             ← QwenVL (cleaned from models/__init__.py)
│   ├── internvl_runner.py         ← InternVL
│   └── llava_runner.py            ← LLaVA
├── evaluation/
│   ├── base.py                    ← abstract BaseEvaluator
│   ├── vqa.py                     ← accuracy, BLEU, METEOR, ROUGE-L
│   └── damage_assessment.py       ← per-class accuracy, RMSE
├── experiments/
│   ├── runner.py                  ← orchestrates the full pipeline
│   └── tracker.py                 ← MLflow / W&B logging wrapper
├── utils/
│   └── image_utils.py             ← image preprocessing (moved from models/__init__.py)
└── main.py                        ← entry point: reads config, runs experiment
```

### How Each Limitation Is Resolved

| Limitation | Resolution |
|---|---|
| **L1** No dataset abstraction | `BaseDataset` abstract class with `load()`, `__getitem__()`, `__len__()`. Each dataset is a separate adapter file. The pipeline never knows which dataset is loaded. |
| **L2** No evaluation | Dedicated `evaluation/` module. Each evaluator exposes `update(pred, gt)` per sample and `compute()` for final metrics. Pattern adapted from EarthVQA's `VQA_OA_Metric` and MONITRS's `eval.py`. |
| **L3** No config system | YAML files in `configs/` control dataset, model, evaluator, and output path. `main.py` reads the config (nothing is hardcoded in Python). |
| **L4** No experiment tracking | `experiments/tracker.py` wraps MLflow or W&B. Logs dataset, model, task, metrics, and runtime automatically after each evaluation run. |
| **L5** Tight coupling | Each stage lives in its own module. `datasets/` only loads data. `models/` only runs inference. `evaluation/` only computes metrics. `experiments/runner.py` connects them through abstract interfaces. |
| **L6** Model extension requires source edits | New models added as new files in `models/` inheriting `BaseModelRunner`. Config file references the class by name. Zero changes to existing files. |
| **L7** Utilities mixed into model module | Image/video helpers moved to `utils/image_utils.py`. `models/` contains only runner logic. |

### What Is Reused vs. Redesigned

| Component | Source | Treatment |
|---|---|---|
| `ModelConfig` abstract class | DisasterM3 `models/__init__.py` | Renamed `BaseModelRunner`, moved to `models/base.py`, interface preserved |
| `QwenVL`, `InternVL`, `Llava` | DisasterM3 `models/__init__.py` | Split into separate files, preprocessing utilities removed |
| `build_model_config()` factory | DisasterM3 `models/__init__.py` | Replaced by config-driven class instantiation |
| MCQ accuracy, BLEU, METEOR, ROUGE-L | MONITRS `Evaluate/eval.py` | Adapted into `evaluation/vqa.py` evaluator class |
| `EarthVQADataset` class pattern | EarthVQA `data/earthvqa.py` | Used as the template for `BaseDataset` and all concrete adapters |
| Config dict pattern | EarthVQA `configs/earthvqa.py` | Upgraded to YAML files in `configs/` |
| Prompt templates | DisasterM3 `run_vllm.py` | Moved into dataset adapters or YAML config files |
| Experiment tracking | — | Entirely new — not present in any of the three repos |