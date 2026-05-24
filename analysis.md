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