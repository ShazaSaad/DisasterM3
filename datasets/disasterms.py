"""
Concrete dataset adapter for DisasterM3.

Refactored from the inline data loading logic in:
    pyscripts/run_vllm.py (original DisasterM3 codebase)

Changes from original:
    - Data loading extracted into a proper class with a clean interface
    - Hardcoded PROJECT_ROOT path replaced with configurable data_dir argument
    - Hardcoded subset branching replaced with per-subset loader methods
    - Each sample returned as a standardised dict matching BaseDataset schema
    - Prompt templates moved here from run_vllm.py (they are dataset-specific)
    - Resume logic (finish_ids) removed — handled by experiment runner instead
"""

import json
import os
from os.path import join
from typing import Dict, List, Optional

from datasets.base import BaseDataset


# ---------------------------------------------------------------------------
# Prompt templates (moved from run_vllm.py prompt_libs dict)
# These are DisasterM3-specific and belong with the dataset adapter,
# not in the main pipeline script.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "bearing_body": (
        "Analyze both the pre-disaster and post-disaster images to answer "
        "the following question. Choose the best option(s) from the candidate "
        "options provided.\n\n"
        "pre-disaster image:\n<image>\n\n"
        "post-disaster image:\n<image>\n\n"
        "Question: {question}\nOptions: {options_str}\n\n"
        "Your task is to respond with ONLY the capital letters of the correct "
        "options, separated by a comma and a space (e.g., C, D, H). "
        "Do not include any explanation or other text."
    ),
    "single_choice": (
        "Analyze both the pre-disaster and post-disaster images to answer "
        "the following question. Choose the best option from the candidate "
        "options provided.\n\n"
        "pre-disaster image:\n<image>\n\n"
        "post-disaster image:\n<image>\n\n"
        "Question: {question}\nOptions: {options_str}\n\n"
        "Your task is to respond with ONLY the capital letter of the correct "
        "option (e.g., C). Do not include any explanation or other text."
    ),
    "landuse": (
        "Analyze the image to answer the following question. Choose the best "
        "option(s) from the candidate options provided.\n\n"
        "Question: {question}\nOptions: {options_str}\n\n"
        "Your task is to respond with ONLY the capital letters of the correct "
        "options, separated by a comma and a space (e.g., C, D, H). "
        "Do not include any explanation or other text."
    ),
    "relational_reasoning_qa": (
        "Analyze the image to answer the following question. Choose the best "
        "option from the candidate options provided.\n\n"
        "Question: {question}\nOptions: {options_str}\n\n"
        "Your task is to respond with ONLY the capital letter of the correct "
        "option (e.g., C). Do not include any explanation or other text."
    ),
    "caption": (
        "Your TASK is to analyze the provided pair of pre-disaster and "
        "post-disaster remote sensing images.\n\n"
        "pre-disaster image:\n<image>\n\n"
        "post-disaster image:\n<image>\n\n"
        "Your analysis must be formatted as follows:\n"
        "DISASTER: [the name of the disaster]\n"
        "BUILDING: [describe impacts on buildings]\n"
        "ROAD: [describe impacts on road networks]\n"
        "VEGETATION: [describe impacts on natural vegetation cover]\n"
        "WATER_BODY: [describe changes to water bodies]\n"
        "AGRICULTURE: [describe impacts on managed agricultural land]\n"
        "CONCLUSION: [provide a concise 1-2 sentence summary]"
    ),
    "recovery": (
        "Your TASK is to generate concise recovery recommendations for the "
        "affected area based on the provided pre-disaster and post-disaster "
        "remote sensing images.\n\n"
        "pre-disaster image:\n<image>\n\n"
        "post-disaster image:\n<image>\n\n"
        "Based on your analysis:\n"
        "1. First determine if recovery actions are necessary.\n"
        "2. If recovery is needed, provide:\n"
        "IMMEDIATE_RECOVERY: [integrated paragraph, max 50 words]\n"
        "LONG_TERM_RECOVERY: [integrated paragraph, max 50 words]"
    ),
}

# Map each subset name to its prompt template key and image count
SUBSET_CONFIG = {
    "bearing_body":            {"prompt_key": "bearing_body",          "n_images": 2, "task_type": "multiple_choice"},
    "building_damage_counting":{"prompt_key": "single_choice",         "n_images": 2, "task_type": "multiple_choice"},
    "disaster_type":           {"prompt_key": "single_choice",         "n_images": 2, "task_type": "multiple_choice"},
    "road_damage_counting":    {"prompt_key": "single_choice",         "n_images": 2, "task_type": "multiple_choice"},
    "landuse":                 {"prompt_key": "landuse",               "n_images": 1, "task_type": "multiple_choice"},
    "relational_reasoning_qa": {"prompt_key": "relational_reasoning_qa","n_images": 1, "task_type": "multiple_choice"},
    "caption":                 {"prompt_key": "caption",               "n_images": 2, "task_type": "open_ended"},
    "recovery":                {"prompt_key": "recovery",              "n_images": 2, "task_type": "open_ended"},
}

VALID_SUBSETS = list(SUBSET_CONFIG.keys())


class DisasterM3Dataset(BaseDataset):
    """
    Dataset adapter for DisasterM3.

    Loads one task subset at a time from the DisasterM3 JSON annotation
    files and returns samples in the standardised BaseDataset format.

    Data is expected at:
        {data_dir}/{subset}.json          ← annotation file
        {data_dir}/images/                ← satellite image files

    Args:
        data_dir: Path to the DisasterM3 data root directory.
        subset:   One of the 8 DisasterM3 task subsets.
        split:    Dataset split — always "test" for DisasterM3 benchmark.

    Example:
        dataset = DisasterM3Dataset(data_dir="./data", subset="bearing_body")
        dataset.load()
        print(len(dataset))    # e.g. 500
        sample = dataset[0]
        print(sample["id"])           # "bearing_body_0"
        print(sample["image_paths"])  # ["/path/to/pre.png", "/path/to/post.png"]
        print(sample["question"])     # full prompt string
        print(sample["answer"])       # ground truth answer letter(s)
        print(sample["task_type"])    # "multiple_choice"
    """

    def __init__(self, data_dir: str, subset: str, split: str = "test"):
        if subset not in VALID_SUBSETS:
            raise ValueError(
                f"Unknown DisasterM3 subset: '{subset}'\n"
                f"Valid subsets are: {VALID_SUBSETS}"
            )
        super().__init__(data_dir=data_dir, subset=subset, split=split)
        self.images_dir = join(data_dir, "images")
        self.subset_config = SUBSET_CONFIG[subset]

    def load(self) -> None:
        """
        Load all samples for self.subset from the annotation JSON file.

        Reads {data_dir}/{subset}.json and builds self.items as a list of
        standardised sample dicts. Each dict contains the fields required
        by BaseDataset plus DisasterM3-specific metadata.

        Raises:
            FileNotFoundError: if the annotation JSON file does not exist.
        """
        annotation_path = join(self.data_dir, f"{self.subset}.json")

        if not os.path.exists(annotation_path):
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_path}\n"
                f"Please download the DisasterM3 dataset and place the JSON "
                f"files at: {self.data_dir}"
            )

        with open(annotation_path, "r") as f:
            raw_data = json.load(f)

        self.items = []
        for idx, raw_sample in enumerate(raw_data):
            sample = self._parse_sample(
                raw_sample=raw_sample,
                sample_id=f"{self.subset}_{idx}"
            )
            self.items.append(sample)

        print(f"Loaded {len(self.items)} samples from DisasterM3 subset '{self.subset}'")

    def _parse_sample(self, raw_sample: Dict, sample_id: str) -> Dict:
        """
        Convert a raw JSON sample into the standardised BaseDataset format.

        Handles the three image configurations in DisasterM3:
          - Two images (pre + post): bearing_body, building_damage_counting,
            disaster_type, road_damage_counting, caption, recovery
          - Single pre image: landuse
          - Single image with different field name: relational_reasoning_qa

        Args:
            raw_sample: Raw dict loaded from the DisasterM3 JSON file.
            sample_id:  Unique string ID for this sample.

        Returns:
            Standardised sample dict with BaseDataset-required fields.
        """
        cfg = self.subset_config
        prompt_template = PROMPT_TEMPLATES[cfg["prompt_key"]]

        # --- Build image paths ---
        if self.subset in ["bearing_body", "building_damage_counting",
                           "disaster_type", "road_damage_counting",
                           "caption", "recovery"]:
            image_paths = [
                join(self.images_dir, raw_sample["pre_image_path"]),
                join(self.images_dir, raw_sample["post_image_path"]),
            ]
        elif self.subset == "landuse":
            image_paths = [
                join(self.images_dir, raw_sample["pre_image_path"])
            ]
        elif self.subset == "relational_reasoning_qa":
            # relational_reasoning_qa uses "image_path" with a different root
            image_paths = [
                join(self.data_dir, raw_sample["image_path"].replace("\\", "/"))
            ]

        # --- Build question prompt ---
        if cfg["task_type"] == "multiple_choice":
            # use "option_str" for relational_reasoning_qa, "options_str" elsewhere
            options_key = "option_str" if self.subset == "relational_reasoning_qa" else "options_str"
            question = prompt_template.format(
                question=raw_sample["prompts"],
                options_str=raw_sample[options_key]
            )
        else:
            # open-ended subsets (caption, recovery) have no options
            question = prompt_template

        # --- Ground truth answer ---
        answer = raw_sample.get("answer", raw_sample.get("answers", ""))

        return {
            # --- BaseDataset required fields ---
            "id":          sample_id,
            "image_paths": image_paths,
            "question":    question,
            "answer":      answer,
            "task_type":   cfg["task_type"],
            # --- DisasterM3-specific metadata ---
            "subset":      self.subset,
            "raw":         raw_sample,   # preserve original for debugging
        }

    def __getitem__(self, idx: int) -> Dict:
        if not self.items:
            raise RuntimeError(
                "Dataset is empty. Did you call dataset.load() first?"
            )
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)