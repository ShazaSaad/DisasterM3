"""
Concrete dataset adapter for MONITRS (Multi-temporal Observations of
Natural dIsasters from TempRal Satellite imagery).

MONITRS differs fundamentally from DisasterM3 in one key way:
instead of pre/post image pairs, it uses *sequences* of satellite
images taken at multiple dates — tracking how a disaster unfolds
over time. Each sample includes a list of image paths, a list of
timestamps, and a question about the temporal progression of the event.

Dataset format expected on disk:
    {data_dir}/
    ├── train_multiple_choice.json   ← MCQ samples
    ├── test_multiple_choice.json
    ├── train_generated_q_a.json     ← open-ended QA samples
    ├── test_generated_q_a.json
    └── all_events/
        └── {event_id}/
            ├── image_0.png
            ├── image_1.png
            └── ...

Each JSON file is a list of samples with the structure:
    {
        "folder_id": 42,
        "video": ["all_events/42/img1.png", ...],   ← image sequence
        "timestamp": ["2021-12-11T00:00:00", ...],  ← one per image
        "lat_lon": [[lat, lon], ...],
        "task": "temporal_grounding",
        "conversations": [
            {"from": "human", "value": "Question text..."},
            {"from": "gpt",   "value": "A"}
        ],
        "metadata": {
            "correct_answer": "A",
            "all_options": [...]
        }
    }

References:
    MONITRS repo: https://github.com/ShreelekhaR/MONITRS
    Evaluation:   MONITRS/Evaluate/eval.py
"""

import json
import os
from os.path import join
from typing import Dict, List, Optional

from datasets.base import BaseDataset


# ---------------------------------------------------------------------------
# Task type definitions
# Sourced from MONITRS_QA/templated_mcq.py mc_templates keys
# ---------------------------------------------------------------------------

MONITRS_TASK_TYPES = {
    "temporal_grounding":      "multiple_choice",
    "event_type":              "multiple_choice",
    "location_identification": "multiple_choice",
    "event_sequence":          "multiple_choice",
    "custom":                  "open_ended",
    "multiple_choice":         "multiple_choice",
}

SPLIT_FILES = {
    "train": [
        "train_multiple_choice.json",
        "train_generated_q_a.json",
        "new_train_multiple_choice.json",
        "new_train_generated_q_a.json",
    ],
    "test": [
        "test_multiple_choice.json",
        "test_generated_q_a.json",
        "new_test_multiple_choice.json",
        "new_test_generated_q_a.json",
    ],
}

VALID_SUBSETS = list(MONITRS_TASK_TYPES.keys()) + ["all"]


class MONITRSDataset(BaseDataset):
    """
    Dataset adapter for MONITRS.

    Loads temporal satellite image sequences with associated QA pairs
    in the standardised BaseDataset format. Supports both multiple-choice
    and open-ended question types, and can filter by task type (subset).

    Key difference from DisasterM3Dataset: each sample's "image_paths"
    is a list of multiple images in temporal order rather than a
    pre/post pair. The model must reason across the full sequence.

    Args:
        data_dir: Path to the MONITRS data root directory.
        subset:   Task type to load, or "all" to load every task type.
        split:    "train" or "test". Defaults to "test".

    Example:
        dataset = MONITRSDataset(
            data_dir="./data/monitrs",
            subset="temporal_grounding",
            split="test"
        )
        dataset.load()
        print(len(dataset))

        sample = dataset[0]
        print(sample["id"])            # "monitrs_test_0"
        print(sample["image_paths"])   # ["/path/img1.png", "/path/img2.png", ...]
        print(sample["timestamps"])    # ["2021-12-11T00:00:00", ...]
        print(sample["question"])      # full question string
        print(sample["answer"])        # ground truth answer letter or text
        print(sample["task_type"])     # "multiple_choice" or "open_ended"
    """

    def __init__(self, data_dir: str, subset: str = "all", split: str = "test"):
        if subset not in VALID_SUBSETS:
            raise ValueError(
                f"Unknown MONITRS subset: '{subset}'\n"
                f"Valid subsets are: {VALID_SUBSETS}"
            )
        if split not in ("train", "test"):
            raise ValueError(f"Split must be 'train' or 'test', got: '{split}'")

        super().__init__(data_dir=data_dir, subset=subset, split=split)
        self.events_dir = join(data_dir, "all_events")
        self._subset_filter = None if subset == "all" else subset

    def load(self) -> None:
        """
        Load MONITRS samples from all available annotation files for the
        current split, optionally filtering by task type (subset).

        Searches for known annotation JSON filenames in data_dir and loads
        all that exist. Robust to naming variations across MONITRS versions.

        Raises:
            FileNotFoundError: if no annotation files are found at all.
        """
        candidate_files = SPLIT_FILES.get(self.split, [])
        found_files = []

        for filename in candidate_files:
            filepath = join(self.data_dir, filename)
            if os.path.exists(filepath):
                found_files.append(filepath)

        if not found_files:
            raise FileNotFoundError(
                f"No MONITRS annotation files found in: {self.data_dir}\n"
                f"Expected one or more of: {candidate_files}\n"
                f"Please download MONITRS and place the JSON files at this path."
            )

        self.items = []
        global_idx = 0

        for filepath in found_files:
            with open(filepath, "r") as f:
                raw_data = json.load(f)

            for raw_sample in raw_data:
                raw_task = raw_sample.get("task", "custom")
                if self._subset_filter and raw_task != self._subset_filter:
                    continue

                sample = self._parse_sample(
                    raw_sample=raw_sample,
                    sample_id=f"monitrs_{self.split}_{global_idx}"
                )
                if sample is not None:
                    self.items.append(sample)
                    global_idx += 1

        print(
            f"Loaded {len(self.items)} samples from MONITRS "
            f"(split='{self.split}', subset='{self.subset}')"
        )

    def _parse_sample(self, raw_sample: Dict, sample_id: str) -> Optional[Dict]:
        """
        Convert a raw MONITRS JSON sample into the standardised BaseDataset format.

        MONITRS stores conversations as:
            [{"from": "human", "value": "<question>"},
             {"from": "gpt",   "value": "<answer>"}]

        Image paths in the raw data are relative to the repo root or
        absolute cluster paths — both are resolved to absolute paths
        under self.data_dir.

        Args:
            raw_sample: Raw dict loaded from a MONITRS JSON file.
            sample_id:  Unique string ID for this sample.

        Returns:
            Standardised sample dict, or None if the sample is malformed.
        """
        try:
            conversations = raw_sample.get("conversations", [])
            if len(conversations) < 2:
                return None

            question = conversations[0].get("value", "")
            answer   = conversations[1].get("value", "")

            if not question or not answer:
                return None

            raw_image_paths = raw_sample.get("video", [])
            image_paths = self._resolve_image_paths(raw_image_paths)
            timestamps  = raw_sample.get("timestamp", [])
            raw_task    = raw_sample.get("task", "custom")
            task_type   = MONITRS_TASK_TYPES.get(raw_task, "open_ended")

            # For MCQ samples, extract the clean answer letter
            if task_type == "multiple_choice":
                metadata = raw_sample.get("metadata", {})
                if "correct_answer" in metadata:
                    answer = metadata["correct_answer"]
                else:
                    # Fallback: first uppercase letter in the answer text
                    for char in answer:
                        if char.isupper():
                            answer = char
                            break

            return {
                # --- BaseDataset required fields ---
                "id":          sample_id,
                "image_paths": image_paths,
                "question":    question,
                "answer":      answer,
                "task_type":   task_type,
                # --- MONITRS-specific fields ---
                "timestamps":  timestamps,
                "event_id":    str(raw_sample.get("folder_id", "")),
                "lat_lon":     raw_sample.get("lat_lon", []),
                "raw_task":    raw_task,
                "raw":         raw_sample,
            }

        except (KeyError, TypeError, ValueError) as e:
            print(f"Warning: skipping malformed MONITRS sample ({sample_id}): {e}")
            return None

    def _resolve_image_paths(self, raw_paths: List[str]) -> List[str]:
        """
        Resolve raw image path strings to absolute paths under self.data_dir.

        MONITRS stores paths in two formats depending on version:
          - Relative: "all_events/42/2021-12-11.png"
          - Absolute cluster path: "/scratch/datasets/ssr234/all_events/42/..."

        Both are normalised to: {data_dir}/all_events/{event_id}/{filename}

        Args:
            raw_paths: List of path strings from the JSON file.

        Returns:
            List of resolved absolute path strings.
        """
        resolved = []
        for raw_path in raw_paths:
            if "all_events/" in raw_path:
                relative = "all_events/" + raw_path.split("all_events/")[-1]
                resolved.append(join(self.data_dir, relative))
            else:
                resolved.append(join(self.data_dir, raw_path))
        return resolved

    def __getitem__(self, idx: int) -> Dict:
        if not self.items:
            raise RuntimeError(
                "Dataset is empty. Did you call dataset.load() first?"
            )
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)