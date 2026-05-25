"""
Abstract base class for all dataset adapters in the framework.
Every dataset (DisasterM3, MONITRS, EarthVQA, etc.) must inherit
from BaseDataset and implement all abstract methods.

This interface allows the rest of the pipeline (model runner,
evaluator, experiment tracker) to work with any dataset without
knowing which specific one is loaded.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseDataset(ABC):
    """
    Abstract base class for disaster analysis datasets.

    Defines the standard interface that all dataset adapters must implement.
    The pipeline interacts only with this interface — never with concrete
    dataset classes directly.

    Subclasses must implement:
        - load()
        - __getitem__()
        - __len__()

    Example usage:
        dataset = DisasterM3Dataset(data_dir="./data", subset="bearing_body")
        dataset.load()
        print(len(dataset))       # number of samples
        sample = dataset[0]       # first sample as a dict
    """

    def __init__(self, data_dir: str, subset: str, split: str = "test"):
        """
        Args:
            data_dir: Path to the root directory containing dataset files.
            subset:   Name of the task subset to load (e.g. "bearing_body").
            split:    Dataset split to use — "train", "val", or "test".
                      Defaults to "test" since this framework is eval-only.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.split = split
        self.items: List[Dict] = []  # populated by load()

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Please download the dataset and place it at this path."
            )

    @abstractmethod
    def load(self) -> None:
        """
        Load dataset samples from disk into self.items.

        Each item in self.items must be a dict containing at minimum:
            - "id":           unique string identifier for the sample
            - "image_paths":  list of absolute paths to image file(s)
            - "question":     the question string
            - "answer":       the ground truth answer string
            - "task_type":    string describing the task (e.g. "multiple_choice")

        Raises:
            FileNotFoundError: if the expected data files are not found.
            NotImplementedError: if not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement load()"
        )

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """
        Return the sample at position idx as a standardised dict.

        The returned dict must follow the same schema as items in self.items.

        Args:
            idx: Integer index of the sample.

        Returns:
            A dict with keys: id, image_paths, question, answer, task_type.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __getitem__()"
        )

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the loaded dataset.

        Returns:
            Integer count of samples in self.items.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __len__()"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"subset='{self.subset}', "
            f"split='{self.split}', "
            f"n_samples={len(self.items)})"
        )