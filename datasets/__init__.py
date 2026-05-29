from datasets.base import BaseDataset
from datasets.disasterm3 import DisasterM3Dataset
from datasets.monitrs import MONITRSDataset

DATASET_REGISTRY = {
    "disasterm3": DisasterM3Dataset,
    "monitrs":    MONITRSDataset,
}

def build_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)