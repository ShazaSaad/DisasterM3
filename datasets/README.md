## Datasets explaination and usage notes

### Adding a new dataset
1. Create `datasets/your_dataset.py`
2. Inherit from `BaseDataset`
3. Implement `load()`, `__getitem__()`, `__len__()`
4. Register it in `datasets/__init__.py`

### Usage
from datasets import build_dataset
dataset = build_dataset("disasterm3", data_dir="./data", subset="bearing_body")
dataset.load()
sample = dataset[0]