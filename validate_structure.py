import json
import os
from os.path import dirname, abspath, join

# Simulate what run_vllm.py does
PROJECT_ROOT = os.getcwd()
subset = "bearing_body"

subset_json = join(PROJECT_ROOT, "data", f"{subset}.json")
print(f"Looking for dataset at: {subset_json}")
print(f"File exists: {os.path.exists(subset_json)}")

images_dir = join(PROJECT_ROOT, "data", "images")
print(f"Looking for images at: {images_dir}")
print(f"Directory exists: {os.path.exists(images_dir)}")