import json
import os
from glob import glob
from pathlib import Path
from typing import List

from weak_label.types import Instance


def load_dataset(src: str) -> List[Instance]:
    dataset = []
    for filename in glob(os.path.join(src, "*.json")):
        with open(filename, "r") as f:
            dataset += [Instance(**i) for i in json.load(f)]
    for i, x in enumerate(dataset):
        x.id = i
    return dataset


def export_labeled_dataset(dataset: List[Instance], dst: str) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as out:
        json.dump(dataset, out, default=lambda x: x.__dict__)
