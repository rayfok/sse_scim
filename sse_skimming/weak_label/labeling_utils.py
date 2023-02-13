import json
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import List

import pysbd
from tqdm import tqdm

from weak_label.types import Instance


def load_csabstruct_from_jsonl(src: str) -> List[Instance]:
    """
    Returns a single list of Instances from a .jsonl file.
    Each line of the file should contain a json for a single paper.
    """
    dataset = []
    num_papers = sum(1 for _ in open(src, "r"))
    seg = pysbd.Segmenter(language="en", clean=False)
    with open(src, "r") as f:
        for line in tqdm(
            f, desc="Loading dataset", total=num_papers, unit="paper"
        ):
            paper = json.loads(line)
            for x in paper["full_text"]:
                # "full_text" contains all sentences for a single section
                # first split it into sentences using pysbd
                for sentence in seg.segment(x["text"]):
                    dataset.append(
                        Instance(
                            id=len(dataset),
                            doc_id=paper["paper_id"],
                            text=sentence,
                            section=x["title"],  # Can be None
                        )
                    )
    return dataset


def load_dataset(src: str) -> List[Instance]:
    dataset = []
    for filename in glob(os.path.join(src, "*.json")):
        doc_id = os.path.splitext(os.path.basename(filename))[0]
        with open(filename, "r") as f:
            for x in json.load(f):
                dataset.append(Instance(**x, doc_id=doc_id))
    return dataset


def export_labeled_dataset(
    dataset: List[Instance],
    dst: str,
    text_and_labels_only: bool = True,
    filter_abstain: bool = False,
) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if filter_abstain:
        dataset = [x for x in dataset if x.label != -1]
    if text_and_labels_only:
        by_doc = defaultdict(list)
        for x in dataset:
            by_doc[x.doc_id].append(x)
        with open(dst.with_suffix(".jsonl"), "w") as out:
            for instances in by_doc.values():
                out.write(
                    json.dumps(
                        {
                            "sents": [x.text for x in instances],
                            "labels": [x.label for x in instances],
                        }
                    )
                    + "\n"
                )
    else:
        with open(dst.with_suffix(".json"), "w") as out:
            json.dump(dataset, out, default=lambda x: x.__dict__)
