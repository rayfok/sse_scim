import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import springs as sp
from cached_path import cached_path
from pdf2sents.pipeline import Pipeline, PipelineConfig
from pdf2sents.typed_predictors import TypedBlockPredictor
from pdf2sents.visualization import VizAny

from sse_skimming.heuristics import *
from sse_skimming.predictor import Predictor, PredictorConfig


class OpacityCalculator:
    def __init__(
        self, max_opacity: float, min_opacity: float, threshold: float,
    ):
        self.max_opacity = max_opacity
        self.threshold = threshold
        self.min_opacity = min_opacity

    def __call__(self, score: float):
        if score < self.threshold:
            return 0

        # normalize score to [0, 1]
        score = (score - self.threshold) / (1 - self.threshold)

        # scale to [min_opacity, max_opacity]
        return self.min_opacity + score * (self.max_opacity - self.min_opacity)


@dataclass
class PipelineObjConfig:
    _target_: str = sp.Target.to_string(Pipeline)
    config: PipelineConfig = PipelineConfig()


@dataclass
class PredictorObjConfig:
    _target_: str = sp.Target.to_string(Predictor)
    config: PredictorConfig = PredictorConfig()
    artifacts_dir: str = str(
        cached_path(
            url_or_filename=(
                "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/"
                "lucas/skimming/MiniLMv2-L6-H384-BERT-Large-csabstruct"
                "_2022-04-16_02-45-24_epoch_2-step_993.ckpt.hf.tar.gz"
            ),
            extract_archive=True,
        )
    )


@dataclass
class OpacityCalculatorConfig:
    _target_: str = sp.Target.to_string(OpacityCalculator)
    threshold: float = 0.7
    max_opacity: float = 0.4
    min_opacity: float = 0.1


@dataclass
class VizConfig:
    _target_: str = sp.Target.to_string(VizAny)
    color_map: Dict[str, str] = field(
        default_factory=lambda: {
            "background": "green",
            "method": "blue",
            "objective": "red",
            "other": "grey",
            "result": "yellow",
        }
    )


@dataclass
class SSESkimmingConfig:
    pipeline: PipelineObjConfig = PipelineObjConfig()
    predictor: PredictorObjConfig = PredictorObjConfig()
    src: str = sp.MISSING
    dst: Optional[str] = None

    valid_types: List[str] = field(
        default_factory=lambda: [
            TypedBlockPredictor.Text,
            TypedBlockPredictor.ListType,
            TypedBlockPredictor.Abstract,
        ]
    )
    viz: VizConfig = VizConfig()
    opacity: OpacityCalculatorConfig = OpacityCalculatorConfig()


@sp.cli(SSESkimmingConfig)
def main(config: SSESkimmingConfig):
    # Pipeline is responsible for parsing pdfs, while the predictor
    # predicts the label for each sentence.
    pipeline = sp.init.now(config.pipeline, Pipeline)
    predictor = sp.init.now(config.predictor, Predictor)

    # this returns an mmda object annotated with sentence
    doc = pipeline.run(config.src)

    # this is used to map each sentence to an enclosing block
    block_to_sents = {}
    for block in doc.typed_blocks:
        block_to_sents[(block.spans[0].start, block.spans[0].end, block.uuid)] = []

    # we only call the predictor on sentences that are of type in
    # config.valid_types; by default, this is main blocks of text and
    # lists.
    to_predict: Dict[str, List[str]] = {"text": []}
    ref_sents = []
    for sent in doc.typed_sents:  # type: ignore
        if sent.type in config.valid_types:
            to_predict["text"].append(clean_sentence(sent.text))
            ref_sents.append(sent)
            print(sent)
            print()

        # We map each sentence/spangroup to an enclosing block based on their start spans
        for block_span_range in block_to_sents.keys():
            block_span_start, block_span_end, block_uuid = block_span_range
            if sent.spans[0].start >= block_span_start:
                block_to_sents[block_span_range].append(sent)
                sent.block_uuid = block_uuid
                break

    # get predictions for this document
    predictions = predictor.predict_one(to_predict)

    # Classify sentences for "novelty"
    novelty_predictions = classify_novelty(ref_sents)

    # we get the labels and probabilities for each sentence
    # here; the opacity_calculator returns a score > 0 if the
    # sentence is relevant, and 0 otherwise.
    opacity_calculator = sp.init.now(config.opacity, OpacityCalculator)
    sents = []
    for sent, pred in zip(ref_sents, predictions):
        pred = {k: round(v, 5) for k, v in pred.items()}
        label, score = max(pred.items(), key=lambda x: x[1])
        sent_opacity = opacity_calculator(score)
        labeled_sent = {
            "id": sent.id,
            "text": clean_sentence(sent.text),
            "pred": pred,
            "label": label,
            "score": score,
            "boxes": [
                {
                    "left": box.l,
                    "top": box.t,
                    "width": box.w,
                    "height": box.h,
                    "page": box.page,
                }
                for box in sent.box_group.boxes
            ],
            "block_id": sent.block_uuid
        }
        if novelty_predictions[sent.id]:
            labeled_sent["label"] = "novelty"
            labeled_sent["score"] = 1
            labeled_sent["pred"]["novelty"] = 1
            sents.append(labeled_sent)
        elif sent_opacity > 0 and label not in ["background", "other"]:
            sents.append(labeled_sent)


    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.basename(config.src).replace("pdf", "json")
    with open(os.path.join(OUTPUT_DIR, output_file), "w") as out:
        json.dump(sents, out, indent=2)


if __name__ == "__main__":
    main()
