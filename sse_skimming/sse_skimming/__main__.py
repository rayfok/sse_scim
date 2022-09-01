import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from email.policy import default
from pathlib import Path
from typing import Dict, List, Optional, Type

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

    # if os.path.exists("author_verbs.txt"):
    #     with open("author_verbs.txt", "r") as f:
    #         verbs = {line.strip() for line in f}
    # else:
    #     verbs = set()
    # for sent in doc.typed_sents:
    #     sent.text = clean_sentence(sent.text)
    #     if sent.type in config.valid_types:
    #         if sentence_has_author_intent(sent):
    #             verbs.update(set(extract_verbs(sent)))
    # with open("author_verbs.txt", "w") as out:
    #     for verb in sorted(list(verbs)):
    #         out.write(verb + "\n")
    # return

    for sent in doc.typed_sents:  # type: ignore
        if sent.type in config.valid_types:
            sent.text = clean_sentence(sent.text)
            to_predict["text"].append(sent.text)
            ref_sents.append(sent)

        # We map each sentence/spangroup to an enclosing block based on their start spans
        for block_span_range in block_to_sents.keys():
            block_span_start, block_span_end, block_uuid = block_span_range
            if sent.spans[0].start <= block_span_end:
                block_to_sents[block_span_range].append(sent)
                sent.block_uuid = block_uuid
                break

    # assign sections to sentences
    sent_sect_map, sect_box_map = assign_sections_to_sentences(doc.typed_sents)
    for sent in ref_sents:
        sent.section = sent_sect_map[sent.id]

    # get trained model predictions for this document
    predictions = predictor.predict_one(to_predict)

    highlights = []

    # get all sentences with "author" statements
    author_sentences = {s.id for s in ref_sents if sentence_has_author_intent(s)}

    # classify sentences for "objective"
    objective_predictions = classify_objective_batch(ref_sents)

    # classify sentences for "novelty"
    novelty_predictions = classify_novelty_batch(ref_sents)

    # classify sentences for "result"
    result_predictions = classify_result_batch(ref_sents)

    # model predictions
    predictions = predictor.predict_one(to_predict)

    # we want to disable highlights in the abstract only if it's
    # properly extracted by the layout parser (i.e., should be < 300 words)
    abstract_sents = [s for s in ref_sents if is_sentence_in_section(s, ["abstract"])]
    abstract_length = sum(len(s.text.split()) for s in abstract_sents)
    abstract_correctly_extracted = abstract_length <= 350

    block_discount_factor = 0.1
    score_threshold = 0.8
    model_score_threshold = 0.85
    model_score_factor = 0.1
    block_to_sents = defaultdict(set)
    for i, sent in enumerate(ref_sents):
        num_tokens = len(sent.text.split())
        if num_tokens > 80 or num_tokens < 5:
            print("[Abnormal sentence length]", sent.text)
            continue

        if is_sentence_in_section(sent, ["acknowledgement"]):
            print("[Acknowledgement]", sent.text)
            continue

        if "github.com" in sent.text:
            print("[Github link]", sent.text)
            continue

        # if abstract_correctly_extracted and is_sentence_in_section(sent, ["abstract"]):
        #     print("[Skipping abstract]", sent.text)
        #     continue

        score = 0
        labels = []
        if sent.id in author_sentences:
            score += 1
            if sent.block_uuid in block_to_sents:
                score -= max(
                    0, block_discount_factor * len(block_to_sents[sent.block_uuid])
                )
            block_to_sents[sent.block_uuid].add(sent.id)

            if sent.id in objective_predictions:
                labels.append("objective")
                score += 1

            if sent_contains_contribution(sent):
                labels.append("novelty")
                score += 1

            if sent.id in result_predictions or is_sentence_in_section(
                sent, ["result"]
            ):
                if not is_sentence_in_section(sent, ["method", "approach", "setup"]):
                    labels.append("result")
                    score += 1

            if sent.id in novelty_predictions:
                labels.append("novelty")
                score += 1

            if not labels:
                labels.append("method")

            if is_sentence_in_section(sent, ["abstract", "introduction"]):
                score += 2

            model_label, model_score = max(predictions[i].items(), key=lambda x: x[1])
            if model_score > model_score_threshold:
                if model_label in ["objective", "method", "result"]:
                    if model_label == labels[0]:
                        score += model_score * model_score_factor

        # print(sent.section, sent.text, labels, score, "\n")

        if score >= score_threshold:
            highlight = {
                "id": sent.id,
                "text": sent.text,
                "section": sent.section,
                "label": labels[0],
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
                "block_id": sent.block_uuid,
            }
            highlights.append(highlight)

    sections = [
        {
            "section": section,
            "box": {
                "left": box.l,
                "top": box.t,
                "width": box.w,
                "height": box.h,
                "page": box.page,
            },
        }
        for section, box in sect_box_map.items()
    ]

    # Write output
    output_file = os.path.basename(config.src).replace("pdf", "json")

    HIGHLIGHTS_DIR = "output/highlights"
    os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)
    with open(os.path.join(HIGHLIGHTS_DIR, output_file), "w") as out:
        json.dump(highlights, out, indent=2)

    SECTIONS_DIR = "output/sections"
    os.makedirs(SECTIONS_DIR, exist_ok=True)
    with open(os.path.join(SECTIONS_DIR, output_file), "w") as out:
        json.dump(sections, out, indent=2)


if __name__ == "__main__":
    main()
