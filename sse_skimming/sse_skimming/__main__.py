from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pdf2sents.pipeline import Pipeline, PipelineConfig
from pdf2sents.typed_predictors import TypedBlockPredictor
from pdf2sents.visualization import VizAny
from sse_skimming.predictor import PredictorConfig, Predictor

from cached_path import cached_path


import springs as sp


class OpacityCalculator:
    def __init__(
        self,
        max_opacity: float,
        min_opacity: float,
        threshold: float,
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
                'https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/'
                'lucas/skimming/MiniLMv2-L6-H384-BERT-Large-csabstruct'
                '_2022-04-16_02-45-24_epoch_2-step_993.ckpt.hf.tar.gz'
            ),
            extract_archive=True
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
    color_map: Dict[str, str] = field(default_factory=lambda: {
            'background': 'green',
            'method': 'blue',
            'objective': 'red',
            'other': 'grey',
            'result': 'yellow',
        })


@dataclass
class SSESkimmingConfig:
    pipeline: PipelineObjConfig = PipelineObjConfig()
    predictor: PredictorObjConfig = PredictorObjConfig()
    src: str = sp.MISSING
    dst: Optional[str] = None

    valid_types: List[str] = field(
        default_factory=lambda: [TypedBlockPredictor.Text,
                                 TypedBlockPredictor.ListType]
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

    # we only call the predictor on sentences that are of type in
    # config.valid_types; by default, this is main blocks of text and
    # lists.
    to_predict: Dict[str, List[str]] = {'text': []}
    ref_sents = []
    for sent in doc.typed_sents:    # type: ignore
        if sent.type in config.valid_types:
            to_predict['text'].append(sent.text)
            ref_sents.append(sent)

    # get predictions for this document
    predictions = predictor.predict_one(to_predict)

    # we get the labels and probabilities for each sentence
    # here; the opacity_calculator returns a score > 0 if the
    # sentence is relevant, and 0 otherwise.
    opacity_calculator = sp.init.now(config.opacity, OpacityCalculator)
    to_visualize = []
    labels_opacity = []
    for sent, pred in zip(ref_sents, predictions):
        label, score = max(pred.items(), key=lambda x: x[1])
        sent_opacity = opacity_calculator(score)
        if sent_opacity > 0:
            sent = deepcopy(sent)
            sent.type = label
            to_visualize.append(sent)
            labels_opacity.append(sent_opacity)

    # visualize the sentences
    viz = sp.init.now(config.viz, VizAny)
    viz(doc=doc,
        path=config.dst or Path(config.src).with_suffix('.png'),
        spans=to_visualize,
        opacity=labels_opacity)


if __name__ == '__main__':
    main()
