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
class VizConfig:
    color_map: Dict[str, str] = field(default_factory=lambda: {
            'background': 'green',
            'method': 'blue',
            'objective': 'red',
            'other': 'grey',
            'result': 'yellow',
        })
    threshold: float = 0.6
    max_opacity: float = 0.4


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


def get_opacity(max_opacity: float, score: float, threshold: float) -> float:
    return max_opacity * (1 - (score - threshold) / (1 - threshold))


@sp.cli(SSESkimmingConfig)
def main(config: SSESkimmingConfig):
    pipeline = sp.init.now(config.pipeline, Pipeline)
    predictor = sp.init.now(config.predictor, Predictor)

    doc = pipeline.run(config.src)

    to_predict = {'text': []}
    ref_sents = []
    for sent in doc.typed_sents:    # type: ignore
        if sent.type in config.valid_types:
            to_predict['text'].append(sent.text)
            ref_sents.append(sent)

    predictions = predictor.predict_one(to_predict)

    to_visualize = []
    label_opacity = []
    for sent, pred in zip(ref_sents, predictions):
        label, score = max(pred.items(), key=lambda x: x[1])
        if score > config.viz.threshold:
            sent = deepcopy(sent)
            sent.type = label
            to_visualize.append(sent)
            label_opacity.append(
                get_opacity(max_opacity=config.viz.max_opacity,
                            score=score,
                            threshold=config.viz.threshold)
            )

    viz = VizAny(color_map=config.viz.color_map)
    viz(doc=doc,
        path=config.dst or Path(config.src).with_suffix('.png'),
        spans=to_visualize,
        opacity=label_opacity)


if __name__ == '__main__':
    main()
