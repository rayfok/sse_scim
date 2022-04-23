
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
import json
import logging
from pathlib import Path
from typing import Dict, Sequence, Union
from espresso_config import \
    cli, instantiate, ConfigNode, ConfigFlexNode,\
    ConfigParam, configure_logging
import torch
import warnings
import tqdm

from mmda.types.annotation import SpanGroup, BoxGroup

LOGGER = configure_logging(logger_name=__file__,
                           logging_level=logging.INFO)

MMDA_PRED = 'mmda.predictors.%s_predictors'


class MmdaParserConfig(ConfigNode):
    dpi: ConfigParam(int) = 72

    class parser(ConfigFlexNode):
        _target_: ConfigParam(str) = \
            'mmda.parsers.pdfplumber_parser.PDFPlumberParser'

    class rasterizer(ConfigFlexNode):
        _target_: ConfigParam(str) = \
            'mmda.rasterizers.rasterizer.PDF2ImageRasterizer'

    class lp(ConfigFlexNode):
        _target_: ConfigParam(str) = ('mmda.predictors.lp_predictors.'
                                      'LayoutParserPredictor.from_pretrained')
        config_path: ConfigParam(str) = 'lp://efficientdet/PubLayNet'
        label_map: ConfigParam(dict) = {1: "Text",
                                        2: "Title",
                                        3: "List",
                                        4: "Table",
                                        5: "Figure"}

    class vila(ConfigFlexNode):
        _target_: ConfigParam(str) = ('mmda.predictors.hf_predictors.'
                                      'vila_predictor.IVILAPredictor.'
                                      'from_pretrained')
        model_name_or_path: ConfigParam(str) = \
            'allenai/ivila-block-layoutlm-finetuned-docbank'
        added_special_sepration_token: ConfigParam(str) = "[BLK]"
        agg_level: ConfigParam(str) = "block"

    class sentence(ConfigFlexNode):
        _target_: ConfigParam(str) = ('mmda.predictors.heuristic_predictors.'
                                      'sentence_boundary_predictor.'
                                      'PysbdSentenceBoundaryPredictor')


class VilaPredTypes(IntEnum):
    paragraph: int = 0
    title: int = 1
    equation: int = 2
    reference: int = 3
    section: int = 4
    list: int = 5
    table: int = 6
    caption: int = 7
    author: int = 8
    abstract: int = 9
    footer: int = 10
    date: int = 11
    figure: int = 12


def parse_pdf(path: Union[str, Path],
              config: MmdaParserConfig) -> Sequence[SpanGroup]:

    # cast if not path
    path = Path(path)
    LOGGER.info(f'Processing "{path}"...')

    # initialize all mmda objects
    parser = instantiate.now(config.parser)
    rasterizer = instantiate.now(config.rasterizer)
    layout_predictor = instantiate.now(config.lp)
    # visual_predictor = instantiate.now(config.vila)
    sentences_predictor = instantiate.now(config.sentence)

    LOGGER.info(f'Parsing "{path}" with pdfplumber.')
    doc = parser.parse(path)

    LOGGER.info(f'Rasterizing "{path}" to images.')
    images = rasterizer.rasterize(input_pdf_path=path, dpi=config.dpi)
    doc.annotate_images(images)

    LOGGER.info(f'Predicting layout for "{path}".')
    with torch.no_grad(), warnings.catch_warnings():
        # effdet complains about __floordiv__, so we ignore
        # warnings for a sec
        warnings.simplefilter("ignore")
        layout_regions = layout_predictor.predict(doc)
    doc.annotate(blocks=layout_regions)

    LOGGER.info(f'Found {len(doc.blocks)} blocks in "{path}".')

    all_sentences = []

    prog = tqdm.tqdm(desc='Extracting...', unit=' sentences')
    for page in doc.pages:
        for block in page.blocks:
            block_text = [' '.join(t.symbols) for t in block.tokens]
            slices = sentences_predictor.\
                split_token_based_on_sentences_boundary(block_text)

            for start, end in slices:
                span_groups = block.tokens[start:end]
                spans = list(
                    chain.from_iterable(sg.spans for sg in span_groups)
                )
                text = ' '.join(''.join(sg.symbols) for sg in span_groups)

                boxes = BoxGroup(boxes=[s.box for s in spans],
                                 id=len(all_sentences))

                sentence = SpanGroup(spans=list(spans),
                                     text=text,
                                     box_group=boxes,
                                     type=block.box_group.type,
                                     id=len(all_sentences))
                # TODO lucas@: figure out why doc doc needs to be set this way
                setattr(sentence, 'doc', doc)

                all_sentences.append(sentence)
                prog.update()
    prog.close()

    LOGGER.info(f'Done with "{path}"; {prog.n:,} sentences.')

    return all_sentences


class CliParserConfig(MmdaParserConfig):
    path: ConfigParam(str)
    logging_level: ConfigParam(str) = 'WARN'
    save_path: ConfigParam(str) = None


@cli(CliParserConfig)
def main(config: CliParserConfig):
    configure_logging(logging_level=config.logging_level)
    parsed = parse_pdf(path=config.path, config=config)
    if config.save_path:
        with open(config.save_path, 'w', encoding='utf-8') as f:
            for sentence in parsed:
                f.write(json.dumps(sentence.to_json()) + '\n')


if __name__ == '__main__':
    main()
