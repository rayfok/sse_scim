from os.path import realpath, abspath, dirname, exists
import urllib.request

from espresso_config import (
    ConfigNode, ConfigFlexNode, ConfigParam, TargetType, configure_logging
)
from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor
from mmda.predictors.heuristic_predictors.sentence_boundary_predictor import \
    PysbdSentenceBoundaryPredictor
from mmda.predictors.heuristic_predictors.dictionary_word_predictor import \
    DictionaryWordPredictor

from .noun_chunks import Seq2SeqFeaturesMapperWithFocusConfig
from .scorers import LongestCommonSubsequenceScorer

LOGGER = configure_logging(logger_name=__file__)


def word_alpha() -> str:
    url = 'https://github.com/dwyl/english-words/raw/master/words_alpha.txt'
    dst = realpath(abspath(dirname(__file__) + '/data/words_alpha.txt'))
    if not exists(dst):
        LOGGER.info(f'Downloading {url} for {dst}...')
        urllib.request.urlretrieve(url, dst)
    return dst


class MmdaParserConfig(ConfigNode):
    dpi: ConfigParam(int) = 72

    class parser(ConfigFlexNode):
        _target_: ConfigParam(TargetType) = PDFPlumberParser

    class rasterizer(ConfigFlexNode):
        _target_: ConfigParam(TargetType) = PDF2ImageRasterizer

    class lp(ConfigFlexNode):
        _target_: ConfigParam(TargetType) = \
            LayoutParserPredictor.from_pretrained
        config_path: ConfigParam(str) = 'lp://efficientdet/PubLayNet'
        label_map: ConfigParam(dict) = {1: "Text",
                                        2: "Title",
                                        3: "List",
                                        4: "Table",
                                        5: "Figure"}

    class vila(ConfigFlexNode):
        _target_: ConfigParam(TargetType) = IVILAPredictor.from_pretrained
        model_name_or_path: ConfigParam(str) = \
            'allenai/ivila-block-layoutlm-finetuned-docbank'
        added_special_sepration_token: ConfigParam(str) = "[BLK]"
        agg_level: ConfigParam(str) = "block"

    class sentence(ConfigFlexNode):
        _target_: ConfigParam(TargetType) = PysbdSentenceBoundaryPredictor

    class words(ConfigFlexNode):
        _target_: ConfigParam(TargetType) = DictionaryWordPredictor
        dictionary_file_path: ConfigParam(str) = word_alpha()

    class noun_chunks(Seq2SeqFeaturesMapperWithFocusConfig):
        reuse_spacy_pipeline: ConfigParam(bool) = True

    class scorer(ConfigFlexNode):
        _target_: ConfigParam(TargetType) = LongestCommonSubsequenceScorer


class CliMmdaParserConfig(MmdaParserConfig):
    path: ConfigParam(str)
    logging_level: ConfigParam(str) = 'WARN'
    save_path: ConfigParam(str) = None