from espresso_config import ConfigNode, ConfigFlexNode, ConfigParam
from os.path import realpath, abspath, dirname

from .noun_chunks import Seq2SeqFeaturesMapperWithFocusConfig

DATA_PATH = realpath(abspath(dirname(__file__) + '/data'))


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

    class words(ConfigFlexNode):
        _target_: ConfigParam(str) = ('mmda.predictors.heuristic_predictors.'
                                      'dictionary_word_predictor.'
                                      'DictionaryWordPredictor')
        dictionary_file_path: ConfigParam(str) = f'{DATA_PATH}/words_alpha.txt'

    class noun_chunks(Seq2SeqFeaturesMapperWithFocusConfig):
        reuse_spacy_pipeline: ConfigParam(bool) = True


class CliMmdaParserConfig(MmdaParserConfig):
    path: ConfigParam(str)
    logging_level: ConfigParam(str) = 'WARN'
    save_path: ConfigParam(str) = None
