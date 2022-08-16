
from dataclasses import dataclass
import springs as sp

from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

LOGGING = sp.configure_logging(__file__)


@dataclass
class SlicerConfig:
    num_sentences: int = -1
    num_tokens: int = -1


@sp.make_flexy
@dataclass
class ModelConfig:
    _target_: str = sp.Target.to_string(AutoModel)
    pretrained_model_or_path: str = sp.MISSING


@sp.make_flexy
@dataclass
class TokenizerConfig:
    _target_: str = sp.Target.to_string(AutoTokenizer)
    pretrained_model_or_path: str = sp.MISSING


@dataclass
class BaseScorerConfig:
    slicer: SlicerConfig = SlicerConfig()
    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()



class BaseScorer:
    def __init__(self, config: BaseScorerConfig):
        self.config = config
        self.model = sp.init.now(config.model, PreTrainedModel)
        self.tokenizer = sp.init.now(config.tokenizer, PreTrainedTokenizerBase)
