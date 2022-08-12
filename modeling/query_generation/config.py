import datasets
from typing import Any, Dict, Optional, List, Union

import springs as sp

from datasets.features import Features, Sequence, ClassLabel
from datasets.arrow_dataset import Dataset

from transformers.models.auto.modeling_auto \
    import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from smashed.base.pipeline import Pipeline


PL = 'pytorch_lightning'


@sp.make_flexy
@sp.dataclass
class TargetConfig:
    _target_: str = sp.MISSING


@sp.make_flexy
@sp.dataclass
class LoaderConfig:
    _target_: str = 'datasets.load_dataset'
    path: Optional[str] = None
    split: str = sp.MISSING
    task: Optional[str] = None


@sp.make_flexy
@sp.dataclass
class HuggingFaceModuleConfig:
    _target_: str = sp.MISSING
    pretrained_model_name_or_path: str = '${backbone}'


@sp.make_flexy
@sp.dataclass
class DataSplitConfig:
    _target_: str = 'smashed.base.pipeline.Pipeline.chain'
    loader: LoaderConfig = LoaderConfig()
    mappers: List[Dict[str, Any]] = sp.field(default_factory=list)
    collator: TargetConfig = TargetConfig()


@sp.dataclass
class TrainDataConfig:
    _target_: str = 'sse.data.DataModule'
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    train_split_config: DataSplitConfig = DataSplitConfig()
    valid_split_config: Optional[DataSplitConfig] = None
    test_split_config: Optional[DataSplitConfig] = None


@sp.dataclass
class EnvironmentConfig:
    root_dir: Optional[str] = '~/plruns'
    run_name: Optional[str] = 'sse'
    s3_prefix: Optional[str] = None
    seed: int = 5663


@sp.dataclass
class ModelConfig:
    _target_: str = 'sse.models.TokenClassificationModule'
    tokenizer: HuggingFaceModuleConfig = HuggingFaceModuleConfig(
        _target_=sp.Target.to_string(AutoTokenizer.from_pretrained)
    )
    transformer: HuggingFaceModuleConfig = HuggingFaceModuleConfig(
        _target_=sp.Target.to_string(AutoModelForSequenceClassification.
                                     from_pretrained)
    )
    val_loss_label: str = 'val_loss'
    loss: Optional[TargetConfig] = None
    optimizer: Optional[TargetConfig] = None
    scheduler: Optional[TargetConfig] = None
    transfer: Optional[TargetConfig] = None
    metrics: Dict[str, TargetConfig] = sp.field(default_factory=dict)


@sp.make_flexy
@sp.dataclass
class CheckpointConfig:
    _target_: str = f'{PL}.callbacks.ModelCheckpoint'
    mode: str = 'min'
    monitor: str = '${model.val_loss_label}'
    verbose: bool = False


@sp.make_flexy
@sp.dataclass
class GraphicLoggerConfig:
    _target_: str = f'{PL}.loggers.TensorBoardLogger'
    log_graph: bool = False


@sp.make_flexy
@sp.dataclass
class TextLoggerConfig:
    _target_: str = f'{PL}.loggers.CSVLogger'
    name: str = ''
    version: str = ''


@sp.dataclass
class LoggersConfig:
    graphic: GraphicLoggerConfig = GraphicLoggerConfig()
    text: TextLoggerConfig = TextLoggerConfig()


@sp.make_flexy
@sp.dataclass
class EarlyStoppingConfig:
    _target_: str = f'{PL}.callbacks.EarlyStopping'
    check_on_train_epoch_end: bool = False
    min_delta: float = 0
    mode: str = 'min'
    monitor: str = '${model.val_loss_label}'
    patience: int = 10
    verbose: bool = False


@sp.make_flexy
@sp.dataclass
class TrainerConfig:
    _target_: str = f'{PL}.Trainer'
    accelerator: str = 'auto'
    devices: int = 1
    max_epochs: int = -1
    max_steps: int = -1
    precision: int = 32
    log_every_n_steps: int = 50
    limit_train_batches: Optional[float] = 1.0
    val_check_interval: Union[int, float] = 1
    gradient_clip_val: Optional[float] = None
    strategy: Optional[Dict[str, Any]] = None
    num_sanity_val_steps: int = 2


@sp.dataclass
class TrainConfig:
    # base strings to control where models and tokenizers come from
    backbone: Optional[str] = None
    checkpoint: Optional[str] = None

    # this controls training environment and data
    env: EnvironmentConfig = EnvironmentConfig()
    data: TrainDataConfig = TrainDataConfig()
    model: ModelConfig = ModelConfig()
    loggers: LoggersConfig = LoggersConfig()
    trainer: TrainerConfig = TrainerConfig()

    # optional configurations to deal with checkpointing and early stopping
    checkpointing: Optional[CheckpointConfig] = None
    early_stopping: Optional[EarlyStoppingConfig] = None


@sp.dataclass
class TestDataConfig(TrainDataConfig):
    train_split_config: Optional[DataSplitConfig] = None    # type: ignore
    valid_split_config: Optional[DataSplitConfig] = None
    test_split_config: DataSplitConfig = DataSplitConfig()


@sp.dataclass
class TestConfig(TrainConfig):
    data: TestDataConfig = TestDataConfig()


AllFeaturesTypes = Union[
    Features, List[Features], Dict[str, Features], Sequence
]


@sp.register('guess_num_classes_from_dataset')
class guess_num_classes_from_dataset:
    @classmethod
    def find_all_features(cls, features: AllFeaturesTypes) -> List[Features]:
        discovered_features = []
        if isinstance(features, dict):
            for fs in features.values():
                discovered_features.extend(cls.find_all_features(fs))
        elif isinstance(features, list):
            for fs in features:
                discovered_features.extend(cls.find_all_features(fs))
        elif isinstance(features, datasets.features.Sequence):
            discovered_features.extend(cls.find_all_features(features.feature))
        else:
            discovered_features.append(features)
        return discovered_features

    def __new__(cls, config: LoaderConfig) -> int:  # type: ignore
        dataset: Dataset = sp.init.now(config, Dataset)

        features = dataset.info.features or []
        discovered_features = cls.find_all_features(features)
        discovered_class_labels = [
            f for f in discovered_features if isinstance(f, ClassLabel)
        ]
        cl = len(discovered_class_labels)

        if cl == 0:
            raise ValueError('Could not find any class labels')
        elif cl > 1:
            raise ValueError(f'Ambiguous: found {cl} class labels')

        return discovered_class_labels[0].num_classes
