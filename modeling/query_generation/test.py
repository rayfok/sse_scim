import springs as sp

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from .finetune import load_model_fresh_or_cache
from .config import TestConfig
from .data.data_module import DataModule


@sp.cli(TestConfig, print_fn=rank_zero_info)
def main(config: TestConfig):
    """Test a model"""

    # setup logging
    sp.configure_logging()

    # get the model
    model = load_model_fresh_or_cache(config)

    # data is initialized here
    datamodule = sp.init.now(config.data, DataModule, False)

    # make a "trainer" (even though we are just testing...)
    # we need to set the logger explicitly to None, otherwise a tensorboard
    # logger will be created.
    trainer: Trainer = sp.init.now(config.trainer, Trainer, logger=None)

    # rerun on validation if available to get most up-to-date results
    if config.data.valid_split_config:
        trainer.validate(model, datamodule=datamodule)

    # doing testing
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
