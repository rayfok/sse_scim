import logging
from typing import List, Type
import springs as sp

from .config import TrainConfig
from .utils.setup import RunSetup
from .data.data_module import DataModule

from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers.base import LightningLoggerBase

LOGGER = logging.getLogger(__file__)


def load_model_fresh_or_cache(config: TrainConfig) -> LightningModule:
    # If we are given a checkpoint, we load from it.
    # if not, we init the model from scratch
    if config.checkpoint:
        LOGGER.info(f'Loading from checkpoint {config.checkpoint}')
        model_cls: Type[LightningModule] = \
            sp.init.callable(config.model)  # type: ignore

        # if metrics have been redefined, we want to pass them as
        # arguments even if we're loading from checkpoint
        kwargs = ({'metrics': config.model.metrics}
                  if config.model.metrics else {})

        model = model_cls.load_from_checkpoint(config.checkpoint, **kwargs)
    elif config.backbone:
        LOGGER.info('Creating model from scratch...')
        model = sp.init.now(config.model, LightningModule, False)
    else:
        raise ValueError('`backbone` and `checkpoint` missing from config')

    return model


@sp.cli(TrainConfig, print_fn=rank_zero_info)
def main(config: TrainConfig):
    """Finetune a model. Example on how to run a specific experiment:

        python src/sse/finetune.py -c configs/csabstruct-scibert.yml
    """
    # setting up a bit of things
    run_artifacts = RunSetup(config)

    # Setup the data module here
    datamodule = sp.init.now(config.data, DataModule, _recursive_=False)

    # get the model
    model = load_model_fresh_or_cache(config)

    # Setup three loggers. By default, this logs to tensorboard,
    # and to csv file in the results directory.
    loggers: List[LightningLoggerBase] = [
        sp.init.now(config.loggers.text,
                    LightningLoggerBase,
                    save_dir=run_artifacts.log_dir),
        sp.init.now(config.loggers.graphic,
                    LightningLoggerBase,
                    save_dir=run_artifacts.tb_dir)
    ]

    # default callbacks include the run artifacts callback, which
    # does upload to s3 at the end of training (if configured), and
    # a callback to monitor the learning rate as a metric
    callbacks = [run_artifacts, LearningRateMonitor()]
    if config.checkpointing:
        LOGGER.info('Adding model checkpointing')
        callbacks.append(sp.init.now(config.checkpointing,
                                     Callback,
                                     dirpath=run_artifacts.ckpt_dir))
    if config.early_stopping:
        LOGGER.info('Adding early stopping')
        callbacks.append(sp.init.now(config.early_stopping))

    # don't run validation steps if the validation set is not provided
    num_sanity_val_steps = (config.trainer.num_sanity_val_steps
                            if config.data.valid_split_config else 0)
    LOGGER.info(f'Using {num_sanity_val_steps} sanity steps')

    # create a trainer
    trainer: Trainer = sp.init.now(
        config.trainer,
        Trainer,
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=num_sanity_val_steps
    )

    LOGGER.info('Starting to train!')
    trainer.fit(model, datamodule=datamodule)
    LOGGER.info('Training completed.')

    # # rerun on validation if available to get most up-to-date results
    # if config.data.valid_split_config:
    #     LOGGER.info('Evaluating on validation set!')
    #     trainer.validate(model, datamodule=datamodule)
    #     LOGGER.info('Evaluation completed.')

    # # doing some evaluation if possible
    # if config.data.test_split_config:
    #     LOGGER.info('Evaluating on test set!')
    #     trainer.test(model, datamodule=datamodule)
    #     LOGGER.info('Evaluation completed.')


if __name__ == "__main__":
    main()
