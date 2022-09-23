from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback


log = logging.getLogger(__name__)

# CONFIG_PATH = Path.cwd() / '../config'
CONFIG_PATH = '/home/jialvzou/cvt_io/config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):

    # checkpoints = "/home/jialvzou/cvt_io/logs/cross_view_transformers_test/version_1/checkpoints/model.ckpt"
    # return checkpoints

    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)
    
    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    # model_module 一些设置，如backbone，loss，metrics，optimizer
    # data_module 初始化dataloder
    # viz_fn 可视化，默认为null
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)

    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)



if __name__ == '__main__':
    main()
