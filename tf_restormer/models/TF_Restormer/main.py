import argparse
import os
import torch
from loguru import logger
from .engine import Engine
from .dataset import get_dataloaders
from .model import Model
from tf_restormer.utils.decorators import logger_wraps


# Setup logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/system_log.log")
logger.add(log_file_path, level="DEBUG", mode="w")

@logger_wraps()
def main(args):
    """Entry point for TF_Restormer training, evaluation, and inference."""
    from tf_restormer._config import load_config
    config_name = getattr(args, 'config', 'baseline.yaml')
    yaml_dict = load_config("TF_Restormer", config_name)
    config = yaml_dict["config"]

    model_e = Model(**config["model"])
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')

    dataloaders = get_dataloaders(args, config["dataset_phase"], config["dataset"], config["dataloader"])
    engine = Engine(args, config, model_e, dataloaders, gpuid, device)
    engine.run()


