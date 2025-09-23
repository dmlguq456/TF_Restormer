import os
import torch
from loguru import logger
from .engine import Engine, EngineEval, EngineInfer
from .dataset import get_dataloaders
from .model import Model_Enhance
from utils import util_system
from utils.decorators import *


# Setup logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/system_log.log")
logger.add(log_file_path, level="DEBUG", mode="w")

@logger_wraps()
def main(args):
    
    ''' Build Setting '''
    # Call configuration file based on args.config
    config_name = getattr(args, 'config', 'baseline.yaml')  # default to baseline.yaml if not specified
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", config_name)
    yaml_dict = util_system.parse_yaml(yaml_path)
    logger.info(f"Using config file: {config_name}")
    
    
    # Run wandb and get configuration
    config = yaml_dict["config"] # wandb login success or fail
        
    
    # Call DataLoader [train / valid / test / etc...]
    if args.engine_mode == "train":
        dataloaders = get_dataloaders(args, config["dataset_phase"], config["dataset"], config["dataloader"])
    else:
        dataloaders = get_dataloaders(args, config["dataset_phase"], config["dataset_test"], config["dataloader"])
        
    
    ''' Build Model '''
    # Call network model
    model_e = Model_Enhance(**config["model"])

    ''' Build Engine '''
    # Call gpu id & device
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')

    # Call & Run Engine
    logger.info(f"Training Phase: \"{config['train_phase']}\" and Dataset Phase: \"{config['dataset_phase']}\" ")
    if args.engine_mode == "train":
        engine = Engine(args, config, model_e, dataloaders, gpuid, device)
        engine.run()
    elif args.engine_mode == "infer":
        engine = EngineInfer(args, config, model_e, dataloaders, gpuid, device)
        engine.run_infer()
    elif args.engine_mode == "eval":
        engine = EngineEval(args, config, model_e, dataloaders, gpuid, device)
        engine.run_eval()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_mode", type=str, default="train", 
                        choices=["train", "test", "inference_sample"],
                        help="Engine mode: train, test, or inference_sample")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use")
    parser.add_argument("--sample_file", type=str, default=None, help="Sample file for inference")
    args = parser.parse_args()
    main(args)

