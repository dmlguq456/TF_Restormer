import os
import torch
from loguru import logger
from .engine import Engine, EngineEval, EngineInfer, EngineInferFolder
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
        
    
    ''' Build Model '''
    # Call network model
    model_e = Model_Enhance(**config["model"])

    ''' Build Engine '''
    # Call gpu id & device
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')

    # Folder-based inference: skip DataLoader entirely
    if args.engine_mode == "infer" and getattr(args, 'input_dir', None):
        logger.info(f"[FolderInfer] input_dir={args.input_dir}  output_dir={args.output_dir}")
        engine = EngineInferFolder(args, config, model_e, gpuid, device)
        engine.run_infer_folder()
        return

    # Call DataLoader [train / valid / test / etc...]
    if args.engine_mode == "train":
        dataloaders = get_dataloaders(args, config["dataset_phase"], config["dataset"], config["dataloader"])
        engine = Engine(args, config, model_e, dataloaders, gpuid, device)
        engine.run()
        return

    # Support testset_key as a list for sequential evaluation
    testset_keys = config['dataset_test']['testset_key']
    if isinstance(testset_keys, str):
        testset_keys = [testset_keys]

    for i, key in enumerate(testset_keys):
        logger.info(f"===== [{i+1}/{len(testset_keys)}] testset_key: \"{key}\" =====")
        config['dataset_test']['testset_key'] = key
        dataloaders = get_dataloaders(args, config["dataset_phase"], config["dataset_test"], config["dataloader"])

        if args.engine_mode == "infer":
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

