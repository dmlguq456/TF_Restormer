import argparse
import importlib

# Parse args
parser = argparse.ArgumentParser(
    description="Command to start PIT training, configured by .yaml files")
parser.add_argument(
    "--model",
    type=str,
    default="IQformer_v6_0_0",
    dest="model",
    help="Insert model name")
parser.add_argument(
    "--engine_mode",
    choices=["train", "eval", "infer", "infer_sample"],
    default="train",
    help="This option is used to chooose the mode")
parser.add_argument(
    "--sample_file",
    type=str,
    default=None,
    help="Sample wav file for inference"
)
parser.add_argument(
    "--config",
    type=str,
    default="baseline.yaml",
    help="Config file name (with .yaml extension)"
)
parser.add_argument(
    "--dump_path",
    type=str,
    default=None,
    help="Path to save inference results"
)
args = parser.parse_args()

# Call target model
main_module = importlib.import_module(f"models.{args.model}.main")
main_module.main(args)