import argparse
import importlib

try:
    import tf_restormer  # noqa: F401
except ImportError as e:
    import sys
    sys.exit(
        f"Error: {e}\n"
        "Install with: pip install -e .  (or: uv sync)"
    )

# Parse args
parser = argparse.ArgumentParser(
    description="Command to start PIT training, configured by .yaml files")
parser.add_argument(
    "--model",
    type=str,
    default="TF_Restormer",
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
main_module = importlib.import_module(f"tf_restormer.models.{args.model}.main")
main_module.main(args)