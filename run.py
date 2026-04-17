import argparse
import importlib
import os
from loguru import logger

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
    description="TF_Restormer CLI -- training, evaluation, and inference for speech enhancement")
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
parser.add_argument("--input", type=str, default=None,
                    help="Input audio file or directory for inference")
parser.add_argument("--output", type=str, default=None,
                    help="Output directory for inference results")
parser.add_argument("--gpuid", type=str, default="0",
                    help="GPU device id(s)")
args = parser.parse_args()

if args.engine_mode == "infer_sample":
    import warnings
    warnings.warn(
        "Deprecated: --engine_mode infer_sample is replaced by --engine_mode infer --input <file>. "
        "Scheduled for removal in the next major release.",
        DeprecationWarning, stacklevel=2
    )
    args.engine_mode = "infer"

# Validate model name before any filesystem operations
from tf_restormer._config import _VARIANT_MAP  # noqa: E402
if args.model not in _VARIANT_MAP:
    import sys
    sys.exit(f"Unknown model: {args.model!r}. Choose from: {set(_VARIANT_MAP)}")

# ---- Setup model-level file logger before dispatch ----
_model_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tf_restormer", "models", args.model,
)
_log_file = os.path.join(_model_dir, "log", "system_log.log")
os.makedirs(os.path.dirname(_log_file), exist_ok=True)
logger.add(_log_file, level="DEBUG", mode="w")


def resolve_module_path(model_name, module_name):
    return f"tf_restormer.models.{model_name}.{module_name}"


if args.engine_mode in ("infer", "eval"):
    try:
        infer_module = importlib.import_module(resolve_module_path(args.model, "main_infer"))
    except ModuleNotFoundError as e:
        if e.name is not None and "main_infer" not in e.name:
            raise
        logger.warning(
            f"main_infer module not found for model '{args.model}' "
            f"(engine_mode='{args.engine_mode}'). "
            "Falling back to main.main — this model does not support eval/infer mode "
            "and will run training instead. If this is unintended, check your --model argument."
        )
        if args.engine_mode != "train":
            raise RuntimeError(
                f"engine_mode='{args.engine_mode}' requires main_infer module, "
                f"but it was not found for model '{args.model}'."
            )
        main_module = importlib.import_module(resolve_module_path(args.model, "main"))
        main_module.main(args)
    else:
        infer_module.main_infer(args)
else:
    main_module = importlib.import_module(resolve_module_path(args.model, "main"))
    main_module.main(args)