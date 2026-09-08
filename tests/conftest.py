import os
import importlib.util
import sys
from pathlib import Path

import torch

core = Path(os.environ["COMFYUI_PATH"])
sys.path.insert(0, str(core))
sys.path.insert(0, str(core / "custom_nodes"))
from comfy.cli_args import args

args.cpu = not (
    os.environ.get("LLADA_IMAGE_PARITY_DEVICE") == "cuda"
    and torch.cuda.is_available()
)

# Test-only alias: import this checkout regardless of its installation folder name.
package = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("llada_image_t8", package / "__init__.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
