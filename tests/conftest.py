import os
import sys
from pathlib import Path

core = Path(os.environ["COMFYUI_PATH"])
sys.path.insert(0, str(core))
sys.path.insert(0, str(core / "custom_nodes"))
from comfy.cli_args import args
args.cpu = True
