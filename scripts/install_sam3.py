#!/usr/bin/env python3
"""Install SAM3 into the active Agent environment, preserving the CUDA pair."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise SystemExit(
            "Install the CUDA torch/torchvision pair from docs/SETUP.md first"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the Agent environment; check torch and the driver")
    if (
        torch.__version__.split("+")[0] != "2.7.0"
        or torchvision.__version__.split("+")[0] != "0.22.0"
    ):
        raise SystemExit("This deployment uses torch 2.7.0 + torchvision 0.22.0; see docs/SETUP.md")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-c",
            str(ROOT / "requirements/constraints-tested-linux.txt"),
            "-r",
            str(ROOT / "requirements/sam3.txt"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from sam3.model.sam3_image_processor import Sam3Processor; "
            "from sam3.model_builder import build_sam3_image_model; "
            "print('SAM3 imports passed; set SAM3_CHECKPOINT_PATH before running')",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
