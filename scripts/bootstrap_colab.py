#!/usr/bin/env python3
"""Install the training stack without replacing Colab's CUDA-matched PyTorch."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_TORCH_PACKAGES = ("torchvision", "torchaudio", "torchtext")


def public_version(version: str) -> str:
    """Return a PEP 440 public version without the CUDA local suffix."""

    return str(Version(version).public)


def required_torch_specifier(distribution: str) -> str | None:
    """Return an active torch constraint declared by a distribution, if present."""

    try:
        requirements = importlib.metadata.requires(distribution) or []
    except importlib.metadata.PackageNotFoundError:
        return None

    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        if requirement.name.lower() != "torch":
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        return str(requirement.specifier) or None
    return None


def check_preinstalled_stack(torch_version: str) -> None:
    """Reject a runtime that a previous pip command has already corrupted."""

    installed = Version(public_version(torch_version))
    torchvision_specifier = required_torch_specifier("torchvision")
    if torchvision_specifier and installed not in Requirement(
        f"torch{torchvision_specifier}"
    ).specifier:
        raise SystemExit(
            "FAIL: this Colab runtime is already inconsistent: "
            f"torch {installed}, torchvision requires torch{torchvision_specifier}. "
            "Delete the runtime and start again with the v2 notebook."
        )


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: select a Colab GPU runtime before installing packages")

    original_torch = public_version(torch.__version__)
    check_preinstalled_stack(torch.__version__)

    with tempfile.TemporaryDirectory(prefix="uds-colab-") as temp_dir:
        constraint = Path(temp_dir) / "constraints.txt"
        constraint.write_text(f"torch=={original_torch}\n", encoding="utf-8")

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "-q",
                *OPTIONAL_TORCH_PACKAGES,
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--constraint",
                str(constraint),
                "--requirement",
                str(ROOT / "requirements-colab.txt"),
            ]
        )

    import accelerate
    import bitsandbytes
    import datasets
    import peft
    import transformers
    import trl

    final_torch = public_version(torch.__version__)
    if final_torch != original_torch:
        raise SystemExit(
            f"FAIL: pip replaced Colab torch {original_torch} with {final_torch}"
        )

    result = {
        "status": "ready",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
