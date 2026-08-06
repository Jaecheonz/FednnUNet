import json
import os
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch


def seed_everything(
    seed: int,
    deterministic: bool = True,
) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # warn_only=True prevents unsupported deterministic CUDA
        # operations from terminating the entire experiment.
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    else:
        torch.backends.cudnn.deterministic = False


def write_json(
    path: Union[str, Path],
    payload: Any,
) -> None:
    """Write a JSON file safely, creating parent directories if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    # Atomic replacement avoids leaving a partially written JSON file.
    temporary_path.replace(output_path)