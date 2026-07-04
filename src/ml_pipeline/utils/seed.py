"""Global seeding so runs are reproducible across numpy, random, and torch."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline may touch.

    Called once at pipeline start. Torch is imported lazily so classical-only
    runs don't pay the import cost.

    Args:
        seed: the seed applied to ``random``, ``numpy``, ``PYTHONHASHSEED`` and,
            when available, ``torch`` (CPU and CUDA).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional at runtime for classical-only use
        logger.debug("torch not installed; skipping torch seeding")
    logger.info("Global seed set to %d", seed)
