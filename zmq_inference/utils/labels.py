"""ImageNet label loader using package-bundled data."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, Union

# importlib.resources is the correct way to access package data in Python 3.9+.
try:
    from importlib.resources import files  # Python 3.9+
except ImportError:  # pragma: no cover
    from importlib_resources import files  # type: ignore[no-redef]


@lru_cache(maxsize=1)
def load_imagenet_labels() -> Dict[str, str]:
    """Load the bundled ImageNet class labels.

    The JSON file maps string-formatted class indices (e.g. ``"0"``) to
    human-readable labels (e.g. ``"tench"``).

    Returns:
        A dict mapping ``str(class_id)`` to label string.

    Note:
        The result is cached after the first call so the file is read only once.
    """
    data_pkg = files("zmq_inference.data")
    label_file = data_pkg.joinpath("imagenet_classes.json")
    return json.loads(label_file.read_text(encoding="utf-8"))


def get_label(class_id: Union[int, str], labels: Dict[str, str]) -> str:
    """Look up a human-readable label for a given class index.

    Args:
        class_id: Integer or string class index (e.g. ``285`` or ``"285"``).
        labels:   Label dict as returned by :func:`load_imagenet_labels`.

    Returns:
        The label string, or ``"unknown"`` when the index is not found.
    """
    return labels.get(str(class_id), "unknown")
