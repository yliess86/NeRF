from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class History:
    """"Training history

    Arguments:
        train (List[Iterable[float]]): training history
        val (List[Iterable[float]]): validation history
        test (Iterable[float]): testing history
    """
    train: List[Iterable[float]] = field(default_factory=list)
    val: List[Iterable[float]] = field(default_factory=list)
    test: Iterable[float] = None