from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseOPENIEModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, text: str) -> dict[str, Any]:
        raise NotImplementedError
