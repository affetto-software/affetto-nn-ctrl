from __future__ import annotations

from enum import Enum, auto
from typing import Any, Final, Literal, TypeAlias, TypeVar

Unknown: TypeAlias = Any
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class _NoDefault(Enum):
    """Enum to represent the absence of a default value in method parameters."""

    no_default = auto()


no_default: Final = _NoDefault.no_default
NoDefault: TypeAlias = Literal[_NoDefault.no_default]

# Local Variables:
# jinx-local-words: "Enum"
# End:
