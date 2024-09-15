# global
from typing import Any, Optional

# local
import ivy


def load(
    f,
    map_location=None,
    pickle_module: Any = None,
    *,
    weights_only: Optional[bool] = None,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any,
) -> Any:
    raise ivy.exceptions.IvyNotImplementedException(
        "The `torch.load` frontend has not yet been implemented."
    )


def save(
    obj: object,
    f,
    pickle_module: Any = None,
    pickle_protocol: int = 2,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    raise ivy.exceptions.IvyNotImplementedException(
        "The `torch.save` frontend has not yet been implemented."
    )
