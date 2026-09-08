from enum import Enum


class Backend(Enum):
    FROM_SCRATCH = "From Scratch"
    PYTORCH = "PyTorch"


def get_backend(backend_name):
    if backend_name == Backend.FROM_SCRATCH.value:
        return "scratch"

    if backend_name == Backend.PYTORCH.value:
        return "pytorch"

    raise ValueError(
        f"Unsupported backend: {backend_name}"
    )