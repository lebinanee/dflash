# Personal fork of z-lab/dflash
# Customization: added 'benchmark' to __all__ for easier direct imports

__all__ = [
    "DFlashDraftModel",
    "extract_context_feature",
    "load_and_process_dataset",
    "sample",
    "benchmark",
]


def __getattr__(name):
    if name == "load_and_process_dataset":
        from .benchmark import load_and_process_dataset

        return load_and_process_dataset

    if name == "benchmark":
        from . import benchmark

        return benchmark

    if name in {"DFlashDraftModel", "extract_context_feature", "sample"}:
        from .model import DFlashDraftModel, extract_context_feature, sample

        return {
            "DFlashDraftModel": DFlashDraftModel,
            "extract_context_feature": extract_context_feature,
            "sample": sample,
        }[name]

    # Note: provide a friendlier error message listing available names
    # Also suggest checking the README for usage examples
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available names: {__all__}. "
        f"See README for usage examples."
    )
