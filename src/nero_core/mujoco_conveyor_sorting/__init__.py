"""MuJoCo continuous conveyor sorting research environment."""

__all__ = ["ConveyorSortingEnv"]
__version__ = "0.1.0"


def __getattr__(name):
    # Dataset validation/export should not require MuJoCo or an OpenGL runtime.
    if name == "ConveyorSortingEnv":
        from .envs.conveyor_sorting_env import ConveyorSortingEnv
        globals()[name] = ConveyorSortingEnv
        return ConveyorSortingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
