from pathlib import Path

from ..training import (
    WeightInitializationMethod as WeightInitializationMethod,
    create_training_dataset as create_training_dataset,
    get_available_pose_models as get_available_pose_models,
    get_available_super_animals as get_available_super_animals,
    build_superanimal_weight_init as build_superanimal_weight_init,
    get_available_object_detectors as get_available_object_detectors,
    build_conditional_top_down_conditions as build_conditional_top_down_conditions,
)

_CONTEXT_SETTINGS: dict[str, int]
_CONDITIONAL_TOP_DOWN_PREFIX: str

def prepare_command(
    config_path: Path,
    shuffle: int,
    network: str | None,
    detector: str | None,
    weight_initialization: str,
    super_animal: str | None,
    conditional_top_down_conditions: Path | None,
    from_shuffle: int | None,
    *,
    memory_replay: bool,
    overwrite: bool,
) -> None: ...
