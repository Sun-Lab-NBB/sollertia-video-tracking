"""Provides the ``slvt create-training-dataset`` command that creates a DeepLabCut shuffle at parity with the GUI."""

from pathlib import Path

import click

from ..training import (
    create_training_dataset,
    get_available_augmenters,
    get_available_pose_models,
    get_available_super_animals,
    build_superanimal_weight_init,
    get_available_object_detectors,
    build_conditional_top_down_conditions,
)

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""

_IMAGENET_WEIGHT_INIT: str = "imagenet"
"""The default weight-initialization choice that uses ImageNet transfer learning rather than a SuperAnimal model."""


@click.command("create-training-dataset", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-s", "--shuffle", default=1, show_default=True, help="The shuffle index to create.")
@click.option(
    "-n",
    "--network-type",
    type=click.Choice(get_available_pose_models()),
    default=None,
    help="The pose-model architecture. Omit to use the project default. A top-down architecture also trains an "
    "object detector. Required when a SuperAnimal --weight-initialization is selected.",
)
@click.option(
    "-d",
    "--detector-type",
    type=click.Choice(get_available_object_detectors()),
    default=None,
    help="The object detector for top-down models. Omit to use the default detector.",
)
@click.option(
    "-a",
    "--augmenter-type",
    type=click.Choice(get_available_augmenters()),
    default=None,
    help="The data-augmentation pipeline. Omit to use the default.",
)
@click.option(
    "-wi",
    "--weight-initialization",
    type=click.Choice([_IMAGENET_WEIGHT_INIT, "transfer", "fine-tune"]),
    default=_IMAGENET_WEIGHT_INIT,
    show_default=True,
    help="How the model's starting weights are set: ImageNet transfer learning, SuperAnimal transfer learning, or "
    "SuperAnimal fine-tuning.",
)
@click.option(
    "-sa",
    "--super-animal",
    type=click.Choice(get_available_super_animals()),
    default=None,
    help="The SuperAnimal dataset to initialize from. Required when --weight-initialization is 'transfer' or "
    "'fine-tune'.",
)
@click.option(
    "-mr",
    "--memory-replay",
    is_flag=True,
    help="Enable SuperAnimal memory replay (only valid with --weight-initialization 'fine-tune'). Note: 'slvt "
    "train' cannot train memory-replay shuffles.",
)
@click.option(
    "-cc",
    "--ctd-conditions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The conditioning file for conditional top-down models: a predictions file (.h5 or .json) or a model "
    "snapshot (.pt).",
)
@click.option(
    "-fs",
    "--from-shuffle",
    type=int,
    default=None,
    help="An existing shuffle whose train/test split to reuse instead of drawing a fresh one.",
)
@click.option(
    "-ftsi",
    "--from-training-set-index",
    default=0,
    show_default=True,
    help="The training-set fraction index of --from-shuffle when reusing its split.",
)
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    help="Overwrite the shuffle if its index already exists. WARNING: this replaces the existing shuffle's "
    "training-dataset files.",
)
def create_training_dataset_command(
    config: Path,
    shuffle: int,
    network_type: str | None,
    detector_type: str | None,
    augmenter_type: str | None,
    weight_initialization: str,
    super_animal: str | None,
    ctd_conditions: Path | None,
    from_shuffle: int | None,
    from_training_set_index: int,
    *,
    memory_replay: bool,
    overwrite: bool,
) -> None:
    """Creates a training-dataset shuffle for a project, selecting the model, weights, augmentation, and split.

    CONFIG is the path to the DeepLabCut project's config.yaml. The shuffle bakes in the model architecture, weight
    initialization, and a train/test split; training is run afterward with ``slvt train``. Multi-animal projects are
    handled automatically. This mirrors the DeepLabCut GUI's create-training-dataset tab for headless and scripted
    use.
    """
    uses_super_animal = weight_initialization != _IMAGENET_WEIGHT_INIT
    if uses_super_animal and super_animal is None:
        message = (
            "Unable to create the training dataset. Provide --super-animal when --weight-initialization is "
            "'transfer' or 'fine-tune'."
        )
        raise click.ClickException(message=message)
    if not uses_super_animal and super_animal is not None:
        message = (
            "Unable to create the training dataset. --super-animal is only valid when --weight-initialization is "
            "'transfer' or 'fine-tune'."
        )
        raise click.ClickException(message=message)
    if uses_super_animal and network_type is None:
        message = (
            "Unable to create the training dataset. Provide --network-type when a SuperAnimal "
            "--weight-initialization is selected."
        )
        raise click.ClickException(message=message)
    if memory_replay and weight_initialization != "fine-tune":
        message = (
            "Unable to create the training dataset. --memory-replay is only valid when --weight-initialization is "
            "'fine-tune'."
        )
        raise click.ClickException(message=message)

    try:
        weights = None
        if uses_super_animal and super_animal is not None and network_type is not None:
            weights = build_superanimal_weight_init(
                config=config,
                super_animal=super_animal,
                network_type=network_type,
                detector_type=detector_type,
                fine_tune=weight_initialization == "fine-tune",
                memory_replay=memory_replay,
            )
        conditions = None
        if ctd_conditions is not None:
            conditions = build_conditional_top_down_conditions(conditions_path=ctd_conditions)
        summary = create_training_dataset(
            config=config,
            shuffle=shuffle,
            network_type=network_type,
            detector_type=detector_type,
            augmenter_type=augmenter_type,
            weight_initialization=weights,
            conditional_top_down_conditions=conditions,
            from_shuffle=from_shuffle,
            from_training_set_index=from_training_set_index,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError, NotImplementedError) as error:
        raise click.ClickException(message=str(error)) from error

    click.echo(message=summary.describe())
