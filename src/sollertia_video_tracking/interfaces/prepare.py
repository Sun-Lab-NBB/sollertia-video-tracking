"""Provides the ``slvt prepare`` command that creates a DeepLabCut shuffle for the subsequent model training."""

from pathlib import Path

import click

from ..training import (
    WeightInitializationMethod,
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


@click.command("prepare", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-cfg",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The path to the DeepLabCut project's config.yaml to create the training shuffle for.",
)
@click.option("-s", "--shuffle", default=1, show_default=True, help="The shuffle index to create.")
@click.option(
    "-n",
    "--network",
    type=click.Choice(get_available_pose_models()),
    default=None,
    help="The pose-model architecture. Omit to use the project default. A top-down architecture also trains an "
    "object detector. Required when a SuperAnimal --weight-initialization is selected.",
)
@click.option(
    "-d",
    "--detector",
    type=click.Choice(get_available_object_detectors()),
    default=None,
    help="The object detector for top-down models. Omit to use the default detector.",
)
@click.option(
    "-a",
    "--augmenter",
    type=click.Choice(get_available_augmenters()),
    default=None,
    help="The data-augmentation pipeline. Omit to use the default.",
)
@click.option(
    "-wi",
    "--weight-initialization",
    type=click.Choice([method.value for method in WeightInitializationMethod]),
    default=WeightInitializationMethod.IMAGENET.value,
    show_default=True,
    help="How the model's starting weights are set. 'imagenet' uses ImageNet transfer learning and needs nothing "
    "further. 'transfer' and 'fine-tune' are for SuperAnimal models only: both require a --super-animal dataset and "
    "a --network, doing SuperAnimal transfer learning (a fresh head) or fine-tuning (the SuperAnimal head).",
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
    help="Determines whether to enable SuperAnimal memory replay (only valid with --weight-initialization "
    "'fine-tune'). Note: 'slvt train' cannot train memory-replay shuffles.",
)
@click.option(
    "-ctdc",
    "--conditional-top-down-conditions",
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
    "-o",
    "--overwrite",
    is_flag=True,
    help="Determines whether to overwrite the shuffle if its index already exists. WARNING: this replaces the existing "
    "shuffle's training-dataset files.",
)
def prepare_command(
    config_path: Path,
    shuffle: int,
    network: str | None,
    detector: str | None,
    augmenter: str | None,
    weight_initialization: str,
    super_animal: str | None,
    conditional_top_down_conditions: Path | None,
    from_shuffle: int | None,
    *,
    memory_replay: bool,
    overwrite: bool,
) -> None:
    """Creates a training-dataset shuffle for a project, selecting the model, weights, augmentation, and split.

    ``--config-path`` names the DeepLabCut project's config.yaml. The shuffle bakes in the model architecture, weight
    initialization, and a train/test split. Training is run afterward with ``slvt train``. Multi-animal projects are
    handled automatically. This mirrors the DeepLabCut GUI's create-training-dataset tab for headless and scripted
    use.
    """
    method = WeightInitializationMethod(weight_initialization)
    uses_super_animal = method != WeightInitializationMethod.IMAGENET
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
    if uses_super_animal and network is None:
        message = (
            "Unable to create the training dataset. Provide --network when a SuperAnimal "
            "--weight-initialization is selected."
        )
        raise click.ClickException(message=message)
    if memory_replay and method != WeightInitializationMethod.FINE_TUNE:
        message = (
            "Unable to create the training dataset. --memory-replay is only valid when --weight-initialization is "
            "'fine-tune'."
        )
        raise click.ClickException(message=message)

    try:
        weights = None
        if uses_super_animal and super_animal is not None and network is not None:
            weights = build_superanimal_weight_init(
                config=config_path,
                super_animal=super_animal,
                network_type=network,
                detector_type=detector,
                fine_tune=method == WeightInitializationMethod.FINE_TUNE,
                memory_replay=memory_replay,
            )
        conditions = None
        if conditional_top_down_conditions is not None:
            conditions = build_conditional_top_down_conditions(conditions_path=conditional_top_down_conditions)
        summary = create_training_dataset(
            config=config_path,
            shuffle=shuffle,
            network_type=network,
            detector_type=detector,
            augmenter_type=augmenter,
            weight_initialization=weights,
            conditional_top_down_conditions=conditions,
            from_shuffle=from_shuffle,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError, NotImplementedError) as error:
        raise click.ClickException(message=str(error)) from error

    click.echo(message=summary.describe())
