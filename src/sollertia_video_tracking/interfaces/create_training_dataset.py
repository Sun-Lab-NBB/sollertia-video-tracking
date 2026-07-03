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
    "--net-type",
    "net_type",
    type=click.Choice(get_available_pose_models()),
    default=None,
    help="The pose-model architecture. Omit to use the project default. A 'top_down_' prefix creates a top-down "
    "model that also trains a detector. Required when a SuperAnimal --weight-init is selected.",
)
@click.option(
    "-d",
    "--detector-type",
    "detector_type",
    type=click.Choice(get_available_object_detectors()),
    default=None,
    help="The object detector for top-down models. Omit to use the DeepLabCut default detector.",
)
@click.option(
    "-a",
    "--augmenter-type",
    "augmenter_type",
    type=click.Choice(get_available_augmenters()),
    default=None,
    help="The data-augmentation pipeline. Omit to use the engine default.",
)
@click.option(
    "--weight-init",
    "weight_init",
    type=click.Choice([_IMAGENET_WEIGHT_INIT, "transfer", "fine-tune"]),
    default=_IMAGENET_WEIGHT_INIT,
    show_default=True,
    help="The weight initialization: ImageNet transfer learning, SuperAnimal transfer learning (new head), or "
    "SuperAnimal fine-tuning (reuse the decoder head, needs a conversion table).",
)
@click.option(
    "--super-animal",
    "super_animal",
    type=click.Choice(get_available_super_animals()),
    default=None,
    help="The SuperAnimal dataset to initialize from. Required when --weight-init is 'transfer' or 'fine-tune'.",
)
@click.option(
    "--memory-replay",
    "memory_replay",
    is_flag=True,
    help="Enables SuperAnimal memory replay (only valid with --weight-init 'fine-tune'). Note: 'slvt train' cannot "
    "train memory-replay shuffles; use deeplabcut.train_network for those.",
)
@click.option(
    "--ctd-conditions",
    "ctd_conditions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The conditioning file for conditional top-down (ctd_*) models: a .h5/.json predictions file, or a .pt "
    "snapshot whose path contains a 'shuffleN' segment (the shuffle index is parsed from it).",
)
@click.option(
    "--from-shuffle",
    "from_shuffle",
    type=int,
    default=None,
    help="An existing shuffle whose train/test split to reuse instead of drawing a fresh one.",
)
@click.option(
    "--from-training-set-index",
    "from_training_set_index",
    default=0,
    show_default=True,
    help="The training-set fraction index of --from-shuffle when reusing its split.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Determines whether to overwrite the shuffle if its index already exists. WARNING: this replaces the "
    "existing shuffle's training-dataset files.",
)
def create_training_dataset_command(
    config: Path,
    shuffle: int,
    net_type: str | None,
    detector_type: str | None,
    augmenter_type: str | None,
    weight_init: str,
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
    initialization, and a train/test split; training is run afterwards with ``slvt train``. Multi-animal projects are
    handled automatically. This mirrors the DeepLabCut GUI's create-training-dataset tab for headless and scripted
    use.
    """
    uses_super_animal = weight_init != _IMAGENET_WEIGHT_INIT
    if uses_super_animal and super_animal is None:
        message = (
            "Unable to create the training dataset. Provide --super-animal when --weight-init is 'transfer' or "
            "'fine-tune'."
        )
        raise click.ClickException(message=message)
    if not uses_super_animal and super_animal is not None:
        message = (
            "Unable to create the training dataset. --super-animal is only valid when --weight-init is 'transfer' "
            "or 'fine-tune'."
        )
        raise click.ClickException(message=message)
    if uses_super_animal and net_type is None:
        message = (
            "Unable to create the training dataset. Provide --net-type when a SuperAnimal --weight-init is selected."
        )
        raise click.ClickException(message=message)
    if memory_replay and weight_init != "fine-tune":
        message = (
            "Unable to create the training dataset. --memory-replay is only valid when --weight-init is 'fine-tune'."
        )
        raise click.ClickException(message=message)

    try:
        weights = None
        if uses_super_animal and super_animal is not None and net_type is not None:
            weights = build_superanimal_weight_init(
                config=config,
                super_animal=super_animal,
                net_type=net_type,
                detector_type=detector_type,
                fine_tune=weight_init == "fine-tune",
                memory_replay=memory_replay,
            )
        conditions = None
        if ctd_conditions is not None:
            conditions = build_conditional_top_down_conditions(conditions_path=ctd_conditions)
        summary = create_training_dataset(
            config=config,
            shuffle=shuffle,
            net_type=net_type,
            detector_type=detector_type,
            augmenter_type=augmenter_type,
            weight_init=weights,
            ctd_conditions=conditions,
            from_shuffle=from_shuffle,
            from_training_set_index=from_training_set_index,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError, NotImplementedError) as error:
        raise click.ClickException(message=str(error)) from error

    click.echo(message=summary.describe())
