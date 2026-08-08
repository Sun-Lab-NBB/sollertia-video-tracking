from .gui import gui_command as gui_command
from .cuda import cuda_command as cuda_command
from .infer import infer_command as infer_command
from .train import train_command as train_command
from .extract import extract_group as extract_group
from .prepare import prepare_command as prepare_command

_CONTEXT_SETTINGS: dict[str, int]

def slvt_cli() -> None: ...
