from pathlib import Path

SUPPORTED_MODELS = [".pt", ".pth", ".pb"]


def is_model_file(file_path: Path | str) -> bool:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not file_path.is_file():
        return False
    return file_path.suffix.lower() in SUPPORTED_MODELS
