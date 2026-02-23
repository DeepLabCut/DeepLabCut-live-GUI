from enum import Enum
from pathlib import Path


# TODO @C-Achard decide if this moves to utils,
# or if we update dlclive.Engine to have these methods and use that instead of a separate enum here.
# The latter would be more cohesive but also creates a dependency from utils to dlclive,
# pending release of dlclive
class Engine(Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"

    @staticmethod
    def is_pytorch_model_path(model_path: str | Path) -> bool:
        path = Path(model_path)
        return path.is_file() and path.suffix.lower() in (".pt", ".pth")

    @staticmethod
    def is_tensorflow_model_dir_path(model_path: str | Path) -> bool:
        path = Path(model_path)
        if not path.is_dir():
            return False
        has_cfg = (path / "pose_cfg.yaml").is_file()
        has_pb = any(p.suffix.lower() == ".pb" for p in path.glob("*.pb"))
        return has_cfg and has_pb

    @classmethod
    def from_model_type(cls, model_type: str) -> "Engine":
        if model_type.lower() == "pytorch":
            return cls.PYTORCH
        elif model_type.lower() in ("tensorflow", "base", "tensorrt", "lite"):
            return cls.TENSORFLOW
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> "Engine":
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if path.is_dir():
            if cls.is_tensorflow_model_dir_path(path):
                return cls.TENSORFLOW
        elif path.is_file():
            if cls.is_pytorch_model_path(path):
                return cls.PYTORCH

        raise ValueError(f"Could not determine engine from model path: {model_path}")
