"""POET pose-estimation backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dlclivegui.services.dlc_processor import (
    PoseBackends,
    PosePacket,
    PoseSource,
)
from dlclivegui.services.inference.base import PoseBackend

from .skeleton import POET_KEYPOINT_NAMES, POET_SKELETON_EDGES

try:
    from poet_live import POET, PostProcess
    from poet_live.models.backbone import Backbone, Joiner
    from poet_live.models.position_encoding import PositionEmbeddingSine
    from poet_live.models.transformer import Transformer
except ImportError as exc:
    raise ImportError("POET imports failed. Ensure the POET package and its dependencies are installed.") from exc


logger = logging.getLogger(__name__)


class POETBackend(PoseBackend):
    """POET pose-inference backend using COCO-17 keypoints."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "auto",
        threshold: float = 0.7,
        use_amp: bool = True,
    ) -> None:
        checkpoint = Path(checkpoint_path).expanduser()

        if not checkpoint.is_file():
            raise FileNotFoundError(f"POET checkpoint not found: {checkpoint}")

        if checkpoint.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("POET checkpoint must use a .pt or .pth extension.")

        self._checkpoint_path = checkpoint
        self._requested_device = device
        self._threshold = float(threshold)
        self._use_amp = bool(use_amp)

        self._model: Any | None = None
        self._postprocessor: Any | None = None
        self._device: torch.device | None = None

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @staticmethod
    def _resolve_torch_device(
        device: str | None,
    ) -> torch.device:
        requested = device.strip().lower() if device else "auto"

        if requested in {"auto", "best"}:
            if torch.cuda.is_available():
                return torch.device("cuda")

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")

            return torch.device("cpu")

        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "Requested device %r but CUDA is unavailable; using CPU.",
                device,
            )
            return torch.device("cpu")

        return torch.device(requested)

    def init_inference(
        self,
        init_frame: np.ndarray,
    ) -> None:
        (
            self._model,
            self._postprocessor,
            self._device,
        ) = self._build_model()

        # Warm up without emitting a result. PoseProcessor emits the
        # initial frame's result after init_inference() returns.
        self.get_pose(init_frame)

    def _build_model(
        self,
    ) -> tuple[Any, Any, torch.device]:
        device = self._resolve_torch_device(self._requested_device)
        hidden_dimension = 256

        backbone = Backbone(
            "resnet50",
            train_backbone=False,
            return_interm_layers=False,
            dilation5=False,
            dilation4=False,
        )
        position_encoding = PositionEmbeddingSine(
            hidden_dimension // 2,
            normalize=True,
        )
        joined_backbone = Joiner(
            backbone,
            position_encoding,
        )
        joined_backbone.num_channels = backbone.num_channels

        transformer = Transformer(
            d_model=hidden_dimension,
            return_intermediate_dec=True,
        )
        model = POET(
            joined_backbone,
            transformer,
            num_classes=2,
            num_queries=25,
            aux_loss=False,
        )

        checkpoint = torch.load(
            self._checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        model.to(device).eval()
        postprocessor = PostProcess().to(device)

        self._mean = self._mean.to(device)
        self._std = self._std.to(device)

        return model, postprocessor, device

    @torch.no_grad()
    def get_pose(
        self,
        frame: np.ndarray,
        frame_time: float | None = None,
    ) -> np.ndarray | None:
        del frame_time

        model = self._model
        postprocessor = self._postprocessor
        device = self._device

        if model is None or postprocessor is None or device is None:
            raise RuntimeError("POET backend is not initialized.")

        rgb = frame[..., ::-1].copy()
        height, width = rgb.shape[:2]

        image = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        image = (image - self._mean) / self._std

        use_amp = self._use_amp and device.type == "cuda"

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(image)

        target_sizes = torch.tensor(
            [[height, width]],
            device=device,
        )
        result = postprocessor(
            outputs,
            target_sizes=target_sizes,
        )[0]

        scores = result["scores"]
        keypoints = result["keypoints"]

        keep = scores >= self._threshold
        if keep.sum().item() == 0:
            return None

        scores = scores[keep]
        keypoints = keypoints[keep].reshape(
            -1,
            len(POET_KEYPOINT_NAMES),
            3,
        )

        keypoints = keypoints.clone()
        keypoints[:, :, 2] = scores[:, None].clamp(0, 1)

        ordering = torch.argsort(
            scores,
            descending=True,
        )
        keypoints = keypoints[ordering]

        return keypoints.detach().cpu().numpy().astype(np.float32)

    def make_pose_packet(
        self,
        pose: np.ndarray | None,
    ) -> PosePacket:
        individual_count = pose.shape[0] if pose is not None and pose.ndim == 3 else 0

        return PosePacket(
            schema_version=1,
            keypoints=pose,
            keypoint_names=list(POET_KEYPOINT_NAMES),
            individual_ids=[f"person_{index}" for index in range(individual_count)],
            skeleton_id="poet.coco17",
            skeleton_edges=POET_SKELETON_EDGES,
            source=PoseSource(
                backend=PoseBackends.POET,
                model_type=None,
            ),
            raw=pose,
        )

    def close(self) -> None:
        self._model = None
        self._postprocessor = None
        self._device = None
