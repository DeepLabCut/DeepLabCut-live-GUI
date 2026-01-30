# dlclivegui/cameras/adapters.py
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from dlclivegui.config import CameraSettingsModel

from dlclivegui.config import CameraSettings
from dlclivegui.utils.config_models import CameraSettingsModel

CameraSettingsLike = Union[CameraSettings, "CameraSettingsModel", dict[str, Any]]


def ensure_dc_camera(settings: CameraSettingsLike) -> CameraSettings:
    """
    Normalize any supported camera settings payload to the legacy dataclass CameraSettings.
    - If already a dataclass: deep-copy and return.
    - If it's a Pydantic CameraSettingsModel: convert via model_dump().
    - If it's a dict: unpack into CameraSettings.
    Ensures default application and type coercions via dataclass.apply_defaults().
    """
    # Case 1: Already the dataclass
    if isinstance(settings, CameraSettings):
        dc = copy.deepcopy(settings)
        return dc.apply_defaults()

    # Case 2: Pydantic model (if available in this environment)
    if CameraSettingsModel is not None and isinstance(settings, CameraSettingsModel):
        data = settings.model_dump()
        dc = CameraSettings(**data)
        return dc.apply_defaults()

    # Case 3: Plain dict (best-effort flexibility)
    if isinstance(settings, dict):
        dc = CameraSettings(**settings)
        return dc.apply_defaults()

    raise TypeError(
        "Unsupported camera settings type. Expected CameraSettings dataclass, CameraSettingsModel (Pydantic), or dict."
    )
