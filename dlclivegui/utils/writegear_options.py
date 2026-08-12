from collections.abc import Mapping
from typing import TypeAlias

WriteGearOptionValue: TypeAlias = str | int | float | bool | None
WriteGearOptions: TypeAlias = dict[
    str,
    WriteGearOptionValue,
]
WriteGearOptionOverrides: TypeAlias = Mapping[
    str,
    WriteGearOptionValue,
]


def normalize_writegear_options(
    options: WriteGearOptionOverrides,
) -> WriteGearOptions:
    """Normalize known options while retaining supported extensions."""
    normalized = dict(options)

    try:
        normalized["-input_framerate"] = float(normalized["-input_framerate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("WriteGear option '-input_framerate' must be numeric.") from exc

    if normalized["-input_framerate"] <= 0:
        raise ValueError("WriteGear option '-input_framerate' must be positive.")

    try:
        normalized["-crf"] = int(normalized["-crf"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("WriteGear option '-crf' must be an integer.") from exc

    if not 0 <= normalized["-crf"] <= 51:
        raise ValueError("WriteGear option '-crf' must be between 0 and 51.")

    codec = str(normalized.get("-vcodec") or "").strip()
    if not codec:
        raise ValueError("WriteGear option '-vcodec' must be a non-empty string.")

    normalized["-vcodec"] = codec
    return normalized
