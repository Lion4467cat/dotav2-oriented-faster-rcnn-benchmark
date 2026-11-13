"""Canonical DOTA v2.0 18-class metadata and palette for MMRotate dev-1.x."""

DOTA_V2_CLASSES = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
    "container-crane",
    "airport",
    "helipad",
)

DOTA_V2_PALETTE = [
    (165, 42, 42),
    (189, 183, 107),
    (0, 255, 0),
    (255, 0, 0),
    (138, 43, 226),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 193, 193),
    (0, 51, 153),
    (255, 250, 205),
    (0, 139, 139),
    (255, 255, 0),
    (147, 116, 116),
    (0, 0, 255),
    (100, 149, 237),
    (34, 139, 34),
    (255, 20, 147),
]

# Mapping for label normalization if source annotations use spaces instead of hyphens.
DOTA_V2_SPACE_TO_HYPHEN = {cls.replace("-", " "): cls for cls in DOTA_V2_CLASSES}
