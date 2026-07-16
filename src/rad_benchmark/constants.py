"""Dataset constants reported by the RAD paper."""

SEMANTIC_CATEGORIES = (
    "binderclip",
    "bowl",
    "box",
    "can",
    "charger",
    "cup1",
    "cup2",
    "gluebottle",
    "phonecase",
    "rubberduck",
    "spoon",
    "spraybottle",
    "tennisball",
)

# Public archives use one directory per physical instance. The paper aggregates
# these 18 instances into the 13 semantic categories above.
INSTANCE_TO_CATEGORY = {
    "binderclip": "binderclip",
    "binderclip2": "binderclip",
    "bowl_upright": "bowl",
    "box": "box",
    "can": "can",
    "charger": "charger",
    "cup1_upright": "cup1",
    "cup2_upright": "cup2",
    "cup2_upright2": "cup2",
    "cup2_upright3": "cup2",
    "gluebottle": "gluebottle",
    "gluebottle2": "gluebottle",
    "phonecase": "phonecase",
    "phonecase2": "phonecase",
    "rubberduck": "rubberduck",
    "spoon_upright": "spoon",
    "spraybottle2": "spraybottle",
    "tennisball": "tennisball",
}

DEFECT_TYPES = ("missing", "stained", "scratched", "squeezed")
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
