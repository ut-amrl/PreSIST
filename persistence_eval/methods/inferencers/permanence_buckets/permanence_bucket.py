

# Create an enum with string values for the different dynamic buckets
from enum import StrEnum


class PermanenceBucket(StrEnum):
    DYNAMIC = "dynamic"
    # MOVABLE = "movable"
    SEMI_STATIC = "semi_static"
    STATIC = "static"