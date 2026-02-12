# # labeling/schema.py
# from __future__ import annotations
# from dataclasses import dataclass, asdict
# from typing import Any, Dict, List, Optional

# # ---- Hardcoded fields / defaults (edit as you like) ----

# UNCERTAINTY_CHOICES = ["low", "medium", "high"]

# @dataclass
# class SegmentLabel:
#     # Keep id as string to match your example; can change to int if desired.
#     # id: str
#     # expected_persistence_hours: float = 1000.0
#     # persistence_uncertainty: str = "low"
#     # static_likelihood: float = 1.0
#     # notes: str = ""

#     id: int
#     median_persistence_hours: float
#     p90_persistence_hours: float
#     persistence_conf: float
#     notes: str = ""

#     def to_json_dict(self, key_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
#         """
#         key_map lets you rename keys without changing internal names.
#         Example: {"expected_persistence_hours": "persistence_hours"}
#         """
#         d = asdict(self)
#         if key_map:
#             out = {}
#             for k, v in d.items():
#                 out[key_map.get(k, k)] = v
#             return out
#         return d


# labeling/schema.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict, Optional

UNCERTAINTY_CHOICES = ["low", "medium", "high"]

LABEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # Treat as "default stable object" guess.
    "static": {
        "median_persistence_hours": 1_000_000.0,
        "p90_persistence_hours": 10_000_000.0,
        "persistence_conf": 0.9,
        "notes": "preset: static",
    },
    "months": {
        "median_persistence_hours": 24 * 30,
        "p90_persistence_hours": 24 * 30 * 6,
        "persistence_conf": 0.6,
        "notes": "preset: Weeks-to-months",
    },
    "weeks": {
        "median_persistence_hours": 24 * 7,
        "p90_persistence_hours": 24 * 7 * 4,
        "persistence_conf": 0.5,
        "notes": "preset: Days-to-weeks",
    },
    "days": {
        "median_persistence_hours": 24,
        "p90_persistence_hours": 24 * 3,
        "persistence_conf": 0.5,
        "notes": "preset: Multiple days",
    },
    "next_day": {
        "median_persistence_hours": 8,
        "p90_persistence_hours": 24,
        "persistence_conf": 0.5,
        "notes": "preset: Next day",
    },
    "several_hours": {
        "median_persistence_hours": 4,
        "p90_persistence_hours": 8,
        "persistence_conf": 0.5,
        "notes": "preset: Several hours",
    },
    "couple_hours": {
        "median_persistence_hours": 0.5,
        "p90_persistence_hours": 2,
        "persistence_conf": 0.5,
        "notes": "preset: Couple hours",
    },
    "under_hour": {
        "median_persistence_hours": 0.25,
        "p90_persistence_hours": 1,
        "persistence_conf": 0.5,
        "notes": "preset: Under hour",
    },
    "under_30min": {
        "median_persistence_hours": 0.1,
        "p90_persistence_hours": 0.5,
        "persistence_conf": 0.5,
        "notes": "preset: Under 30min",
    },
    "under_10min": {
        "median_persistence_hours": 0.05,
        "p90_persistence_hours": 0.25,
        "persistence_conf": 0.5,
        "notes": "preset: Under 10min",
    },
    "under_2min": {
        "median_persistence_hours": 0.013,
        "p90_persistence_hours": 0.034,
        "persistence_conf": 0.5,
        "notes": "preset: Under 2min",
    },
    # Treat as "often changes" guess.
    "dynamic": {
        "median_persistence_hours": 0.001,
        "p90_persistence_hours": 0.03,
        "persistence_conf": 0.9,
        "notes": "preset: dynamic",
    },
}

        # self.static_btn = QPushButton("Static (Ctrl+1)")
        # self.months_btn = QPushButton("Weeks-to-Months (Ctrl+2)")
        # self.weeks_btn = QPushButton("Days-to-Weeks (Ctrl+3)")
        # self.days_btn = QPushButton("Multiple Days (Ctrl+4)")
        # self.next_day_btn = QPushButton("Next Day (Ctrl+5)")
        # self.several_hours_btn = QPushButton("Several Hours (Ctrl+6)")
        # self.couple_hours_btn = QPushButton("Couple Hours (Ctrl+7)")
        # self.under_hour_btn = QPushButton("Under Hour (Ctrl+8)")
        # self.under_30min_btn = QPushButton("Under 30min (Ctrl+9)")
        # self.under_10min_btn = QPushButton("Under 10min (Ctrl+0)")
        # self.under_2min_btn = QPushButton("Under 2min (Ctrl+-)")
        # self.dynamic_btn = QPushButton("Dynamic (Ctrl+=)")

        # self.static_btn.clicked.connect(lambda: self._apply_preset("static"))
        # self.months_btn.clicked.connect(lambda: self._apply_preset("months"))
        # self.weeks_btn.clicked.connect(lambda: self._apply_preset("weeks"))
        # self.days_btn.clicked.connect(lambda: self._apply_preset("days"))
        # self.next_day_btn.clicked.connect(lambda: self._apply_preset("next_day"))
        # self.several_hours_btn.clicked.connect(lambda: self._apply_preset("several_hours"))
        # self.couple_hours_btn.clicked.connect(lambda: self._apply_preset("couple_hours"))
        # self.under_hour_btn.clicked.connect(lambda: self._apply_preset("under_hour"))
        # self.under_30min_btn.clicked.connect(lambda: self._apply_preset("under_30min"))
        # self.under_10min_btn.clicked.connect(lambda: self._apply_preset("under_10min"))
        # self.under_2min_btn.clicked.connect(lambda: self._apply_preset("under_2min"))
        # self.dynamic_btn.clicked.connect(lambda: self._apply_preset("dynamic"))

def get_label_presets() -> Dict[str, Dict[str, Any]]:
    """
    Returns reusable label presets keyed by string.
    The GUI uses these to populate form fields.
    Keys should match SegmentLabel field names (internal schema names).
    """
    return LABEL_PRESETS



@dataclass
class SegmentLabel:
    id: int = field(metadata={"ui": {"kind": "readonly"}})

    median_persistence_hours: float = field(metadata={"ui": {"kind": "float", "min": 0.0, "max": 1e9, "step": 10.0, "decimals": 6}})
    p90_persistence_hours: float = field(metadata={"ui": {"kind": "float", "min": 0.0, "max": 1e9, "step": 10.0, "decimals": 6}})
    persistence_conf: float = field(metadata={"ui": {"kind": "float", "min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2}})

    notes: str = field(default="", metadata={"ui": {"kind": "text"}})

    def to_json_dict(self, key_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        d = asdict(self)
        if key_map:
            return {key_map.get(k, k): v for k, v in d.items()}
        return d

    # id: int = field(metadata={"ui": {"kind": "readonly"}})

    # median_persistence_hours: float = field(metadata={"ui": {"kind": "float", "min": 0.0, "max": 1e9, "step": 10.0, "decimals": 2}})
    # p90_persistence_hours: float = field(metadata={"ui": {"kind": "float", "min": 0.0, "max": 1e9, "step": 10.0, "decimals": 2}})
    # persistence_conf: float = field(metadata={"ui": {"kind": "float", "min": 0.0, "max": 1.0, "step": 0.05, "decimals": 3}})

    # notes: str = field(default="", metadata={"ui": {"kind": "text"}})

    # def to_json_dict(self, key_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    #     d = asdict(self)
    #     if key_map:
    #         return {key_map.get(k, k): v for k, v in d.items()}
    #     return d


def schema_fields_for_gui() -> list[dict[str, Any]]:
    """
    Returns a list of UI field descriptors derived from SegmentLabel dataclass metadata.
    The GUI can use this to auto-build widgets and do read/write.
    """
    out: list[dict[str, Any]] = []
    for f in fields(SegmentLabel):
        ui = (f.metadata or {}).get("ui")
        if not ui:
            continue
        out.append({"name": f.name, **ui})
    return out


# def build_default_labels_for_segment_ids(
#     segment_ids: List[int],
#     key_map: Optional[Dict[str, str]] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Create a default label record for each segment id.
#     Returns list of dicts (already key-mapped if provided).
#     """
#     labels = []
#     for sid in segment_ids:
#         rec = SegmentLabel(id=str(sid))
#         labels.append(rec.to_json_dict(key_map=key_map))
#     return labels

def build_default_labels_for_segment_ids(
    segment_ids: List[int],
    key_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    for sid in segment_ids:
        # minimal record: exists, but contains no label data yet
        d: Dict[str, Any] = {"id": int(sid)}
        if key_map:
            d = {key_map.get(k, k): v for k, v in d.items()}
        labels.append(d)
    return labels


