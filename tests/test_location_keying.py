"""Tests for location_key_for_track."""

from persistence_eval.methods.training.location_keying import location_key_for_track


def test_persistent_uses_space_id():
    track = {"track_id": "t1", "space_id": "persistent_-89", "observations": []}
    assert location_key_for_track(track, cell_size_px=64) == "persistent_-89"


def test_transient_with_bbox_buckets_into_cell():
    track = {
        "track_id": "t2",
        "observations": [
            {"state": "Negative", "bbox": [0, 0, 10, 10], "timestamp": 1},
            # First positive bbox is centered at (160, 160) → cell (2, 2) at size 64.
            {"state": "Positive", "bbox": [128, 128, 192, 192], "timestamp": 2},
        ],
    }
    assert location_key_for_track(track, cell_size_px=64) == "cell_2_2_sz64"


def test_transient_changing_cell_size():
    track = {
        "track_id": "t3",
        "observations": [{"state": "Positive", "bbox": [0, 0, 32, 32], "timestamp": 1}],
    }
    # Center (16, 16) → at size 32 → cell (0, 0); at size 16 → cell (1, 1).
    assert location_key_for_track(track, cell_size_px=32) == "cell_0_0_sz32"
    assert location_key_for_track(track, cell_size_px=16) == "cell_1_1_sz16"


def test_empty_space_id_falls_through_to_cells():
    track = {
        "track_id": "t4",
        "space_id": "",
        "observations": [{"state": "Positive", "bbox": [0, 0, 64, 64], "timestamp": 1}],
    }
    assert location_key_for_track(track, cell_size_px=64).startswith("cell_")


def test_unlocated_when_no_positive_bbox():
    track = {
        "track_id": "t5",
        "observations": [{"state": "Negative", "bbox": [0, 0, 10, 10], "timestamp": 1}],
    }
    assert location_key_for_track(track, cell_size_px=64) == "transient_unlocated_t5"
