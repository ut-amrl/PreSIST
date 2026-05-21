import json, collections
from pathlib import Path

SIDECAR = Path("/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30/test_from_2012-10-30.track_id_to_location_cell128.json")
print(f"sidecar exists: {SIDECAR.exists()}")
sidecar = json.load(SIDECAR.open())
total = len(sidecar)
sidecar_locs = collections.Counter(sidecar.values())
print(f"total test track_ids: {total}")
print(f"distinct location_keys in sidecar: {len(sidecar_locs)}")
print()
print("=== bucket-fallback for mixture_exp:location ===")
print(f"{'D':>4s} {'real_buckets':>13s} {'sidecar_locs':>13s} {'tracks_hit':>11s} {'tracks_fb':>10s} {'hit %':>8s}")
for D in [1, 2, 5, 7, 14, 28, 35]:
    P = Path(f"/workspace/data/output/perpetua/perpetua_mixture_exp_location_pucpr_split_50_20_30_daycount_fullK_mins5_p0p01/days{D}_seed0/persistence_params.json")
    if not P.exists():
        print(f"{D:>4d}  (no params)"); continue
    params = json.load(P.open())
    real_buckets = set(k for k in params.keys() if k != "__global__")
    hits = sum(1 for loc in sidecar.values() if loc in real_buckets)
    fb = total - hits
    print(f"{D:>4d} {len(real_buckets):>13d} {len(sidecar_locs):>13d} {hits:>11d} {fb:>10d} {100.0*hits/total:>7.1f}%")
print()
print("=== same for single_exp:location ===")
for D in [1, 2, 5, 7, 14, 28, 35]:
    P = Path(f"/workspace/data/output/perpetua/perpetua_single_exp_location_pucpr_split_50_20_30_daycount_emk1_noise0p01_mins5/days{D}_seed0/persistence_params.json")
    if not P.exists():
        print(f"{D:>4d}  (no params)"); continue
    params = json.load(P.open())
    real_buckets = set(k for k in params.keys() if k != "__global__")
    hits = sum(1 for loc in sidecar.values() if loc in real_buckets)
    fb = total - hits
    print(f"{D:>4d} {len(real_buckets):>13d} {len(sidecar_locs):>13d} {hits:>11d} {fb:>10d} {100.0*hits/total:>7.1f}%")
