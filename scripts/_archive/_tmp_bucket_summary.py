import json, collections
from pathlib import Path
SIDECAR = Path("/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30/test_from_2012-10-30.track_id_to_location_cell128.json")
sidecar = json.load(SIDECAR.open())
sidecar_locs = collections.Counter(sidecar.values())
total = len(sidecar)

for trainer in ("single_lognormal", "single_exp", "mixture_lognormal", "mixture_exp"):
    print(f"\n=== {trainer}:location (min_em_samples / min_mixture_samples = 5) ===")
    print(f"{'D':>4s} {'n_buckets_real':>15s} {'n_fallback':>11s} {'small_bks':>10s} {'tracks_hit':>11s} {'tracks_fb':>10s}  trained_buckets_size_dist")
    for D in [1, 2, 3, 5, 7, 10, 14, 21, 28, 35]:
        suffix = "_daycount_emk1_noise0p01_mins5" if trainer.startswith("single") else "_daycount_fullK_mins5_p0p01"
        P = Path(f"/workspace/data/output/perpetua/perpetua_{trainer}_location_pucpr_split_50_20_30{suffix}/days{D}_seed0/persistence_params.json")
        if not P.exists():
            print(f"{D:>4d}  (no params)")
            continue
        params = json.load(P.open())
        glob_p = params.get("__global__", {}).get("params")
        real = {k: v for k, v in params.items() if k != "__global__"}
        fallback_keys = [k for k, v in real.items() if v.get("metadata", {}).get("fallback") or v.get("params") == glob_p]
        trained_keys = [k for k in real if k not in fallback_keys]
        n_samples_per_bucket = sorted(
            [real[k]["metadata"].get("n_samples", 0) for k in trained_keys]
        )
        # Histogram bins
        smin = min(n_samples_per_bucket) if n_samples_per_bucket else 0
        smax = max(n_samples_per_bucket) if n_samples_per_bucket else 0
        smed = n_samples_per_bucket[len(n_samples_per_bucket)//2] if n_samples_per_bucket else 0
        hits = sum(1 for loc in sidecar.values() if loc in trained_keys)
        fb = total - hits
        print(f"{D:>4d}  {len(trained_keys):>15d} {len(fallback_keys):>11d}            {hits:>11d} {fb:>10d}  min={smin} median={smed} max={smax} (n={len(trained_keys)})")
