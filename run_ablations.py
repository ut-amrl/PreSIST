#!/usr/bin/env python3
"""Single-host GPU job pool for ablation training runs.

One python process per GPU; CUDA_VISIBLE_DEVICES isolates each child.
When a GPU frees up, dispatch the next queued config onto it.
Skips configs already marked complete (<log-dir>/<cfg>.done) unless
--force is passed. Refuses to dispatch onto a GPU with >= --busy-gb
GB already in use (per nvidia-smi).
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


def parse_args():
    # Split argv on a bare '--': everything after is forwarded to every
    # train.py invocation. Examples:
    #   run_ablations.py --gpus 0 --configs foo -- +train_dataset.x=0.02
    argv = sys.argv[1:]
    extra_args = []
    if "--" in argv:
        idx = argv.index("--")
        extra_args = argv[idx + 1:]
        argv = argv[:idx]

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", required=True,
                   help="Comma-separated GPU IDs, e.g. '0,1,2,3'")
    p.add_argument("--configs", nargs="+", required=True,
                   help="Hydra --config-name values (no .yaml suffix)")
    p.add_argument("--train-script", default="train.py")
    p.add_argument("--log-dir", default="ablation_logs")
    p.add_argument("--poll-seconds", type=float, default=15.0)
    p.add_argument("--force", action="store_true",
                   help="Re-run configs even if their .done marker exists")
    p.add_argument("--busy-gb", type=float, default=10.0,
                   help="A GPU with >= this many GB used is treated as busy")
    p.add_argument("--on-busy", choices=["skip", "abort", "poll"], default="skip",
                   help="If a configured GPU is busy: drop (skip), exit (abort), "
                        "or wait until free (poll)")
    p.add_argument("--busy-poll-minutes", type=float, default=10.0,
                   help="With --on-busy poll, re-check busy GPUs this often")
    args = p.parse_args(argv)
    args.extra_args = extra_args
    return args


def gpu_used_mb(gpu: str) -> int:
    """memory.used in MiB for the given GPU index, via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}",
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return int(out.splitlines()[0])


def filter_busy_gpus(gpus, busy_gb, on_busy, poll_minutes):
    """Return GPUs usable per the busy-check policy.

    on_busy='skip'  -> drop busy GPUs immediately.
    on_busy='abort' -> exit if any are busy.
    on_busy='poll'  -> re-check busy GPUs every poll_minutes until free.
    """
    pending = list(gpus)
    available = []
    while pending:
        still_busy = []
        for g in pending:
            try:
                used_mb = gpu_used_mb(g)
            except Exception as e:
                print(f"[preflight] gpu={g} nvidia-smi failed: {e}; skipping",
                      file=sys.stderr)
                continue
            if used_mb >= busy_gb * 1024:
                msg = (f"[preflight] gpu={g} busy "
                       f"({used_mb} MiB used, threshold {busy_gb} GB)")
                if on_busy == "abort":
                    print(msg + " - aborting", file=sys.stderr)
                    sys.exit(2)
                if on_busy == "skip":
                    print(msg + " - skipping")
                else:  # poll
                    print(msg + f" - will retry in {poll_minutes} min")
                    still_busy.append(g)
            else:
                print(f"[preflight] gpu={g} OK ({used_mb} MiB used)")
                available.append(g)
        if not still_busy:
            break
        time.sleep(poll_minutes * 60)
        pending = still_busy
    return available


def done_marker(log_dir, cfg):
    return os.path.join(log_dir, f"{cfg}.done")


def launch(gpu, cfg, args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{cfg}.gpu{gpu}.{ts}.log")
    lf = open(log_path, "w", buffering=1)
    cmd = [sys.executable, args.train_script,
           f"--config-name={cfg}", *args.extra_args]
    print(f"[launch] gpu={gpu} cfg={cfg} log={log_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc, lf, log_path


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    raw_gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    gpus = filter_busy_gpus(raw_gpus, args.busy_gb, args.on_busy,
                            args.busy_poll_minutes)
    if not gpus:
        print("[preflight] no usable GPUs after busy-check; exiting",
              file=sys.stderr)
        sys.exit(2)

    queue = []
    for cfg in args.configs:
        marker = done_marker(args.log_dir, cfg)
        if not args.force and os.path.exists(marker):
            print(f"[skip] cfg={cfg} already complete (marker: {marker})")
            continue
        queue.append(cfg)
    if not queue:
        print("[skip] all configs already complete; pass --force to re-run")
        sys.exit(0)

    running = {}     # gpu -> (Popen, cfg, log_file, log_path)
    results = []     # (cfg, gpu, rc, log_path)

    for gpu in gpus:
        if not queue:
            break
        cfg = queue.pop(0)
        proc, lf, path = launch(gpu, cfg, args)
        running[gpu] = (proc, cfg, lf, path)

    try:
        while running:
            time.sleep(args.poll_seconds)
            for gpu in list(running):
                proc, cfg, lf, path = running[gpu]
                rc = proc.poll()
                if rc is None:
                    continue
                lf.close()
                status = "ok" if rc == 0 else f"FAIL rc={rc}"
                print(f"[done] gpu={gpu} cfg={cfg} status={status}")
                results.append((cfg, gpu, rc, path))
                if rc == 0:
                    with open(done_marker(args.log_dir, cfg), "w") as f:
                        f.write(f"{datetime.now().isoformat()} "
                                f"gpu={gpu} log={path}\n")
                if queue:
                    next_cfg = queue.pop(0)
                    np_proc, np_lf, np_path = launch(gpu, next_cfg, args)
                    running[gpu] = (np_proc, next_cfg, np_lf, np_path)
                else:
                    del running[gpu]
    except KeyboardInterrupt:
        print("[interrupt] terminating children...")
        for gpu, (proc, cfg, lf, _) in running.items():
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            lf.close()
        sys.exit(130)

    print("\n[summary]")
    for cfg, gpu, rc, path in results:
        tag = "OK  " if rc == 0 else "FAIL"
        print(f"  [{tag}] gpu={gpu}  rc={rc}  cfg={cfg}  log={path}")
    sys.exit(0 if all(rc == 0 for _, _, rc, _ in results) else 1)


if __name__ == "__main__":
    main()
