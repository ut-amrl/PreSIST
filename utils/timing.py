# timing.py
import time
import atexit
import signal
import threading
from collections import defaultdict
from typing import Optional

_UNSET = object()


class _Stats:
    __slots__ = (
        "count", "total", "min", "max",
        "item_count", "per_item_min", "per_item_max",
        "batch_size_min", "batch_size_max",
    )

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min = 0.0
        self.max = 0.0
        # Per-item / batch-size tracking. For non-batched callers
        # (batch_count=1), item_count == count, per_item_* == min/max, and
        # batch_size_* == 1 — report() suppresses the extra lines in that case.
        self.item_count = 0
        self.per_item_min = 0.0
        self.per_item_max = 0.0
        self.batch_size_min = 0
        self.batch_size_max = 0

    def add(self, duration: float, batch_count: int = 1):
        per_item = duration / batch_count if batch_count else duration
        if self.count == 0:
            self.min = duration
            self.max = duration
            self.per_item_min = per_item
            self.per_item_max = per_item
            self.batch_size_min = batch_count
            self.batch_size_max = batch_count
        else:
            if duration < self.min:
                self.min = duration
            if duration > self.max:
                self.max = duration
            if per_item < self.per_item_min:
                self.per_item_min = per_item
            if per_item > self.per_item_max:
                self.per_item_max = per_item
            if batch_count < self.batch_size_min:
                self.batch_size_min = batch_count
            if batch_count > self.batch_size_max:
                self.batch_size_max = batch_count
        self.count += 1
        self.total += duration
        self.item_count += batch_count

    def copy(self) -> "_Stats":
        s = _Stats()
        s.count = self.count
        s.total = self.total
        s.min = self.min
        s.max = self.max
        s.item_count = self.item_count
        s.per_item_min = self.per_item_min
        s.per_item_max = self.per_item_max
        s.batch_size_min = self.batch_size_min
        s.batch_size_max = self.batch_size_max
        return s


class TimingManager:
    def __init__(
        self,
        separate_success_error: bool = True,
        report_to_stdout: bool = True,
    ):
        """
        separate_success_error:
            If True, keep separate stats for successful vs error runs.
            If False, everything is counted as 'success'.
        report_to_stdout:
            If True, print the summary to stdout on exit.
        """
        # { name: {"success": _Stats, "error": _Stats} }
        # Running stats only (count/total/min/max) — storing every duration
        # blew up memory for high-frequency timers (e.g. per-object describer
        # calls at 200 objects/image over many thousands of images).
        self._records = defaultdict(lambda: {"success": _Stats(), "error": _Stats()})
        # Guards _records mutations and report() reads. The global `timer`
        # singleton is shared across all threads in the dataset-generation
        # pipeline; without this lock, concurrent record() calls could race on
        # the defaultdict insertion.
        self._records_lock = threading.Lock()
        self._installed_signal_handlers = False
        self._separate_success_error = separate_success_error
        self._report_to_stdout = report_to_stdout

        # Path to write timing report to on exit (if set)
        self._output_file_path: Optional[str] = None

        # Print/write results on normal program exit
        atexit.register(self.report)

    # -------- Public API --------

    def time_block(self, name: str, batch_count: int = 1):
        """
        Context manager:

            with timer.time_block("some_block"):
                ...

        For batched operations (one timed call covers N items), pass
        batch_count upfront when known, or call set_batch_count(n) on the
        returned context object before the block exits:

            with timer.time_block("vlm_query", batch_count=len(reqs)):
                results = engine.chat_many(reqs)

            with timer.time_block("vlm_query") as t:
                results = engine.chat_many(reqs)
                t.set_batch_count(len(results))
        """
        return _TimingContext(self, name, batch_count=batch_count)

    def record(self, name: str, duration: float, success: bool = True):
        """
        Record a duration (in seconds) for the given block name.

        success = True  -> goes into 'success' bucket
        success = False -> goes into 'error' bucket (if separation enabled)
        """
        self.record_batch(name, duration, batch_count=1, success=success)

    def record_batch(self, name: str, duration: float, batch_count: int,
                     success: bool = True):
        """
        Record a batched duration: one observation that covers `batch_count`
        items processed together. Tracks both batch-level (duration) and
        per-item (duration / batch_count) statistics, plus batch-size min/max.
        """
        if self._separate_success_error:
            bucket = "success" if success else "error"
        else:
            bucket = "success"  # everything lumped together

        with self._records_lock:
            self._records[name][bucket].add(duration, batch_count=batch_count)

    def set_output_file(self, path: str):
        """
        Set the path of the file where the timing report will be written
        on exit (normal or via signal).
        """
        self._output_file_path = path

    def report(self, output_file_path=_UNSET, to_stdout: Optional[bool] = None):
        """
        Print (and optionally write) a summary of all recorded timings.
        This is automatically called at interpreter shutdown and when
        SIGINT/SIGTERM are caught (via install_signal_handlers()).

        For periodic snapshots during a long run, callers can override the
        output destination for a single call:
          - output_file_path: pass a path to write only this snapshot there
            (does not change the manager's persistent setting); pass None to
            skip the file write entirely; omit to use the configured default.
          - to_stdout: pass True/False to override _report_to_stdout for this
            call only; omit to use the configured default.
        """
        if output_file_path is _UNSET:
            output_file_path = self._output_file_path
        if to_stdout is None:
            to_stdout = self._report_to_stdout

        # Snapshot under the lock so we don't iterate a dict that's being
        # mutated by other threads.
        with self._records_lock:
            if not self._records:
                # Do not print anything if there is no timing data
                if output_file_path is not None:
                    # Still clear/create the file so downstream tools don't fail
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        f.write("")  # write empty file
                return
            snapshot = [
                (name, buckets["success"].copy(), buckets["error"].copy())
                for name, buckets in self._records.items()
            ]

        lines = []
        lines.append("\n[Timing] Summary")
        lines.append("-" * 60)

        for name, success_stats, error_stats in snapshot:

            def fmt_stats(label, stats):
                if stats.count == 0:
                    return [f"{label}: (no runs)"]
                avg_t = stats.total / stats.count
                out = [
                    (f"{label}:        count={stats.count:4d}  "
                     f"min={stats.min:.6f}s  "
                     f"max={stats.max:.6f}s  "
                     f"avg={avg_t:.6f}s")
                ]
                # Only emit per-item / batch-size lines when at least one
                # observation was actually a batch (size > 1).
                if stats.item_count > stats.count:
                    per_item_avg = stats.total / stats.item_count
                    batch_avg = stats.item_count / stats.count
                    out.append(
                        f"{label}/item:   items={stats.item_count:4d}  "
                        f"min={stats.per_item_min:.6f}s  "
                        f"max={stats.per_item_max:.6f}s  "
                        f"avg={per_item_avg:.6f}s"
                    )
                    out.append(
                        f"{label}/batch:  size min={stats.batch_size_min}  "
                        f"max={stats.batch_size_max}  "
                        f"avg={batch_avg:.2f}"
                    )
                return out

            lines.append(f"{name}:")
            for line in fmt_stats("SUCCESS", success_stats):
                lines.append("  " + line)
            if self._separate_success_error:
                for line in fmt_stats("ERROR  ", error_stats):
                    lines.append("  " + line)

        output = "\n".join(lines)

        if to_stdout:
            print(output)

        if output_file_path is not None:
            try:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(output + "\n")
            except Exception as e:
                if to_stdout:
                    print(f"[Timing] Failed to write timing report to {output_file_path}: {e}")

    def install_signal_handlers(self):
        """
        Also print/write a report if the program is terminated
        with Ctrl+C (SIGINT) or SIGTERM.
        """
        if self._installed_signal_handlers:
            return  # avoid double-install

        def handler(signum, frame):
            # print(f"\n[Timing] Caught signal {signum}, printing summary before exit...")
            self.report()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        self._installed_signal_handlers = True


class _TimingContext:
    """
    Internal context manager used by TimingManager.
    """

    def __init__(self, manager: TimingManager, name: str, batch_count: int = 1):
        self._manager = manager
        self._name = name
        self._start = None
        self._batch_count = batch_count

    def set_batch_count(self, batch_count: int):
        """
        Override the batch count for this timed block. Useful when the count
        is only known after the call returns (e.g., len(results)). Wins over
        any value passed to time_block().
        """
        self._batch_count = batch_count

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        duration = end - self._start

        # success = True if no exception was raised in the block
        success = exc_type is None
        self._manager.record_batch(
            self._name, duration, self._batch_count, success=success,
        )

        # Do not suppress exceptions
        return False


def querier_timer_suffix(
    *,
    use_vllm_server: bool = True,
    enable_thinking: Optional[bool] = None,
) -> str:
    """
    Build a suffix for per-VLM/LLM-query timer names that captures runtime
    flags. Defaults assume the server (vLLM) path.

    - enable_thinking=True  -> "_thinkOn"
    - enable_thinking=False -> "_thinkOff"
    - enable_thinking=None  -> (omit; querier has no thinking knob)
    - use_vllm_server=False -> "_local"  (otherwise omit; vLLM is the default)
    """
    parts = []
    if enable_thinking is not None:
        parts.append("_thinkOn" if enable_thinking else "_thinkOff")
    if not use_vllm_server:
        parts.append("_local")
    return "".join(parts)


# Global singleton you can import everywhere
timer = TimingManager()
timer.install_signal_handlers()