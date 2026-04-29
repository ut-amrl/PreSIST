# timing.py
import time
import atexit
import signal
from collections import defaultdict
from typing import Optional


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
        # { name: {"success": [durations...], "error": [durations...]} }
        self._records = defaultdict(lambda: {"success": [], "error": []})
        self._installed_signal_handlers = False
        self._separate_success_error = separate_success_error
        self._report_to_stdout = report_to_stdout

        # Path to write timing report to on exit (if set)
        self._output_file_path: Optional[str] = None

        # Print/write results on normal program exit
        atexit.register(self.report)

    # -------- Public API --------

    def time_block(self, name: str):
        """
        Context manager:

            with timer.time_block("some_block"):
                ...

        Records the elapsed time for the block under 'name'.
        """
        return _TimingContext(self, name)

    def record(self, name: str, duration: float, success: bool = True):
        """
        Record a duration (in seconds) for the given block name.

        success = True  -> goes into 'success' bucket
        success = False -> goes into 'error' bucket (if separation enabled)
        """
        if self._separate_success_error:
            bucket = "success" if success else "error"
        else:
            bucket = "success"  # everything lumped together

        self._records[name][bucket].append(duration)

    def set_output_file(self, path: str):
        """
        Set the path of the file where the timing report will be written
        on exit (normal or via signal).
        """
        self._output_file_path = path

    def report(self):
        """
        Print (and optionally write) a summary of all recorded timings.
        This is automatically called at interpreter shutdown and when
        SIGINT/SIGTERM are caught (via install_signal_handlers()).
        """
        if not self._records:
            # Do not print anything if there is no timing data
            if self._output_file_path is not None:
                # Still clear/create the file so downstream tools don't fail
                with open(self._output_file_path, "w", encoding="utf-8") as f:
                    f.write("")  # write empty file
            return

        lines = []
        lines.append("\n[Timing] Summary")
        lines.append("-" * 60)

        for name, buckets in self._records.items():
            success_times = buckets["success"]
            error_times = buckets["error"]

            def fmt_stats(label, times):
                if not times:
                    return f"{label}: (no runs)"
                count = len(times)
                total = sum(times)
                min_t = min(times)
                max_t = max(times)
                avg_t = total / count if count else 0.0
                return (f"{label}: count={count:4d}  "
                        f"min={min_t:.6f}s  "
                        f"max={max_t:.6f}s  "
                        f"avg={avg_t:.6f}s")

            lines.append(f"{name}:")
            lines.append("  " + fmt_stats("SUCCESS", success_times))
            if self._separate_success_error:
                lines.append("  " + fmt_stats("ERROR  ", error_times))

        output = "\n".join(lines)

        if self._report_to_stdout:
            print(output)

        if self._output_file_path is not None:
            try:
                with open(self._output_file_path, "w", encoding="utf-8") as f:
                    f.write(output + "\n")
            except Exception as e:
                if self._report_to_stdout:
                    print(f"[Timing] Failed to write timing report to {self._output_file_path}: {e}")

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

    def __init__(self, manager: TimingManager, name: str):
        self._manager = manager
        self._name = name
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self  # not used, but conventional

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        duration = end - self._start

        # success = True if no exception was raised in the block
        success = exc_type is None
        self._manager.record(self._name, duration, success=success)

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