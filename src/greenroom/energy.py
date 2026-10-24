"""Wall-clock and GPU-energy measurement per pipeline stage.

GPU power is sampled through NVML (pynvml or nvidia-ml-py, whichever is
installed) in a background thread and integrated to joules. When NVML is
unavailable the meter degrades to wall-clock only and says so in its result,
so a missing dependency never fails a run and never produces a fake number.
"""
from __future__ import annotations

import threading
import time
import warnings


class StageMeter:
    def __init__(self, name: str, gpu: bool = True, interval_s: float = 0.5, device_index: int = 0):
        self.name = name
        self.want_gpu = gpu
        self.interval_s = interval_s
        self.device_index = device_index
        self.wall_seconds = 0.0
        self.gpu_joules: float | None = None
        self.gpu_samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self._joules = 0.0

    def _init_nvml(self) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml  # provided by pynvml or nvidia-ml-py

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml = pynvml
            return True
        except Exception:  # noqa: BLE001
            self._nvml = None
            return False

    def _sample_loop(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            now = time.perf_counter()
            try:
                milliwatts = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
            except Exception:  # noqa: BLE001
                break
            self._joules += (milliwatts / 1000.0) * (now - last)
            self.gpu_samples += 1
            last = now

    def __enter__(self) -> "StageMeter":
        self._t0 = time.perf_counter()
        if self.want_gpu and self._init_nvml():
            self._thread = threading.Thread(target=self._sample_loop, name=f"nvml-{self.name}", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_seconds = time.perf_counter() - self._t0
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self.gpu_joules = round(self._joules, 2)
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def to_dict(self) -> dict:
        return {
            "stage": self.name,
            "wall_seconds": round(self.wall_seconds, 3),
            "gpu_joules": self.gpu_joules,
            "gpu_power_samples": self.gpu_samples,
            "gpu_energy_measured": self.gpu_joules is not None,
        }
