"""Low-overhead resource sampling during benchmarks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil

from app.core.config import BYTES_PER_MIB


@dataclass
class RawSample:
    elapsed_seconds: float
    cpu_pct: float
    process_cpu_pct: float
    ram_mb: float
    gpu_util_pct: float | None = None
    gpu_memory_used_mb: float | None = None


@dataclass
class SamplerSummary:
    samples: list[RawSample] = field(default_factory=list)
    cpu_avg: float = 0.0
    cpu_peak: float = 0.0
    process_cpu_avg: float = 0.0
    ram_peak_mb: float = 0.0
    gpu_avg: float | None = None
    gpu_peak: float | None = None
    vram_baseline_mb: float | None = None
    vram_peak_mb: float | None = None


class ResourceSampler:
    """Sample system CPU, process RSS, and NVML GPU telemetry in a thread."""

    def __init__(self, gpu_index: int | None = None, interval: float = 0.05) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: list[RawSample] = []
        self._started_at = 0.0
        self._process = psutil.Process()
        self._nvml = None
        self._handle = None

    def _init_nvml(self) -> None:
        if self.gpu_index is None:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:  # noqa: BLE001 - GPU telemetry is best effort
            self._nvml = None
            self._handle = None

    def _gpu_values(self) -> tuple[float | None, float | None]:
        if self._nvml is None or self._handle is None:
            return None, None
        try:
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            return float(utilization.gpu), round(memory.used / BYTES_PER_MIB, 1)
        except Exception:  # noqa: BLE001
            return None, None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            gpu_util, gpu_memory = self._gpu_values()
            self._records.append(
                RawSample(
                    elapsed_seconds=round(time.perf_counter() - self._started_at, 4),
                    cpu_pct=psutil.cpu_percent(interval=None),
                    process_cpu_pct=self._process.cpu_percent(interval=None),
                    ram_mb=round(self._process.memory_info().rss / BYTES_PER_MIB, 1),
                    gpu_util_pct=gpu_util,
                    gpu_memory_used_mb=gpu_memory,
                )
            )

    def _capture_final(self) -> None:
        gpu_util, gpu_memory = self._gpu_values()
        self._records.append(
            RawSample(
                elapsed_seconds=round(time.perf_counter() - self._started_at, 4),
                cpu_pct=psutil.cpu_percent(interval=None),
                process_cpu_pct=self._process.cpu_percent(interval=None),
                ram_mb=round(self._process.memory_info().rss / BYTES_PER_MIB, 1),
                gpu_util_pct=gpu_util,
                gpu_memory_used_mb=gpu_memory,
            )
        )

    def start(self) -> None:
        # Prime psutil's non-blocking counters so the first sample is meaningful.
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        self._init_nvml()
        self._records = []
        self._stop.clear()
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="mlcp-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> SamplerSummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._started_at:
            self._capture_final()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
        summary = SamplerSummary(samples=list(self._records))
        if not self._records:
            return summary
        summary.cpu_avg = round(sum(s.cpu_pct for s in self._records) / len(self._records), 2)
        summary.cpu_peak = round(max(s.cpu_pct for s in self._records), 2)
        summary.process_cpu_avg = round(
            sum(s.process_cpu_pct for s in self._records) / len(self._records), 2
        )
        summary.ram_peak_mb = max(s.ram_mb for s in self._records)
        gpu_utils = [s.gpu_util_pct for s in self._records if s.gpu_util_pct is not None]
        gpu_memory = [s.gpu_memory_used_mb for s in self._records if s.gpu_memory_used_mb is not None]
        if gpu_utils:
            summary.gpu_avg = round(sum(gpu_utils) / len(gpu_utils), 2)
            summary.gpu_peak = round(max(gpu_utils), 2)
        if gpu_memory:
            summary.vram_baseline_mb = round(gpu_memory[0], 1)
            summary.vram_peak_mb = round(max(gpu_memory), 1)
        return summary
