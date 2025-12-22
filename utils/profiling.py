# utils/profiling.py
from __future__ import annotations

from contextlib import nullcontext
import time
import torch.cuda


import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Literal, Any

import torch


class TimedStageKeys(str, Enum):
    H2D = "time_h2d_ms"
    FWD = "time_fwd_ms"
    LOSS = "time_loss_ms"
    BWD = "time_bwd_ms"
    OPT = "time_opt_ms"


class IterMark(str, Enum):
    """
    Typed CPU marks for iteration-level timing.
    """
    ITER_BEGIN = "iter_begin"         # start of "inside-iteration" work
    ITER_END = "iter_end"             # end of iteration work (after logging/printing)
    BATCH_READY = "batch_ready"       # optional: right when batch is received (top of loop)


@dataclass
class StepTimingsMs:
    # Names match ProfilingLogRow fields (already in your profiling.py)
    time_h2d_ms: float = 0.0
    time_fwd_ms: float = 0.0
    time_loss_ms: float = 0.0
    time_bwd_ms: float = 0.0
    time_opt_ms: float = 0.0


@dataclass
class IterationCpuMs:
    """
    CPU-side iteration timings that are useful for diagnosing input pipeline overhead.

    dataloader_ms:
        Time spent *between* iterations waiting for the next batch from the dataloader
        (includes dataset IO + augmentation + worker time + any queuing).
    iter_cpu_ms:
        CPU wall-time spent inside this iteration (Python + framework overhead), excluding dataloader wait.
    """
    dataloader_ms: float = 0.0
    iter_cpu_ms: float = 0.0


class StepTimer:
    """
    Strongly typed timer for a training iteration.

    GPU stage timing:
        start(stage) / end(stage) record CUDA events (when enabled on CUDA), else CPU fallback.

    CPU iteration timing:
        mark(IterMark.ITER_BEGIN), mark(IterMark.ITER_END) each iteration.
        dataloader wait time is computed as time between last ITER_END and current ITER_BEGIN.
    """

    def __init__(self, enabled: bool = True, device: Literal["cuda", "cpu"] = "cuda"):
        self.enabled = enabled
        self.device = device

        self._use_cuda_events = enabled and device == "cuda" and torch.cuda.is_available()
        self._stage_times_cuda: Optional[Dict[TimedStageKeys, tuple[torch.cuda.Event, torch.cuda.Event]]] = None
        if self._use_cuda_events:
            self._stage_times_cuda = {
                s: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for s in TimedStageKeys
            }

        # CPU marks
        self._cpu_marks: Dict[IterMark, float] = {}

        # For CPU fallback stage timing
        self._stage_times_cpu_start: Dict[TimedStageKeys, float] = {}
        self._stage_times_cpu_end: Dict[TimedStageKeys, float] = {}

    # ---------- CPU marks (typed) ----------

    def mark(self, m: IterMark) -> None:
        if not self.enabled:
            return
        self._cpu_marks[m] = time.perf_counter()

    def _mark_time(self, m: IterMark) -> Optional[float]:
        return self._cpu_marks.get(m)

    # ---------- GPU/Stage timing ----------

    def start(self, stage: TimedStageKeys) -> None:
        if not self.enabled:
            return
        if self._stage_times_cuda is not None:
            self._stage_times_cuda[stage][0].record()
        else:
            self._stage_times_cpu_start[stage] = time.perf_counter()

    def end(self, stage: TimedStageKeys) -> None:
        if not self.enabled:
            return
        if self._stage_times_cuda is not None:
            self._stage_times_cuda[stage][1].record()
        else:
            self._stage_times_cpu_end[stage] = time.perf_counter()

    def snapshot_gpu(self, *, sync_cuda: bool = True) -> StepTimingsMs:
        """
        Reads GPU stage times into a StepTimingsMs whose member names match ProfilingLogRow.
        If using CUDA events, you typically want sync_cuda=True once per iteration.
        """
        if not self.enabled:
            return StepTimingsMs()

        out = StepTimingsMs()

        if self._stage_times_cuda is not None:
            if sync_cuda:
                torch.cuda.synchronize()
            for stage, (t0, t1) in self._stage_times_cuda.items():
                ms = float(0.0)
                try:
                    ms = float(t0.elapsed_time(t1))
                except:
                    pass
                if stage == TimedStageKeys.H2D:
                    out.time_h2d_ms = ms
                elif stage == TimedStageKeys.FWD:
                    out.time_fwd_ms = ms
                elif stage == TimedStageKeys.LOSS:
                    out.time_loss_ms = ms
                elif stage == TimedStageKeys.BWD:
                    out.time_bwd_ms = ms
                elif stage == TimedStageKeys.OPT:
                    out.time_opt_ms = ms
            return out

        # CPU fallback stage timing
        for stage, t0 in self._stage_times_cpu_start.items():
            t1 = self._stage_times_cpu_end.get(stage, t0)
            ms = (t1 - t0) * 1000.0
            if stage == TimedStageKeys.H2D:
                out.time_h2d_ms = ms
            elif stage == TimedStageKeys.FWD:
                out.time_fwd_ms = ms
            elif stage == TimedStageKeys.LOSS:
                out.time_loss_ms = ms
            elif stage == TimedStageKeys.BWD:
                out.time_bwd_ms = ms
            elif stage == TimedStageKeys.OPT:
                out.time_opt_ms = ms

        return out

    # ---------- Dataloader/CPU iteration separation ----------

    def snapshot_cpu_iteration(self) -> IterationCpuMs:
        """
        Computes:
          - dataloader_ms: time between previous ITER_END and current ITER_BEGIN
          - iter_cpu_ms: time between ITER_BEGIN and ITER_END (this iteration's CPU wall-time)
        """
        if not self.enabled:
            return IterationCpuMs()

        t_begin = self._mark_time(IterMark.ITER_BEGIN)
        t_end = self._mark_time(IterMark.ITER_END)

        # dataloader wait: previous end -> current begin
        prev_end = self._cpu_marks.get(IterMark.ITER_END)  # last recorded end
        # However, because ITER_END is overwritten each iter, we need to store previous value.
        # We'll do it via an internal stash:
        # If you call mark_iter_begin(), it will compute dataloader_ms using a stored prev_end.
        # So snapshot_cpu_iteration() is mainly for iter_cpu_ms.
        iter_cpu_ms = 0.0
        if t_begin is not None and t_end is not None and t_end >= t_begin:
            iter_cpu_ms = (t_end - t_begin) * 1000.0

        return IterationCpuMs(dataloader_ms=0.0, iter_cpu_ms=iter_cpu_ms)

    # Internal stash for dataloader wait computation
    _prev_iter_end: Optional[float] = None
    _last_dataloader_ms: float = 0.0

    def mark_iter_begin(self) -> None:
        """
        Call at the top of each iteration *after* you receive the batch.
        Computes dataloader wait as prev_iter_end -> now.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._prev_iter_end is not None and now >= self._prev_iter_end:
            self._last_dataloader_ms = (now - self._prev_iter_end) * 1000.0
        else:
            self._last_dataloader_ms = 0.0
        self._cpu_marks[IterMark.ITER_BEGIN] = now

    def mark_iter_end(self) -> None:
        """
        Call at the end of each iteration.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        self._cpu_marks[IterMark.ITER_END] = now
        self._prev_iter_end = now

    def last_dataloader_ms(self) -> float:
        """
        Returns dataloader wait computed in mark_iter_begin().
        """
        if not self.enabled:
            return 0.0
        return self._last_dataloader_ms

    def last_iter_cpu_ms(self) -> float:
        """
        Returns (ITER_END - ITER_BEGIN) in ms for this iteration.
        """
        if not self.enabled:
            return 0.0
        t_begin = self._mark_time(IterMark.ITER_BEGIN)
        t_end = self._mark_time(IterMark.ITER_END)
        if t_begin is None or t_end is None or t_end < t_begin:
            return 0.0
        return (t_end - t_begin) * 1000.0

    # ---------- ProfilingLogRow helper ----------

    def to_profiling_kwargs(self, *, do_step: bool, sync_cuda: bool = True) -> Dict[str, Any]:
        """
        Returns kwargs that match ProfilingLogRow's timing fields:
          time_h2d_ms, time_fwd_ms, time_loss_ms, time_bwd_ms, time_opt_ms

        Note: OPT time can be forced to 0.0 when do_step is False.
        """
        g = self.snapshot_gpu(sync_cuda=sync_cuda)
        return {
            TimedStageKeys.H2D: g.time_h2d_ms,
            TimedStageKeys.FWD: g.time_fwd_ms,
            TimedStageKeys.LOSS: g.time_loss_ms,
            TimedStageKeys.BWD: g.time_bwd_ms,
            TimedStageKeys.OPT: g.time_opt_ms if do_step else 0.0,
            "time_dataloader_ms": self.last_dataloader_ms(),
            "time_iter_cpu_ms": self.last_iter_cpu_ms(),
        }



# csv_types.py
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProfilingLogRow:
    utc_time: str
    machine_name: str
    model_name: str
    workers: int
    train_name: str
    img_y: int
    img_x: int
    batch_size: int
    gpu_index: int
    gpu_name: str
    epoch: int
    iter: int 
    gpu_utilization: float
    gpu_mem_gb: float
    time_h2d_ms: float
    time_fwd_ms: float
    time_loss_ms: float
    time_bwd_ms: float
    time_opt_ms: float
    time_dataloader_ms: float = 0.0
    time_iter_cpu_ms: float = 0.0







# @dataclass(frozen=True)
# class TrainLogRow:
#     epoch: int
#     loss_box: float
#     loss_obj: float
#     loss_cls: float
#     loss_total: float
#     lr: float
#     img_size: int
#     batch_size: int
#     gpu_mem_gb: float
#     iter: Optional[int] = None  # optional for per-iteration logging


# csv_logger.py
import csv
from dataclasses import fields, asdict
from pathlib import Path
from typing import Generic, Type, TypeVar

T = TypeVar("T")


# Example usage:
# with CSVLogger(Path("profiling_log.csv"), ProfilingLogRow) as profilingLogger: 

class CSVLogger(Generic[T]):
    def __init__(self, path: Path, row_type: Type[T]):
        self.path = Path(path)
        self.row_type = row_type

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[f.name for f in fields(row_type)]
        )
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: T) -> None:
        if not isinstance(row, self.row_type):
            raise TypeError(
                f"Expected row of type {self.row_type.__name__}, "
                f"got {type(row).__name__}"
            )

        self.writer.writerow(asdict(row))
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
