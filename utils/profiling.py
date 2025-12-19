from contextlib import nullcontext
import time
import torch.cuda

class StepTimer:
    def __init__(self, enabled=True, device="cuda"):
        self.enabled = enabled
        self.device = device
        if enabled and device == "cuda" and torch.cuda.is_available():
            self.ev = {k: (torch.cuda.Event(True), torch.cuda.Event(True)) for k in
                       ["h2d","fwd","loss","bwd","opt"]}
        else:
            self.ev = None
        self.t = {}

    def _cpu_now(self):
        return time.perf_counter()

    def cpu_mark(self, name):
        if not self.enabled: return
        self.t[name] = self._cpu_now()

    def cpu_elapsed_ms(self, name):
        if not self.enabled: return 0.0
        return (self._cpu_now() - self.t.get(name, self._cpu_now())) * 1000.0

    def gpu_start(self, name):
        if self.ev is None: return
        self.ev[name][0].record()

    def gpu_end(self, name):
        if self.ev is None: return
        self.ev[name][1].record()

    def gpu_elapsed_ms(self, name):
        if self.ev is None: return 0.0
        torch.cuda.synchronize()
        s,e = self.ev[name]
        return s.elapsed_time(e)



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
