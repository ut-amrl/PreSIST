
from typing import Dict


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MeterDict:
    """
    Tracks a dynamic set of AverageMeters keyed by metric name.
    """
    def __init__(self):
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, metrics: Dict[str, float], n: int):
        for k, v in metrics.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v, n=n)

    def averages(self, prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}{k}": float(m.avg) for k, m in self.meters.items()}
