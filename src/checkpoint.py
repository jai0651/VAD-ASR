"""
Resumable training state — what makes a cheap, interruptible GPU usable.

Spot/interruptible instances are 50-80% cheaper than on-demand, and the only
reason not to use them is that the provider can reclaim the machine mid-run.
That objection disappears entirely if a run can pick up exactly where it left
off, so this file is worth more than any instance-type choice.

WHAT HAS TO BE SAVED, AND WHY WEIGHTS ARE NOT ENOUGH.

  model      obviously.
  optimizer  Adam's first/second moment estimates take hundreds of steps to
             re-warm. Dropping them doesn't crash anything — it just quietly
             costs you accuracy, which is the worst kind of bug because you
             only see it in the final metric.
  step       the LR schedule is a function of it. Restart at 0 mid-run and you
             re-warm up and re-decay, undoing the cosine you already paid for.
  bookkeeping best-so-far score and the checkpoint list, so selection and
             averaging survive the interruption too.

ATOMICITY IS NOT OPTIONAL HERE. The entire point is that the process can be
killed at an arbitrary instant — including halfway through writing this file.
Writing to a temp path and renaming makes the swap atomic on POSIX, so a
reclaim can cost you the last interval's progress but never leaves a truncated
checkpoint that poisons the retry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_resume(path: str | Path, model, optimizer, step: int,
                extra: dict[str, Any] | None = None) -> None:
    """Atomically write everything needed to continue as if nothing happened."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        **(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)          # atomic rename — see module docstring


def load_resume(path: str | Path, model, optimizer, device="cpu"
                ) -> tuple[int, dict[str, Any]]:
    """Restore in place. Returns (step, extra); step 0 when there's nothing
    to resume, so callers need no special-casing on a fresh run."""
    path = Path(path)
    if not path.exists():
        return 0, {}
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    extra = {k: v for k, v in ck.items()
             if k not in ("model", "optimizer", "step")}
    return int(ck["step"]), extra


def clear_resume(path: str | Path) -> None:
    """Call once a run has finished, so the next one starts fresh rather than
    instantly 'resuming' a completed job."""
    Path(path).unlink(missing_ok=True)
    Path(str(path) + ".tmp").unlink(missing_ok=True)
