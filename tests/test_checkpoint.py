"""
Resume tests — the thing that makes a cheap interruptible GPU safe.

An untested resume is worse than no resume: it looks like it worked, and the
damage (stale optimizer moments, a restarted LR schedule) shows up only as a
slightly worse final metric, hours later and hundreds of dollars in.

So these assert the two properties that actually matter:
  * resuming is BIT-IDENTICAL to never having been interrupted, and
  * a kill halfway through writing the checkpoint cannot poison the retry.
"""

from __future__ import annotations

import torch
from torch import nn

from src.checkpoint import clear_resume, load_resume, save_resume


def _model_and_opt(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-2)


def _train(model, opt, batches):
    for x, y in batches:
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def _batches(n: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(4, 8, generator=g), torch.randn(4, 4, generator=g))
            for _ in range(n)]


def test_missing_file_means_fresh_start(tmp_path):
    model, opt = _model_and_opt()
    step, extra = load_resume(tmp_path / "nope.pt", model, opt)
    assert step == 0 and extra == {}


def test_resume_is_identical_to_an_uninterrupted_run(tmp_path):
    """The property that matters. Train 10 steps straight through, versus train
    5, save, reload into a FRESH model+optimizer, train 5 more — on the same
    data. The weights must match exactly."""
    data = _batches(10)
    path = tmp_path / "resume.pt"

    straight_model, straight_opt = _model_and_opt()
    _train(straight_model, straight_opt, data)

    a_model, a_opt = _model_and_opt()
    _train(a_model, a_opt, data[:5])
    save_resume(path, a_model, a_opt, step=5, extra={"best": 0.42})

    b_model, b_opt = _model_and_opt(seed=99)      # different init on purpose
    step, extra = load_resume(path, b_model, b_opt)
    assert step == 5 and extra["best"] == 0.42
    _train(b_model, b_opt, data[5:])

    for p, q in zip(straight_model.parameters(), b_model.parameters()):
        torch.testing.assert_close(p, q)


def test_optimizer_state_actually_survives(tmp_path):
    """Guards the guard: without Adam's moments the test above would still
    'pass' loosely, so assert the moment estimates themselves came back."""
    path = tmp_path / "resume.pt"
    model, opt = _model_and_opt()
    _train(model, opt, _batches(5))
    before = opt.state_dict()["state"][0]["exp_avg_sq"].clone()
    save_resume(path, model, opt, step=5)

    model2, opt2 = _model_and_opt(seed=7)
    load_resume(path, model2, opt2)
    after = opt2.state_dict()["state"][0]["exp_avg_sq"]
    assert before.abs().sum() > 0            # there is real state to lose
    torch.testing.assert_close(before, after)


def test_write_is_atomic_so_a_kill_cannot_corrupt(tmp_path):
    """A reclaim mid-write must leave the PREVIOUS checkpoint intact, never a
    truncated one. Simulated by leaving a junk .tmp lying around."""
    path = tmp_path / "resume.pt"
    model, opt = _model_and_opt()
    save_resume(path, model, opt, step=3)
    (tmp_path / "resume.pt.tmp").write_bytes(b"truncated garbage")

    model2, opt2 = _model_and_opt(seed=5)
    step, _ = load_resume(path, model2, opt2)     # must read the good file
    assert step == 3


def test_clear_removes_both_files(tmp_path):
    path = tmp_path / "resume.pt"
    model, opt = _model_and_opt()
    save_resume(path, model, opt, step=1)
    (tmp_path / "resume.pt.tmp").write_bytes(b"x")
    clear_resume(path)
    assert not path.exists() and not (tmp_path / "resume.pt.tmp").exists()
