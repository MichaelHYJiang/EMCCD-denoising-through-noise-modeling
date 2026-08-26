import pytest

torch = pytest.importorskip("torch")

from emccd_denoising.scheduler import GradualWarmupScheduler


def test_paper_warmup_learning_rates():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 997, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, 1, 3, cosine)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4 / 3)
    optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(4e-4 / 3)
    optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
