import pytest

torch = pytest.importorskip("torch")

from emccd_denoising.model import build_uformer


def test_small_model_forward_shape():
    model = build_uformer(128).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 128, 128))
    assert output.shape == (1, 1, 128, 128)
