import numpy as np
import pytest

from emccd_denoising.noise import Calibration, EMCCDNoiseModel


def calibration():
    return Calibration(gain=2.5, read_noise_samples=np.array([[0.0, 3.0], [1.0, 4.0]]))


def test_noise_is_deterministic_and_bounded():
    model = EMCCDNoiseModel(calibration())
    clean = np.full((32, 32), 1000.0)
    first, ratio = model.synthesize(clean, ratio=10, seed=7)
    second, _ = model.synthesize(clean, ratio=10, seed=7)
    np.testing.assert_array_equal(first, second)
    assert ratio == 10
    assert first.shape == clean.shape
    assert first.dtype == np.float32
    assert 0 <= first.min() <= first.max() <= 1


def test_noise_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="ratio"):
        EMCCDNoiseModel(calibration()).synthesize(np.ones((4, 4)), ratio=21)


def test_calibration_loads_and_validates(tmp_path):
    np.save(tmp_path / "params.npy", np.arange(1, 10, dtype=float))
    np.save(tmp_path / "S0-sigma-between-0.001s-0.2s.npy", np.ones((3, 2)))
    loaded = Calibration.load(tmp_path)
    assert loaded.gain == 1
    assert loaded.read_noise_samples.shape == (3, 2)


def test_spatial_read_noise_calibration(tmp_path):
    np.save(tmp_path / "params.npy", np.arange(1, 10, dtype=float))
    np.save(tmp_path / "S0-sigma-between-0.001s-0.2s.npy", np.ones((3, 2, 8, 8)))
    loaded = Calibration.load(tmp_path)
    noisy, _ = EMCCDNoiseModel(loaded).synthesize(np.ones((4, 4)), seed=2)
    assert noisy.shape == (4, 4)
