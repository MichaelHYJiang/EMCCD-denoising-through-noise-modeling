from emccd_denoising.io import read_tiff


def test_example_pair_matches():
    noisy = read_tiff("examples/paper_pair/input.tif")
    target = read_tiff("examples/paper_pair/ground_truth.tif")
    assert noisy.shape == target.shape
    assert noisy.ndim == 2


def test_microscopy_example_is_grayscale():
    assert read_tiff("examples/microscopy/input.tif").ndim == 2
