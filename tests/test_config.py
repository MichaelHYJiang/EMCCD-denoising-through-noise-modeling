from emccd_denoising.config import load_config


def test_finetune_config_inherits_paper_settings():
    config = load_config("configs/cell_finetune.yaml")
    assert config["noise"]["max_ratio"] == 20.0
    assert config["training"]["epochs"] == 2000
    assert config["training"]["batch_size"] == 4
    assert config["training"]["historical_full_frame_noise"] is True
    assert config["training"]["historical_first_epoch_eval_mode"] is True
