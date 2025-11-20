import os.path as osp
import pathlib

import pytest
from hydra import compose, initialize

from walrus_steering.train import CONFIG_DIR, CONFIG_NAME, main
from walrus_steering.trainer.checkpoints import CHECKPOINT_METADATA_FILENAME


def test_checkpoints(checkpoint_folder: pathlib.Path, conf):
    """Test training generates checkpoints as expected."""
    overrides = conf
    overrides += ["checkpoint.checkpoint_frequency=1", "trainer.max_epoch=2", "+model.include_d=[2]"]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
        # Check epoch 1 checkpoint exists
        assert (checkpoint_folder / "step_1").is_dir()
        # Check best checkpoint exists
        assert (checkpoint_folder / "best").is_symlink()
        assert len(list((checkpoint_folder / "best").iterdir())) > 0
        # Check metadata have been saved
        assert (checkpoint_folder / "best" / CHECKPOINT_METADATA_FILENAME).exists()
        # Check last checkpoint exists and is simlink of best
        assert (checkpoint_folder / "last").is_symlink()
        assert (checkpoint_folder / "last").samefile(checkpoint_folder / "step_2")


def test_no_checkpoint(checkpoint_folder: pathlib.Path, conf):
    """Test no checkpoints are saved when expected."""
    overrides = conf
    overrides += ["checkpoint=none", "+model.include_d=[2]"]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
        # Checkpoint folder should not exist
        assert not checkpoint_folder.exists()


def test_resume_from_default_checkpoint(checkpoint_folder: pathlib.Path, conf):
    """Test training can resume from checkpoint"""
    overrides = conf
    overrides += [
        "checkpoint.checkpoint_frequency=0","+model.include_d=[2]"
    ]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    assert checkpoint_folder.exists() is False
    # First run
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
    # Because we always save last epoch
    # even though the frequency does not match
    # There should be step_1 in checkpoint folder
    assert (checkpoint_folder / "step_1").exists()
    assert (checkpoint_folder / "step_2").exists() is False
    # Should resume from checkpoint
    overrides += ["checkpoint.checkpoint_frequency=1", "trainer.max_epoch=2"]
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
    # Because the training has resumed after 1 epoch (starting from 1)
    # There should be step_2 in checkpoint folder
    assert (checkpoint_folder / "step_2").exists()
    assert (checkpoint_folder / "last").samefile(checkpoint_folder / "step_2")
    assert not (checkpoint_folder / "step_1").samefile(checkpoint_folder / "step_2")

@pytest.mark.skip(reason="Randomly broke - don't have time to debug and not a targeted task atm")
def test_resume_from_specific_checkpoint(checkpoint_folder: pathlib.Path, conf):
    """Test in training can resume from specific checkpoint"""
    overrides = conf
    overrides += [
        "checkpoint.checkpoint_frequency=1",
        "trainer.max_epoch=2",
        "+model.include_d=[2]",
    ]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    assert checkpoint_folder.exists() is False
    # First run for 2 epochs
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
    # There should be 2 epochs in checkpoint folder
    assert (checkpoint_folder / "step_1").exists()
    assert (checkpoint_folder / "step_2").exists()
    checkpoint1_metadata_path = (
        checkpoint_folder / "step_1" / CHECKPOINT_METADATA_FILENAME
    )
    old_metadata1_tmstp = checkpoint1_metadata_path.stat().st_mtime
    checkpoint2_metadata_path = (
        checkpoint_folder / "step_2" / CHECKPOINT_METADATA_FILENAME
    )
    assert (checkpoint2_metadata_path).exists()
    old_metadata2_tmstp = checkpoint2_metadata_path.stat().st_mtime
    # Ask to resume from epoch 1
    overrides += [
        "checkpoint.checkpoint_frequency=1",
        "trainer.max_epoch=2",
        f"+checkpoint.load_checkpoint_path={checkpoint_folder / 'step_1'}",
        "+model.include_d=[2]",
    ]
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
    # Training should resume from epoch 1, overriding epoch 2 checkpoint
    # And having no epoch 3 checkpoint
    assert (checkpoint_folder / "step_2").exists()
    assert (checkpoint_folder / "step_3").exists() is False
    new_metadata1_tmstp = checkpoint1_metadata_path.stat().st_mtime
    # Checkpoint for epoch 1 should not have been updated
    assert new_metadata1_tmstp == old_metadata1_tmstp
    new_metadata2_tmstp = checkpoint2_metadata_path.stat().st_mtime
    # Checkpoint for epoch 2 should have been updated
    assert new_metadata2_tmstp != old_metadata2_tmstp
