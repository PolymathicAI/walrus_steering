import pytest
from the_well.data.augmentation import Resize
from the_well.data.datasets import WellDataset

from walrus_steering.data.inflated_dataset import InflatedWellDataset
from walrus_steering.data.multidatamodule import MixedWellDataModule
from walrus_steering.data.multidataset import MixedWellDataset


def test_datamodule(dummy_dataset):
    well_base_path = dummy_dataset
    print(dummy_dataset)
    data_module = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "dummy": {"include_filters": [], "exclude_filters": []},
        },
        batch_size=1,
        data_workers=1,
        max_samples=20,
    )
    assert hasattr(data_module, "train_dataset")
    assert hasattr(data_module, "train_dataloader")
    for batch_index, batch in enumerate(data_module.train_dataloader(), start=1):
        assert "input_fields" in batch
    assert batch_index == data_module.max_samples



@pytest.fixture()
def dummy_resized_dataset(dummy_dataset):
    augmentation = Resize(target_size=16, interpolation_mode="bilinear")
    dataset = MixedWellDataset(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {
                "include_filters": [],
                "exclude_filters": [],
            }
        },
        well_split_name="train",
        use_normalization=False,
        n_steps_input=5,
        n_steps_output=0,
        transform=augmentation,
        dataset_kws={"pad_cartesian_data_to_d": 2}
    )
    return dataset


def test_dummy_resized_dataset(dummy_resized_dataset):
    itm = next(iter(dummy_resized_dataset))
    if isinstance(dummy_resized_dataset.sub_dsets[0], InflatedWellDataset):
        assert itm["input_fields"].shape == (1, 5, 16, 16, 2)
        assert itm["space_grid"].shape == (1, 16, 16, 2)
    elif isinstance(dummy_resized_dataset.sub_dsets[0], WellDataset):
        assert itm["input_fields"].shape == (5, 16, 16, 2)
        assert itm["space_grid"].shape == (16, 16, 2)
    else:
        raise ValueError(f"Unknown dataset type: {type(dummy_resized_dataset.sub_dsets[0])}")




@pytest.fixture()
def dummy_resized_datamodule(dummy_dataset):
    augmentation = Resize(target_size=16, interpolation_mode="bilinear")
    data_module = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {
                "include_filters": [],
                "exclude_filters": [],
            }
        },
        data_workers=1,
        batch_size=8,
        use_normalization=False,
        n_steps_input=5,
        n_steps_output=0,
        transform=augmentation,
        dataset_kws={"pad_cartesian_data_to_d": 2,}
    )
    return data_module


def test_dummy_resized_datamodule(dummy_resized_datamodule):
    trainloader = dummy_resized_datamodule.train_dataloader()
    train_batch = next(iter(trainloader))
    assert train_batch["input_fields"].shape == (8, 5, 16, 16, 2)

    val_dataloaders = dummy_resized_datamodule.val_dataloaders()
    for ldr in val_dataloaders:
        val_batch = next(iter(ldr))
        assert val_batch["input_fields"].shape == (8, 5, 16, 16, 2)
