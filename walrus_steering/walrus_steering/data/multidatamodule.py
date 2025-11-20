import logging
from typing import Dict, List, Literal, Optional, Union

import torch
from the_well.data.augmentation import Augmentation
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, Sampler
from torch.utils.data._utils.collate import default_collate

from .mixed_dset_sampler import BatchedMultisetSampler
from .multidataset import MixedWellDataset
from .utils import get_dict_depth

logger = logging.getLogger(__name__)


def metadata_aware_collate(batch):
    """Collate function that is aware of the metadata of the dataset."""
    # Metadata constant per batch
    metadata = batch[0]["metadata"]
    # Remove metadata from current dicts
    [sample.pop("metadata") for sample in batch]
    batch = default_collate(batch)  # Returns stacked dictionary
    batch["metadata"] = metadata
    return batch


class MixedWellDataModule:
    def __init__(
        self,
        *,
        well_base_path: str,
        well_dataset_info: Dict[
            str,
            Dict[
                Literal[
                    "include_filters", "exclude_filters", "path", "field_transforms"
                ],
                List[str] | str | Dict[str, str],
            ],
        ],
        batch_size: int,
        use_normalization: bool = False,
        field_index_map_override: Dict[str, int] = {},
        max_rollout_steps: int = 100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        world_size: int = 1,
        rank: int = 1,
        data_workers: int = 4,
        epoch: int = 0,
        max_samples: int = 2000,
        transform: Optional[
            Union[
                Augmentation,
                Dict[str, Augmentation],
                Dict[
                    Literal["train", "val", "rollout_val", "test", "rollout_test"],
                    Dict[str, Augmentation],
                ],
            ]
        ] = None,
        global_field_transforms: Optional[Dict[str, str]] = {},
        storage_kwargs: Optional[Dict] = None,
        dataset_kws: Optional[
            Union[
                dict,
                Dict[str, dict],
                Dict[
                    Literal["train", "val", "rollout_val", "test", "rollout_test"],
                    Dict[str, dict],
                ],
            ]
        ] = None,
        validation_mode: bool = False,

    ):
        """Data module class to yield batches of samples.

        Parameters
        ----------
        well_base_path:
            Path to the base directory for the Well dataset.
        well_dataset_info:
            Dictionary containing for each dataset:
            - include_filters: List of strings to filter files to include
            - exclude_filters: List of strings to filter files to exclude
            - path: Optional custom path for this specific dataset
            - field_transforms: Optional dictionary of field transformations
        batch_size:
            Size of the batches yielded by the dataloaders
        max_rollout_steps:
            Maximum number of steps to rollout for the full trajectory mode.
        n_steps_input:
            Number of simulation time frames to include in input.
        n_steps_output:
            Number of simulation time frames to include in output.
        dt_stride:
            Stride for the time dimension.
        world_size:
            Number of total processes in the distributed setting.
        rank:
            Rank of the current GPU in the full torchrun world.
        data_workers:
            Number of workers for the dataloaders in the given process.
        epoch:
            Current epoch number.
        max_samples:
            Maximum number of samples to use for a single training loop.
        transform:
            Transformations to apply to the data.
        storage_kwargs:
            Storage options passed to fsspec for accessing the raw data.
        dataset_kws:
            Additional keyword arguments to pass to each dataset.
        """
        self.global_field_transforms = global_field_transforms or {}
        if transform is not None:
            # If transform is a single Augmentation, apply it to all datasets
            if isinstance(transform, Augmentation):
                transform = {dataset: transform for dataset in well_dataset_info.keys()}

            # If transform is a Dict[str, Augmentation], apply it to all splits
            if isinstance(transform, dict) and all(
                isinstance(k, str) and isinstance(v, Augmentation)
                for k, v in transform.items()
            ):
                transform = {
                    data_split: transform
                    for data_split in [
                        "train",
                        "val",
                        "rollout_val",
                        "test",
                        "rollout_test",
                    ]
                }

            # If transform keys are not a subset of train, val, rollout_val, test, rollout_test, raise an error
            assert set(
                transform.keys()
            ).issubset(
                set(["train", "val", "rollout_val", "test", "rollout_test"])
            ), f"Expected transform keys {transform.keys()} to be a subset of train, val, rollout_val, test, rollout_test."

        if dataset_kws is not None:
            # If dataset_kws is a single dict with depth 1, apply it to all datasets
            if isinstance(dataset_kws, dict) and get_dict_depth(dataset_kws) == 1:
                dataset_kws = {
                    dataset: dataset_kws for dataset in well_dataset_info.keys()
                }

            # If dataset_kws is a dict of dicts with depth 2, apply it to all splits
            if (
                isinstance(dataset_kws, dict)
                and all(
                    isinstance(k, str) and isinstance(v, dict)
                    for k, v in dataset_kws.items()
                )
                and get_dict_depth(dataset_kws) == 2
            ):
                dataset_kws = {
                    data_split: dataset_kws
                    for data_split in [
                        "train",
                        "val",
                        "rollout_val",
                        "test",
                        "rollout_test",
                    ]
                }

            # If dataset_kws keys are not a subset of train, val, rollout_val, test, rollout_test, raise an error
            assert set(
                dataset_kws.keys()
            ).issubset(
                set(["train", "val", "rollout_val", "test", "rollout_test"])
            ), f"Expected dataset_kws keys {dataset_kws.keys()} to be a subset of train, val, rollout_val, test, rollout_test."

        # In Val/Test, we want stats for each dataset
        # but we still use MixedWellDataset to handle the extra info (field indices, etc.)
        self.val_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={key: well_dataset_info[key]},
                well_split_name="valid",
                use_normalization=use_normalization,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                transform=transform["val"][key]
                if transform is not None
                and "val" in transform
                and key in transform["val"]
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                field_index_map_override=field_index_map_override,
                dataset_kws=dataset_kws["val"]
                if dataset_kws is not None and "val" in dataset_kws
                else None,
            )
            for key in well_dataset_info
        ]

        self.rollout_val_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={key: well_dataset_info[key]},
                well_split_name="valid",
                use_normalization=use_normalization,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                transform=transform["rollout_val"][key]
                if transform is not None
                and "rollout_val" in transform
                and key in transform["rollout_val"]
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                field_index_map_override=field_index_map_override,
                dataset_kws=dataset_kws["rollout_val"]
                if dataset_kws is not None and "rollout_val" in dataset_kws
                else None,
            )
            for key in well_dataset_info
        ]

        # Train is a single mixed dataset, if not validation mode
        if validation_mode:
            self.train_dataset = self.val_datasets[0]
        else:
            self.train_dataset = MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info=well_dataset_info,
                well_split_name="train",
                use_normalization=use_normalization,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                min_dt_stride=min_dt_stride,
                max_dt_stride=max_dt_stride,
                transform=transform["train"]
                if transform is not None and "train" in transform
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                field_index_map_override=field_index_map_override,
                dataset_kws=dataset_kws["train"]
                if dataset_kws is not None and "train" in dataset_kws
                else None,
            )

        self.test_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={key: well_dataset_info[key]},
                well_split_name="test",
                use_normalization=use_normalization,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                transform=transform["test"][key]
                if transform is not None
                and "test" in transform
                and key in transform["test"]
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                field_index_map_override=field_index_map_override,
                dataset_kws=dataset_kws["test"]
                if dataset_kws is not None and "test" in dataset_kws
                else None,
            )
            for key in well_dataset_info
        ]

        self.rollout_test_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={key: well_dataset_info[key]},
                well_split_name="test",
                use_normalization=use_normalization,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                transform=transform["rollout_test"][key]
                if transform is not None
                and "rollout_test" in transform
                and key in transform["rollout_test"]
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                field_index_map_override=field_index_map_override,
                dataset_kws=dataset_kws["rollout_test"]
                if dataset_kws is not None and "rollout_test" in dataset_kws
                else None,
            )
            for key in well_dataset_info
        ]
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_workers = data_workers
        self.rank = rank
        self.epoch = epoch
        self.max_samples = max_samples

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def train_dataloader(self, rank_override=None) -> DataLoader:
        if False and self.is_distributed:
            base_sampler: type[Sampler] = DistributedSampler
        else:
            base_sampler = RandomSampler

        sampler = BatchedMultisetSampler(
            self.train_dataset,
            base_sampler,
            self.batch_size,  # seed=seed,
            distributed=self.is_distributed,
            max_samples=self.max_samples,  # TODO Fix max_samples later
            rank=self.rank if rank_override is None else rank_override,
        )
        shuffle = sampler is None

        return DataLoader(
            self.train_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=None,
            shuffle=shuffle,
            # drop_last=True,
            sampler=sampler,
            collate_fn=None,
        )

    def build_loaders_from_dset_list(self, dset_list, batch_size=1,
                                     replicas=None, rank=None, 
                                     full=True) -> List[DataLoader]:
        dataloaders = []
        for dataset in dset_list:
            # If distributed, don't replicate across GPUs
            if self.is_distributed:
                # However, for large enough worlds, we need drop_last=False which causes some replication
                sampler: Sampler = BatchSampler(
                    DistributedSampler(dataset, seed=0, drop_last=False, shuffle=not full, # If doing everyhing
                                       num_replicas=replicas, # World size is default if replicas is None otherwise pass size of sync (FSDP) group
                                       rank=rank), # Global rank is default if rank is None - otherwise pass within sync (FSDP) group rank
                    batch_size=batch_size,
                    drop_last=False)
            else:
                sampler = BatchSampler(
                    RandomSampler(
                        dataset, generator=torch.Generator().manual_seed(0)
                    ),
                    batch_size=batch_size,
                    drop_last=False,
                )

            dataloaders.append(
                DataLoader(
                    dataset,
                    num_workers=self.data_workers,
                    pin_memory=True,
                    batch_size=None,
                    shuffle=None,  # Sampler is set
                    sampler=sampler,
                    collate_fn=None,
                )
            )
        return dataloaders

    def val_dataloaders(self,
                         replicas: Optional[int]=None,
                         rank: Optional[int]=None,
                         full: bool = False) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(self.val_datasets,
                                                  self.batch_size,
                                                  replicas,
                                                  rank,
                                                  full)

    def rollout_val_dataloaders(self,
                         replicas: Optional[int]=None,
                         rank: Optional[int]=None,
                         full: bool = False) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(self.rollout_val_datasets,
                                                  1, # Batch size hardcoded to one since 3D data uses so much memory - can be fixed, but not priority
                                                  replicas,
                                                  rank,
                                                  full)

    def test_dataloaders(self) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(self.test_datasets, self.batch_size)

    def rollout_test_dataloaders(self) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(self.rollout_test_datasets, 1)


if __name__ == "__main__":
    well_base_path = "/mnt/home/polymathic/ceph/the_well/"
    data = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "active_matter": {"include_filters": [], "exclude_filters": []},
            "planetswe": {"include_filters": [], "exclude_filters": []},
        },
        batch_size=32,
        data_workers=4,
    )

    for x in data.train_dataloader():
        print(x)
        break
