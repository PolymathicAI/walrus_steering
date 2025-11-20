import dataclasses
import itertools
import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, cast

import torch
from hydra.utils import get_method
from the_well.data import WellDataset
from the_well.data.augmentation import RandomRotation90
from the_well.data.datasets import BoundaryCondition, TrajectoryData, TrajectoryMetadata, flatten_field_names
from torch.utils.data import default_collate

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

CARTESIAN_DIMS = ["x", "y", "z"]


class InflatedWellDataset(WellDataset):
    """
    Generic dataset for any Well data. Returns data in B x T x H [x W [x D]] x C format.

    Train/Test/Valid is assumed to occur on a folder level.

    Takes in path to directory of HDF5 files to construct dset.

    Args:
        path:
            Path to directory of HDF5 files, one of path or well_base_path+well_dataset_name
            must be specified
        normalization_path:
            Path to normalization constants - assumed to be in same format as constructed data.
        well_base_path:
            Path to well dataset directory, only used with dataset_name
        well_dataset_name:
            Name of well dataset to load - overrides path if specified
        well_split_name:
            Name of split to load - options are 'train', 'valid', 'test'
        include_filters:
            Only include files whose name contains at least one of these strings
        exclude_filters:
            Exclude any files whose name contains at least one of these strings
        use_normalization:
            Whether to normalize data in the dataset
        n_steps_input:
            Number of steps to include in each sample
        n_steps_output:
            Number of steps to include in y
        min_dt_stride:
            Minimum stride between samples
        max_dt_stride:
            Maximum stride between samples
        flatten_tensors:
            Whether to flatten tensor valued field into channels
        cache_small:
            Whether to cache small tensors in memory for faster access
        max_cache_size:
            Maximum numel of constant tensor to cache
        return_grid:
            Whether to return grid coordinates
        boundary_return_type: options=['padding', 'mask', 'exact', 'none']
            How to return boundary conditions. Currently only padding supported.
        full_trajectory_mode:
            Overrides to return full trajectory starting from t0 instead of samples
                for long run validation.
        name_override:
            Override name of dataset (used for more precise logging)
        transform:
            Transform to apply to data. In the form `f(data: TrajectoryData, metadata:
            TrajectoryMetadata) -> TrajectoryData`, where `data` contains a piece of
            trajectory (fields, scalars, BCs, ...) and `metadata` contains additional
            informations, including the dataset itself.
        field_transforms:
            Deterministic transform to apply to specific fields.
        min_std:
            Minimum standard deviation for field normalization. If a field standard
            deviation is lower than this value, it is replaced by this value.
        storage_options :
            Option for the ffspec storage.
        pad_cartesian_data_to_d:
            Pad the data to at least this dimension by including additional singleton dims and zero-padding tensor-values fields.
    """

    def __init__(
        self,
        *args,
        # New parameters
        pad_cartesian_data_to_d: int = 3,
        field_transforms: Optional[Dict[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pad_cartesian_data_to_d = pad_cartesian_data_to_d
        self.original_metadata = self.metadata
        self.metadata = self.pad_metadata()
        self.n_spatial_dims = self.metadata.n_spatial_dims
        self.field_transforms = {k: get_method(v) for k, v in field_transforms.items()}

    def _pad_cartesian_field_names(
        self, field_names: Dict[int, List[str]], padded_d: int
    ):
        """Repeats data over axes not used in storage"""
        # Look at which dimensions currently are not used and tile based on their sizes
        out_dict = dict()

        for order, order_dict in field_names.items():
            if order == 0:
                out_dict[order] = order_dict
            else:
                out_dict[order] = []
                used_field_names = set()
                # Create new extensions as product of cartesian dims
                ti_field_dims = [
                    "".join(xyz)
                    for xyz in itertools.product(
                        CARTESIAN_DIMS[:padded_d],
                        repeat=order,
                    )
                ]
                for field in order_dict:
                    # Get name by truncating text after last "_"
                    core_name = field.rsplit("_", 1)[0]
                    # If first example, create fields up to padded_d
                    if core_name not in used_field_names:
                        out_dict[order].extend(
                            [f"{core_name}_{dims}" for dims in ti_field_dims]
                        )
                    used_field_names.add(core_name)

        return out_dict

    def pad_metadata(self):
        # If cartesian, then pad to higher D if necessary
        if (
            self.original_metadata.grid_type == "cartesian"
            and self.pad_cartesian_data_to_d > self.original_metadata.n_spatial_dims
        ):
            if not hasattr(self, "modded_metadata"):
                pad_level = (
                    self.pad_cartesian_data_to_d - self.original_metadata.n_spatial_dims
                )
                padded_variable_field_names = self._pad_cartesian_field_names(
                    self.original_metadata.field_names, self.pad_cartesian_data_to_d
                )
                padded_constant_field_names = self._pad_cartesian_field_names(
                    self.original_metadata.constant_field_names,
                    self.pad_cartesian_data_to_d,
                )
                padded_boundary_conditions = (
                    self.original_metadata.boundary_condition_types
                    + ["PERIODIC"] * pad_level
                )  # Its a singleton so periodic is technically accurate
                padded_metadata = dataclasses.replace(
                    self.original_metadata,
                    spatial_resolution=self.original_metadata.spatial_resolution
                    + (1,) * pad_level,
                    n_spatial_dims=self.pad_cartesian_data_to_d,
                    field_names=padded_variable_field_names,
                    constant_field_names=padded_constant_field_names,
                    boundary_condition_types=padded_boundary_conditions,
                )
            return padded_metadata
        else:
            return self.original_metadata

    def _pad_extra_spatial_dimension(
        self, data_tensor: torch.Tensor, tensor_order: int, d: int
    ):
        """Pad data tensor to higher spatial dimensions by unsqueezing new dimensions"""
        # Pad fields
        pad_d = d - self.original_metadata.n_spatial_dims
        for dim in range(pad_d):
            data_tensor = data_tensor.unsqueeze(-tensor_order - 1)
        return data_tensor

    def _pad_extra_tensor_order(
        self, data_tensor: torch.Tensor, tensor_order: int, d: int
    ):
        """Pad tensor fields to higher dimensions by adding extra 0 value tensor entries"""
        # Pad fields
        pad_d = d - self.original_metadata.n_spatial_dims
        # These are zeros so just use built-in pad
        data_tensor = torch.nn.functional.pad(
            data_tensor,
            (
                0,
                pad_d,
            )
            * tensor_order,
        )
        return data_tensor

    def _pad_to_higher_d(self, data: TrajectoryData):
        """Pad data to higher dimensions. This currently only applies to euclidean/cartesian data
        as dimensions are not homogeneous for non-euclidean data"""
        orig_spatial_dims = self.original_metadata.n_spatial_dims
        for key in ("variable_fields", "constant_fields"):
            for order, fields in data[key].items():
                for field_name in fields:
                    # First unsqueeze the extra dimension
                    fields[field_name] = self._pad_extra_spatial_dimension(
                        fields[field_name], order, self.pad_cartesian_data_to_d
                    )
                    # Next pad the tensor dimensions if cartesian (not swapping axes otherwise)
                    fields[field_name] = self._pad_extra_tensor_order(
                        fields[field_name], order, self.pad_cartesian_data_to_d
                    )
        if "space_grid" in data:
            data["space_grid"] = self._pad_extra_spatial_dimension(
                data["space_grid"], 1, self.pad_cartesian_data_to_d
            )
            data["space_grid"] = self._pad_extra_tensor_order(
                data["space_grid"], 1, self.pad_cartesian_data_to_d
            )
        if "boundary_conditions" in data:
            # Periodic can be used for singleton dim
            # print("this happening?")
            if self.pad_cartesian_data_to_d > orig_spatial_dims:
                # print("how about this?!?!", orig_spatial_dims, data["boundary_conditions"].shape)
                data["boundary_conditions"][orig_spatial_dims:] = BoundaryCondition["PERIODIC"].value 
        return data

    def _preprocess_data(
        self, data: TrajectoryData, traj_metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        """Preprocess the data before applying transformations."""
        if self.metadata.grid_type == "cartesian":
            data = self._pad_to_higher_d(data)
        for key in [
            "variable_fields",
            "constant_fields",
        ]:
            for order, fields in data[key].items():
                for field_name in fields:
                    if field_name in self.field_transforms:
                        fields[field_name] = self.field_transforms[field_name](
                            fields[field_name]
                        )
        return data


class BatchInflatedWellDataset(InflatedWellDataset):
    def __init__(self, *args, batch_size: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs["well_split_name"] == "train":
            self.transform = RandomRotation90()
        self.batch_size = batch_size
        # VMap new functions over batch dim (0)
        self._preprocess_data = torch.vmap(
            self._preprocess_data, in_dims=(0, None), randomness="same"
        )
        if self.transform is not None:
            self.transform = torch.vmap(
                self.transform, in_dims=(0, None), randomness="same"
            )
        self._postprocess_data = torch.vmap(
            self._postprocess_data, in_dims=(0, None), randomness="same"
        )
        self._construct_sample = torch.vmap(
            self._construct_sample, in_dims=(0, None), randomness="same"
        )

    def get_field_padding_masks(self, data: Dict):
        """ We don't want to evaluate the loss on the padded fields 
        so provide a mask to ignore them. """
        # TODO - separate these out and merge in the processor
        n_dim = self.metadata.n_spatial_dims
        padding_happened = n_dim > self.original_metadata.n_spatial_dims
        spatial_dims = data["input_fields"].shape[2:-1] # including batch so B x T x [Space] x C
        masks = []
        used_fields = set()
        for order, fields in self.metadata.field_names.items():
            for field in fields:
                base_field = field.rsplit("_", 1)[0]
                if order == 0:
                    masks.append(True)
                else:                        
                    field_name, coords = field.rsplit("_", 1)
                    if field_name in used_fields:
                        continue
                    used_fields.add(field_name)
                    if order == 1:
                        for dim_i in range(n_dim):
                            if spatial_dims[dim_i] == 1 and padding_happened:
                                masks.append(False)
                            else:
                                masks.append(True)
                    elif order == 2:
                        for dim_i in range(n_dim):
                            for dim_j in range(n_dim):
                                if (spatial_dims[dim_i] == 1 or spatial_dims[dim_j] == 1) and padding_happened:
                                    masks.append(False)
                                else:
                                    masks.append(True)
        return torch.tensor(masks)
    
    def __getitem__(self, idxes: int | List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the idx-th item in the dataset."""
        if isinstance(idxes, int):
            idxes = [idxes]
        data = []
        file_idxes = []
        sample_idxes = []
        time_idxes = []
        dts = []
        for i in idxes:
            datum, file_idx, sample_idx, time_idx, dt = self._load_one_sample(i)
            data.append(datum)
            file_idxes.append(file_idx)
            sample_idxes.append(sample_idx)
            time_idxes.append(time_idx)
            dts.append(dt)
        data = default_collate(
            data
        )  # {k: torch.stack([d[k] for d in data]) for k in data[0]}
        data = cast(TrajectoryData, data)
        traj_metadata = TrajectoryMetadata(
            dataset=self,
            file_idx=file_idx,
            sample_idx=sample_idx,
            time_idx=time_idx,
            time_stride=dt,
        )
        # Apply any type of pre-processing that needs to be applied before augmentation
        data = self._preprocess_data(data, traj_metadata)
        # Apply augmentations and other transformations
        if self.transform is not None:
            data = self.transform(data, traj_metadata)
        # Convert ingestable format - in this class this flattens the fields
        data = self._postprocess_data(data, traj_metadata)
        # Break apart into x, y
        sample = self._construct_sample(data, traj_metadata)
        # TODO - fast hack for end
        field_mask = self.get_field_padding_masks(sample)
        sample["field_mask"] = field_mask

        # Temporary hack to remove density from acoustic scattering since it means something different
        if "acoustic_scattering" in self.metadata.dataset_name:
            sample["constant_fields"][..., 0] = 0.

        # Add mask which indicates which fields are true vs padded
        

        # Return only non-empty keys - maybe change this later
        return sample
