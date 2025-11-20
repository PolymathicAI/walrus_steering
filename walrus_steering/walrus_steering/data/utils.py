from typing import Dict, Tuple

import torch


def preprocess_batch(
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Given a batch provided by a Dataloader iterating over a WellDataset,
    split the batch as such to provide input and output to the model.

    """
    time_step = batch["output_time_grid"] - batch["input_time_grid"]
    parameters = batch["constant_scalars"]
    x = batch["input_fields"]
    dx = {"x": x, "time": time_step, "parameters": parameters}
    y = batch["output_fields"]
    return dx, y


def get_dict_depth(d: dict) -> int:
    """
    Calculate the minimum depth of a nested dictionary.

    Args:
        d: Input dictionary

    Returns:
        int: Minimum depth of the dictionary
    """
    if not isinstance(d, dict) or not d:
        return 0

    return 1 + min(get_dict_depth(v) for v in d.values())
