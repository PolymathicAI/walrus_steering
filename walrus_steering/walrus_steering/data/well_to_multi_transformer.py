from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from einops import rearrange, repeat


class AbstractFormatter(ABC):
    """
    Default preprocessor for Well to MPP data.
    """

    @abstractmethod
    def process_input(
        self, data: Dict, causal_in_time: bool = False, predict_delta: bool = False
    ) -> Tuple:
        pass

    @abstractmethod
    def process_output(self, output, metadata) -> torch.Tensor:
        pass


class ChannelsFirstWithTimeFormatter(AbstractFormatter):
    """
    Default preprocessor for data in channels first format.
    """

    def process_input(
        self,
        data: Dict,
        causal_in_time: bool = False,
        predict_delta: bool = False,
        train: bool = True,
    ):
        """Convert data from Well format to model format.

        Model format for MPPX is channels first (t, b, c, ...) where t is the time dimension, b is the batch dimension, and c is the channel dimension.

        During training, y is the loss target. During validation, it is always the raw ground truth.
        """
        x = data["input_fields"]
        x = rearrange(x, "b t ... c -> t b c ...")
        if "constant_fields" in data:
            flat_constants = repeat(
                data["constant_fields"],
                "b ... c -> (repeat) b c ...",
                repeat=x.shape[0],
            )
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=2,  # Different from the well due to time
            )
        y = data["output_fields"]
        if train:
            if causal_in_time:
                if predict_delta:
                    y = torch.cat([data["input_fields"], y], dim=1)
                    y = y[:, 1:, ...] - y[:, :-1, ...]
                else:
                    y = torch.cat([data["input_fields"][:, 1:, ...], y], dim=1)
            else:
                # For non-causal predict delta, we only need to append the last step
                # For the sizes we're doing, this could be merged with above, but could
                # be unnecessarily expensive at higher res/content lengths
                if predict_delta:
                    y = torch.cat([data["input_fields"][:, -1:, ...], y], dim=1)
                    y = y[:, 1:, ...] - y[:, :-1, ...]
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (
            x,
            data["field_indices"],
            data["boundary_conditions"],
        ), y

    def process_output(self, output, metadata):
        return rearrange(output, "t b c ... -> b t ... c")
