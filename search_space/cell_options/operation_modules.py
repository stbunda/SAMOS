""" Operations """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

from torch import Tensor

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'I'

    def forward(self, x1, x2=None):
        return x1

class Zero(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride

    def forward(self, x1, x2=None):
        if self.stride == 1:
            return x1 * 0.

        # re-sizing by stride
        return x1[:, :, ::self.stride, ::self.stride] * 0.

class Sum(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'sum'

    def forward(self, inputs):
        """
        Forward pass for the Sum module.

        Args:
            inputs (list of Tensor): A list of tensors to sum.

        Returns:
            Tensor: The element-wise sum of the input tensors.
        """
        if not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError("Sum expects a non-empty list of tensors as input.")

        # check of the image size
        if inputs[0].size(2) != inputs[1].size(2):  # Check if resizing is needed
            small_in_id, large_in_id = (0, 1) if inputs[0].size(2) < inputs[1].size(2) else (1, 0)

            # Compute the number of times pooling is needed
            small_size = inputs[small_in_id].size(2)
            large_size = inputs[large_in_id].size(2)
            while large_size > small_size:
                inputs[large_in_id] = F.max_pool2d(inputs[large_in_id], kernel_size=2, stride=2)
                large_size = inputs[large_in_id].size(2)  # Update after pooling

            # Ensure final sizes match
            if inputs[large_in_id].size(2) != small_size:
                raise ValueError(f"Sizes do not match after pooling: {inputs[large_in_id].size(2)} vs {small_size}")

        # Ensure channel sizes match
        small_ch_id, large_ch_id = (0, 1) if inputs[0].size(1) < inputs[1].size(1) else (1, 0)
        small_ch = inputs[small_ch_id].size(1)
        large_ch = inputs[large_ch_id].size(1)
        offset = large_ch - small_ch

        if offset != 0:
            padding = torch.zeros(
                (inputs[small_ch_id].size(0), offset, inputs[small_ch_id].size(2), inputs[small_ch_id].size(3)),
                dtype=inputs[small_ch_id].dtype, device=inputs[small_ch_id].device)
            inputs[small_ch_id] = torch.cat([inputs[small_ch_id], padding], dim=1)

        # Perform element-wise addition
        out = torch.add(inputs[0], inputs[1])
        return out

class Concatenate(nn.Module):
    """
    A module that concatenates two input tensors along the channel dimension,
    ensuring their spatial dimensions match through downsampling if necessary.
    """
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, inputs):
        """
        Forward pass to concatenate two tensors with size adjustments if necessary.

        Args:
            input (List): List of two input tensors.

        Returns:
            Tensor: Concatenated tensor along the channel dimension.
        """
        if not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError("Concatenate expects a non-empty list of tensors as input.")

        if inputs[0].size(2) != inputs[1].size(2):  # Check if resizing is needed
            small_in_id, large_in_id = (0, 1) if inputs[0].size(2) < inputs[1].size(2) else (1, 0)

            # Compute the number of times pooling is needed
            small_size = inputs[small_in_id].size(2)
            large_size = inputs[large_in_id].size(2)
            while large_size > small_size:
                inputs[large_in_id] = F.max_pool2d(inputs[large_in_id], kernel_size=2, stride=2)
                large_size = inputs[large_in_id].size(2)  # Update after pooling

            # Ensure final sizes match
            if inputs[large_in_id].size(2) != small_size:
                raise ValueError(f"Sizes do not match after pooling: {inputs[large_in_id].size(2)} vs {small_size}")

        # Concatenate along the channel dimension
        out = torch.cat(inputs, dim=1)
        return out

# class Concatenate(nn.Module):
#     def __init__(self, dim=1):
#         """
#         Initialize the Concatenate module.
#
#         Args:
#             dim (int): The dimension along which to concatenate. Defaults to 1 (channel dimension).
#         """
#         super().__init__()
#         self.name = 'concat'
#         self.dim = dim
#
#     def forward(self, inputs):
#         """
#         Forward pass for the Concatenate module.
#
#         Args:
#             inputs (list of Tensor): A list of two tensors to concatenate.
#
#         Returns:
#             Tensor: The concatenated tensor.
#         """
#         if not isinstance(inputs, list) or len(inputs) != 2:
#             raise ValueError("Concatenate expects a list of exactly two tensors as input.")
#
#         # Get the spatial dimensions of the inputs
#         h1, w1 = inputs[0].shape[2:]  # Height and width of the first tensor
#         h2, w2 = inputs[1].shape[2:]  # Height and width of the second tensor
#
#         # Compute the target dimensions for downsampling
#         target_h = min(h1, h2)
#         target_w = min(w1, w2)
#
#         # Downsample both inputs if needed
#         downsampled_inputs = []
#         for tensor in inputs:
#             h, w = tensor.shape[2:]
#             if h != target_h or w != target_w:
#                 tensor = nn.functional.adaptive_max_pool2d(tensor, (target_h, target_w))
#             downsampled_inputs.append(tensor)
#
#         # Concatenate along the specified dimension
#         return torch.cat(downsampled_inputs, dim=self.dim)