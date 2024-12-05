"""Utility for computing gradient norm for the embedding layer.

Based on the algorithm from the paper:
https://proceedings.neurips.cc/paper_files/paper/2023/file/a45d344b28179c8da7646bc38ff50ad8-Paper-Conference.pdf.

TensorFlow implementation:
http://google3/third_party/py/tensorflow_privacy/privacy/fast_gradient_clipping/registry_functions/registry_function_utils.py;rcl=564481407
"""
from typing import Dict, List

import torch
from torch import nn


def compute_embedding_norm_sample(
    layer: nn.Embedding,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
  """Computes per sample gradient norms for ``nn.Embedding`` layer.

  Args:
    layer: The Embedding layer to compute the norms for.
    activations: Activations, inputs to the layer on the forward pass. Must
      have shape (batch_size, ...).
    backprops: Backpropagations, gradients of the loss with respect to the
      layer's outputs (NOT the layer's parameter). Must have shape
      (batch_size, ..., embedding_dim).

  Returns:
    A dictionary of norms for each parameter in the layer. Each norm is a tensor
    of shape (batch_size,).

  NOTE: Here is an example input, and the expected intermediate values. This
  is provided to help in understanding the algorithm:
  Inputs:
    layer:  Embedding(3, 1) # (vocab_size, embedding_dim)
    activations:  [tensor([[1, 1],
        [2, 0],
        [2, 0]])]  # (batch_size, ...)
    backprops:  tensor([[[0.2000],[0.2000]],
        [[0.3000],[0.1000]],
        [[0.3000],[0.1000]]])
    backprops.shape:  torch.Size([3, 2, 1])  # (batch_size, ..., embedding_dim)

  Intermediate values:
    input_ids:  tensor([[1, 1],
        [2, 0],
        [2, 0]])
    input_ids.shape:  torch.Size([3, 2])
    grad_values:  tensor([[0.2000],
        [0.2000],
        [0.3000],
        [0.1000],
        [0.3000],
        [0.1000]])
    grad_values.shape:  torch.Size([6, 1])
    nrows:  3
    ncols:  2
    row_indices:  tensor([[0],
        [0],
        [1],
        [1],
        [2],
        [2]])
    flattened_indices:  tensor([[1],
        [1],
        [2],
        [0],
        [2],
        [0]])
    paired_indices:  tensor([[0, 1],
        [0, 1],
        [1, 2],
        [1, 0],
        [2, 2],
        [2, 0]])
    unique_paired_indices:  tensor([[0, 1],
        [1, 0],
        [1, 2],
        [2, 0],
        [2, 2]])
    new_index_positions:  tensor([0, 0, 2, 1, 4, 3])
    num_unique_paired_indices:  5
    summed_gradients:  tensor([[0.4000],
        [0.1000],
        [0.3000],
        [0.1000],
        [0.3000]])
    sqr_gradient_sum:  tensor([0.1600, 0.0100, 0.0900, 0.0100, 0.0900])
    unique_batch_ids:  tensor([0, 1, 1, 2, 2])
    result:  tensor([0.1600, 0.1000, 0.1000])
    result_sqrt:  tensor([0.4000, 0.3162, 0.3162])
  """
  device = activations[0].device
  # input_ids: (batch_size, ...) with cardinality = vocab_size
  # Product of trailing "..."" dimensions is embeddings_count (1 if empty)
  input_ids = activations[0]

  # Reshape input_ids preserving the batch size as the first dimension
  # input_ids: (batch_size, embeddings_count) with cardinality = vocab_size
  input_ids = input_ids.reshape(input_ids.shape[0], -1)

  # Reshape backprops preserving the embedding dimension as the last dimension
  # backprops: (batch_size, ..., embedding_dim)
  # grad_values: (batch_size * embeddings_count, embedding_dim)
  grad_values = backprops.reshape(-1, backprops.size(-1))

  # Presort input_ids
  input_ids, mapping = torch.sort(input_ids, dim=-1)

  # Compute scale for expanding mapping indices
  scale = torch.numel(grad_values) // torch.numel(mapping)

  if scale == 1:
    expanded_mapping = mapping
  else:
    # Create additional indices for the expanded mapping
    additional_indices = torch.arange(
        grad_values.shape[-1], device=device
    ).view(1, 1, -1)

    expanded_mapping = (
        mapping.unsqueeze(-1) * scale + additional_indices
    ).reshape(mapping.shape[0], -1)

  # Reshape grad_values to match the expanded mapping
  reshaped_grad_values = grad_values.reshape(expanded_mapping.shape).to(device)

  # Reorder grad_values and revert to original shape
  grad_values = torch.gather(
      reshaped_grad_values, -1, expanded_mapping
  ).reshape(grad_values.shape)

  # Create 1D tensor of row indices
  nrows = input_ids.size(0)  # batch_size
  ncols = input_ids.size(1)  # embeddings_count
  # row_indices: (batch_size * embeddings_count, 1)
  # Like: [
  #   [0] * embeddings_count,
  #   [1] * embeddings_count,
  #   ...
  #   [batch_size-1] * embeddings_count,
  # ]
  row_indices = torch.repeat_interleave(
      torch.arange(nrows, device=device), ncols
  ).unsqueeze(-1)

  # Pair the input IDs with the row indices
  # flattened_indices: (batch_size * embeddings_count, 1)
  #   with cardinality = vocab_size
  flattened_indices = input_ids.view(-1, 1)
  # paired_indices: (batch_size * embeddings_count, 2)
  #   paired_indices[:, 0] has cardinality = batch_size
  #   paired_indices[:, 1] has cardinality = vocab_size
  paired_indices = torch.cat([row_indices, flattened_indices], dim=1)

  # Get unique paired indices and new index positions for aggregation
  # batch_size <= num_unique_paired_indices <= batch_size * embeddings_count
  # unique_paired_indices: (num_unique_paired_indices, 2)
  #   unique_paired_indices[:, 0] has cardinality = batch_size
  #   unique_paired_indices[:, 1] has cardinality = vocab_size
  # new_index_positions: (batch_size * embeddings_count,)
  #   with cardinality = num_unique_paired_indices
  unique_paired_indices, new_index_positions = torch.unique_consecutive(
      paired_indices, dim=0, return_inverse=True
  )
  num_unique_paired_indices = unique_paired_indices.size(0)

  # Sum gradients over new index positions and compute squared gradient norms
  # summed_gradients: (num_unique_paired_indices, embedding_dim)
  summed_gradients = torch.zeros(
      num_unique_paired_indices, grad_values.size(-1), device=device
  )
  summed_gradients = summed_gradients.index_add(
      0, new_index_positions, grad_values
  )
  # sqr_gradient_sum: (num_unique_paired_indices,)
  sqr_gradient_sum = torch.sum(torch.square(summed_gradients), dim=1)

  # Scatter add the squared sums back to their respective rows
  # result: (batch_size,)
  result = torch.zeros(nrows, device=device)
  unique_batch_ids = unique_paired_indices[:, 0]
  result.scatter_add_(0, unique_batch_ids, sqr_gradient_sum)

  # Compute the square root for the final result (norm)
  result_sqrt = torch.sqrt(result)
  return {layer.weight: result_sqrt}
