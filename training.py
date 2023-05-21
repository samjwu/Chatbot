"""
Utility functions for training an attention model.
"""

import torch


def calculate_negative_log_likelihood_loss(
    input_vector: Tensor, target: Tensor, mask: Tensor
) -> tuple[Tensor, float]:
    nTotal = mask.sum()
    crossEntropy = -torch.log(
        torch.gather(input_vector, 1, target.view(-1, 1)).squeeze(1)
    )
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
