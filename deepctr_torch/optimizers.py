# -*- coding:utf-8 -*-
"""Optimizers with explicit cross-framework numerical semantics."""

import math

import torch
from torch.optim import Optimizer


class TensorFlowAdam(Optimizer):
    """Adam using the epsilon and bias-correction order used by Keras 2.

    ``named_parameters`` is used to preserve sparse embedding-row semantics:
    rows absent from a mini-batch do not receive momentum-only updates. This
    matches TensorFlow's ``IndexedSlices`` path while keeping the existing
    dense DeepCTR-Torch embeddings unchanged.
    """

    def __init__(self, named_parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-7):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameters: {}".format(betas))
        groups = []
        for name, parameter in list(named_parameters):
            groups.append({
                "params": [parameter],
                "parameter_name": name,
                "sparse_rows": "embedding_dict" in name and parameter.ndim == 2,
            })
        if not groups:
            raise ValueError("optimizer got an empty parameter list")
        super(TensorFlowAdam, self).__init__(
            groups, dict(lr=lr, betas=betas, eps=eps)
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("TensorFlowAdam expects dense PyTorch gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                first = state["exp_avg"]
                second = state["exp_avg_sq"]
                step_size = (
                    group["lr"]
                    * math.sqrt(1.0 - beta2 ** step)
                    / (1.0 - beta1 ** step)
                )
                if group["sparse_rows"]:
                    row_axes = tuple(range(1, gradient.ndim))
                    rows = torch.nonzero(
                        gradient.abs().sum(dim=row_axes) > 0,
                        as_tuple=False,
                    ).flatten()
                    if rows.numel() == 0:
                        continue
                    values = gradient.index_select(0, rows)
                    first_rows = first.index_select(0, rows)
                    second_rows = second.index_select(0, rows)
                    first_rows.mul_(beta1).add_(values, alpha=1.0 - beta1)
                    second_rows.mul_(beta2).addcmul_(
                        values, values, value=1.0 - beta2
                    )
                    first.index_copy_(0, rows, first_rows)
                    second.index_copy_(0, rows, second_rows)
                    update = step_size * first_rows / (
                        second_rows.sqrt() + group["eps"]
                    )
                    parameter.index_add_(0, rows, -update)
                else:
                    first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    second.mul_(beta2).addcmul_(
                        gradient, gradient, value=1.0 - beta2
                    )
                    parameter.addcdiv_(
                        first,
                        second.sqrt().add_(group["eps"]),
                        value=-step_size,
                    )
        return loss
