import torch
import torch.nn as nn
from math import sqrt, sin, cos
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum, reduce
from collections.abc import Callable, Iterable
from typing import Optional

def cross_entropy(
    inputs: Float[torch.Tensor, "... batch_size vocab_size"],
    targets: Int[torch.Tensor, "... batch_size"],
    ) -> Float[torch.Tensor, ""]:
    """
    Compute the cross-entropy loss between the inputs and targets.
    Args:
        inputs: A tensor of shape (batch_size, vocab_size) containing the predicted probabilities for each class.
        targets: A tensor of shape (batch_size,) containing the true class labels.
    Returns:
        A scalar tensor containing the cross-entropy loss.
    """
    max_inputs = torch.amax(inputs, dim=-1, keepdim=True)
    inputs = inputs - max_inputs
    exp_inputs = torch.exp(inputs)
    sum_inputs = torch.sum(exp_inputs, dim=-1, keepdim=False)
    lse_inputs = torch.log(sum_inputs)

    target_inputs = torch.gather(inputs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(lse_inputs - target_inputs)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# for lr in [1e1, 1e2, 1e3]:
#     opt = SGD([weights], lr=lr)
#     print(f"Learning rate: {lr} \n")
#     for t in range(10):
#         opt.zero_grad() # Reset the gradients for all learnable parameters.
#         loss = (weights**2).mean() # Compute a scalar loss value.
#         print(loss.cpu().item())
#         loss.backward() # Run backward pass, which computes gradients.
#         opt.step() # Run optimizer step.

class AdamW(torch.optim.Optimizer):
    def __init__(self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        betas=(0.9,0.999),
        eps=1e-8
        ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                m = state.get("m", torch.zeros_like(p)) # Get first moment from the state, or initial value.
                v = state.get("v", torch.zeros_like(p)) # Get second moment from the state, or initial value.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                m = beta1 * m + (1 - beta1) * grad # Update m
                v = beta2 * v + (1 - beta2) * grad ** 2 # Update v
                alpha = lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t) # Compute adjusted alpha for this iteration
                p.data -= alpha * m / (torch.sqrt(v) + eps) # Update parameters
                p.data -= lr * wd * p.data

                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss