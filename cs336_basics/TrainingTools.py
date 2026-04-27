import os
import torch
import torch.nn as nn
import numpy as np
from math import sqrt, sin, cos, pi
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum, reduce
from collections.abc import Callable, Iterable
from typing import Optional, BinaryIO, IO

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

                original_dtype = p.dtype

                state = self.state[p] # Get state associated with p.
                m = state.get("m", torch.zeros_like(p, dtype=torch.float32)) # Get first moment from the state, or initial value.
                v = state.get("v", torch.zeros_like(p, dtype=torch.float32)) # Get second moment from the state, or initial value.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                grad = p.grad.detach().to(torch.float32) # Get the gradient of loss with respect to p.
                
                m = beta1 * m + (1 - beta1) * grad # Update m
                v = beta2 * v + (1 - beta2) * grad ** 2 # Update v
                alpha = lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t) # Compute adjusted alpha for this iteration
                update = alpha * m / (torch.sqrt(v) + eps)
                with torch.no_grad():
                    p *= 1 - lr * wd # Apply weight decay
                    p -= update.to(original_dtype) # Update parameters

                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    assert min_learning_rate <= max_learning_rate, "min learning rate must be less than max learning rate"
    assert warmup_iters < cosine_cycle_iters, "cosine cycle must start after warmup cycle is complete"
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + cos(pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))/2

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    norm_sq = 0
    for param in parameters:
        if param.grad is not None:
            norm_sq += torch.sum(param.grad ** 2)
    norm = sqrt(norm_sq)
    scale = max(norm / max_l2_norm, 1)
    if scale != 1:
        for param in parameters:
            if param.grad is not None:
                param.grad /= scale

def data_loading(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    total_length = dataset.shape[0]
    start_index = np.random.randint(0, total_length-context_length, size = (batch_size, 1))
    arr = dataset[start_index + np.arange(context_length+1)]
    return torch.LongTensor(arr[:, :-1], device=device), torch.LongTensor(arr[:, 1:], device=device)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    states = {}
    states["model"] = model.state_dict()
    states["optimizer"] = optimizer.state_dict()
    states["iter"] = iteration
    torch.save(states, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    states = torch.load(src)
    model.load_state_dict(states["model"])
    optimizer.load_state_dict(states["optimizer"])
    return states["iter"]