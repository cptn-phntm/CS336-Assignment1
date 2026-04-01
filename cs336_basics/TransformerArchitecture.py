import torch
import torch.nn as nn
from math import sqrt, sin, cos
from einops import rearrange, einsum, reduce


class Linear(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
        **kwargs,
    ) -> None:
        """
        in_features (int): final dimension of the input
        out_features (int): final dimension of the output
        device (torch.device | None): Device to store the parameters on
        dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        init_std = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weights,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            self.weights, x,
            "d_out d_in, ... d_in -> ... d_out"
        )


class Embedding(nn.Module):
    def __init__(self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
        **kwargs,
    ) -> None:
        """
        num_embeddings (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding vectors, i.e., dmodel
        device (torch.device | None): Device to store the parameters on
        dtype (torch.dtype | None):  Data type of the parameters
        """
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embeddings, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        """
        d_model (int): Hidden dimension of the model
        eps (float): Epsilon value for numerical stability
        device (torch.device | None): Device to store the parameters on
        dtype (torch.dtype | None): Data type of the parameter
        """
        super().__init__()
        self.d_model = d_model
        self.gain = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = torch.pow(x, 2)
        x_MS = reduce(x_squared, "... d_model -> ...", "mean")
        x_MS += self.eps
        x_RMS = torch.sqrt(x_MS)
        x_RMS = rearrange(x_RMS, "... -> ... 1")
        x_norm = x / x_RMS
        x_gain = x_norm * self.gain
        return x_gain.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 64 * int(d_model * 8/3 / 64)
        
        self.w1_weights = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2_weights = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3_weights = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

        init_std = sqrt(2 / (d_model + d_ff))

        nn.init.trunc_normal_(
            self.w1_weights,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )

        nn.init.trunc_normal_(
            self.w2_weights,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )

        nn.init.trunc_normal_(
            self.w3_weights,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_prod = einsum(
            self.w1_weights, x,
            "d_out d_in, ... d_in -> ... d_out"
        )
        w3_prod = einsum(
            self.w3_weights, x,
            "d_out d_in, ... d_in -> ... d_out"
        )
        silu_w1 = w1_prod * torch.sigmoid(w1_prod)
        swiglu = silu_w1 * w3_prod
        return einsum(
            self.w2_weights, swiglu,
            "d_out d_in, ... d_in -> ... d_out"
        )


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, 
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None
    ) -> None:
        """
        theta (float): Θ value for the RoPE
        d_k (int): dimension of query and key vectors
        max_seq_len (int): Maximum sequence length that will be inputted
        device (torch.device | None = None): Device to store the buffer on
        """
        super().__init__()
        assert d_k % 2 == 0

        thetas = torch.empty((max_seq_len, d_k//2), device=device, requires_grad=False)
        for k in range(d_k // 2):
            for i in range(max_seq_len):
                thetas[i,k] = i / (theta ** ((2 * k) / d_k))
        
        coses = torch.cos(thetas)
        sines = torch.sin(thetas)
        self.register_buffer("coses", coses)
        self.register_buffer("sines", sines)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        """
        x_paired = rearrange(
            x,
            "... (pair two) -> ... pair two",
            two=2
        ) # shape (..., seq_len, d_k // 2, 2)

        pos_coses = self.coses[token_positions] # shape (seq_len, d_k // 2)
        pos_sines = self.sines[token_positions] # shape (seq_len, d_k // 2)
        pos_rot1 = torch.stack((pos_coses, -pos_sines), axis = -1)
        pos_rot2 = torch.stack((pos_sines, pos_coses), axis = -1)
        pos_rot = torch.stack((pos_rot1, pos_rot2), axis = -2) # shape (seq_len, d_k // 2, 2, 2)
        x_rot = einsum(
            x_paired, pos_rot,
            "... d_in, ... d_out d_in -> ... d_out"
        )
        x_rot = rearrange(
            x_rot,
            "... pair two -> ... (pair two)"
        )
        return x_rot


