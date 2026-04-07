import torch
import torch.nn as nn
from math import sqrt, sin, cos
from jaxtyping import Bool, Float, Int
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
        self.register_buffer("coses", coses, persistent=False)
        self.register_buffer("sines", sines, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        """
        x_paired = rearrange(
            x,
            "... (pair two) -> ... pair two",
            two=2
        ) # shape (..., seq_len, d_k // 2, 2)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device).expand(*x.shape[:-1])
        
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


def softmax(x: torch.Tensor, coord: int):
    assert coord < len(x.shape)
    max_x = torch.amax(x, dim=coord, keepdim=True)
    x = x - max_x
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=coord, keepdim=True)
    return exp_x / sum_x

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    assert Q.shape[-1] == K.shape[-1], "query dim must equal key dim"
    assert (Q.shape[-2] == mask.shape[-2]) & (K.shape[-2] == mask.shape[-1]), "incorrect mask shape"
    assert V.shape[-2] == K.shape[-2], "key and value sequence lengths should match"

    d_k = K.shape[-1]
    temp = einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    ) / sqrt(d_k)
    temp_masked = temp.masked_fill(~mask, -torch.inf)
    return einsum(
        softmax(temp_masked, -1), V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )


class MultiheadSelfAttention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "number of heads must divide model dim"
        self.W_Q = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_K = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_V = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_O = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        init_std = sqrt(1 / d_model)
        nn.init.trunc_normal_(
            self.W_Q,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )
        nn.init.trunc_normal_(
            self.W_K,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )
        nn.init.trunc_normal_(
            self.W_V,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )
        nn.init.trunc_normal_(
            self.W_O,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )

        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        queries = einsum(
            self.W_Q, x,
            "hd_k d_model, ... seq_len d_model -> ... seq_len hd_k"
        )
        keys = einsum(
            self.W_K, x,
            "hd_k d_model, ... seq_len d_model -> ... seq_len hd_k"
        )
        values = einsum(
            self.W_V, x,
            "hd_v d_model, ... seq_len d_model -> ... seq_len hd_v"
        )
        queries = rearrange(
            queries,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads
        )
        keys = rearrange(
            keys,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads
        )
        values = rearrange(
            values,
            "... seq_len (h d_v) -> ... h seq_len d_v",
            h = self.num_heads
        )

        seq_len = x.shape[-2]
        row = torch.arange(seq_len)[None, :]
        col = torch.arange(seq_len)[:, None]
        mask = row <= col

        attention = scaled_dot_product_attention(
            queries,
            keys,
            values,
            mask
        )
        attention = rearrange(
            attention,
            "... h seq_len d_v -> ... seq_len (h d_v)"
        )
        return einsum(
            self.W_O, attention,
            "d_model hd_v, ... hd_v -> ... d_model"
        )


class MultiheadSelfAttentionWithRope(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "number of heads must divide model dim"
        self.W_Q = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_K = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_V = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_O = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        init_std = sqrt(1 / d_model)
        nn.init.trunc_normal_(
            self.W_Q,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )
        nn.init.trunc_normal_(
            self.W_K,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )
        nn.init.trunc_normal_(
            self.W_V,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )
        nn.init.trunc_normal_(
            self.W_O,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = RotaryPositionalEmbedding(
            theta, d_model // num_heads, max_seq_len, device
        )

    def forward(self,
        x: torch.Tensor,
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        queries = einsum(
            self.W_Q, x,
            "hd_k d_model, ... seq_len d_model -> ... seq_len hd_k"
        )
        keys = einsum(
            self.W_K, x,
            "hd_k d_model, ... seq_len d_model -> ... seq_len hd_k"
        )
        values = einsum(
            self.W_V, x,
            "hd_v d_model, ... seq_len d_model -> ... seq_len hd_v"
        )
        queries = rearrange(
            queries,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads
        )
        keys = rearrange(
            keys,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads
        )
        values = rearrange(
            values,
            "... seq_len (h d_v) -> ... h seq_len d_v",
            h = self.num_heads
        )

        queries = self.rope.forward(queries, token_positions)
        keys = self.rope.forward(keys, token_positions)

        seq_len = x.shape[-2]
        row = torch.arange(seq_len)[None, :]
        col = torch.arange(seq_len)[:, None]
        mask = row <= col

        attention = scaled_dot_product_attention(
            queries,
            keys,
            values,
            mask
        )
        attention = rearrange(
            attention,
            "... h seq_len d_v -> ... seq_len (h d_v)"
        )
        return einsum(
            self.W_O, attention,
            "d_model hd_v, ... hd_v -> ... d_model"
        )


class TransformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: int = 10000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttentionWithRope(
            d_model,
            num_heads,
            theta,
            max_seq_len,
            device=device,
            dtype=dtype
        )
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, in_features: Float[torch.Tensor, "batch sequence_length d_model"]):
        step1 = self.norm1.forward(in_features)
        step2 = self.attn.forward(step1)
        temp = in_features + step2
        step3 = self.norm2.forward(temp)
        step4 = self.ffn.forward(step3)
        return temp + step4

class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.emb = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        self.transformer_blocks = []
        for _ in range(num_layers):
            transformer_block = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            self.transformer_blocks.append(transformer_block)
        
        self.final_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.final_linear = Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, x: torch.Tensor):
        x = self.emb.forward(x)
        for block in self.transformer_blocks:
            x = block.forward(x)
        x = self.final_norm(x)
        x = self.final_linear(x)
        return x

