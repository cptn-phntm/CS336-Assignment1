import random
import numpy as np
import torch
import wandb
from dataclasses import dataclass
from cs336_basics.BytePairEncoding import *
from cs336_basics.TransformerArchitecture import TransformerLM
from cs336_basics.TrainingTools import *

@dataclass
class TrainConfig:
    # input config
    training_tokenized_text_input_path: str
    validation_tokenized_text_input_path: str
    token_datatype: np.dtype = np.uint16

    # model config
    seed: int = 42
    vocab_size: int = 10000
    context_length: int = 256
    hidden_dims: int = 512
    ff_dims: int = 1344
    num_attn_heads: int = 16
    num_layers: int = 4
    rope_theta: float = 10000.0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16

    # training config
    num_iterations: int = 10000
    batch_size: int = 64
    max_l2_norm: float = 1.0

    # cosine cycle learning rate config
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-3
    warmup_iters: int = 100
    cosine_cycle_iters: int = 1000

    # optimizer config
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    # checkpointing config
    save_checkpoint_every: int = 1000
    compute_val_metrics_every: int = 100
    checkpoint_path: str = "./model/checkpoint.pt"

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(
    config: TrainConfig,
) -> None:
    # Step 0: init wandb, set random seed
    wandb.init(project="cs336-assignment1", config=config.__dict__)
    set_seed(config.seed)

    # Step 1: initialize model, optimizer, validation data
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.hidden_dims,
        d_ff=config.ff_dims,
        num_heads=config.num_attn_heads,
        num_layers=config.num_layers,
        rope_theta=config.rope_theta,
        device=config.device,
        dtype=config.dtype
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    input_data = np.memmap(
                config.training_tokenized_text_input_path,
                dtype=config.token_datatype,
                mode="r"
            )
    val_data = np.memmap(
                config.validation_tokenized_text_input_path,
                dtype=config.token_datatype,
                mode="r"
            )
    val_inputs, val_targets = data_loading(
        val_data,
        batch_size=config.batch_size,
        context_length=config.context_length,
        device=config.device
    )

    for iteration in range(1, config.num_iterations + 1):
        # Step 2: get training data
        sampled_input, sampled_targets = data_loading(
            input_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device
        )

        # Step 3: run forward and backward pass, and optimizer step
        # zero out gradients from previous iteration
        optimizer.zero_grad()

        # implement learning rate schedule
        learning_rate = learning_rate_schedule(
            iteration,
            min_learning_rate=config.min_learning_rate,
            max_learning_rate=config.max_learning_rate,
            warmup_iters=config.warmup_iters,
            cosine_cycle_iters=config.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        # forward pass
        next_output_probs = model.forward(sampled_input)
        loss = cross_entropy(next_output_probs, sampled_targets)

        # backward pass and optimizer step
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=config.max_l2_norm)
        optimizer.step()

        # Step 4: checkpointing
        if iteration % config.save_checkpoint_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                out=config.checkpoint_path,
            )
        
        # Step 5: calculate metrics and log to wandb
        wandb.log({
            "training_loss": loss.item(),
        }, 
        step=iteration
        )
        if iteration % config.compute_val_metrics_every == 0:
            with torch.no_grad():
                val_next_output_probs = model.forward(val_inputs)
                val_loss = cross_entropy(val_next_output_probs, val_targets)
                wandb.log({
                    "validation_loss": val_loss.item(),
                }, 
                step=iteration
                )
    wandb.finish()
            


        
        
