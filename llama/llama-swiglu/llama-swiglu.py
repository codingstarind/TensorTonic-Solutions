import torch
import torch.nn as nn

def swiglu_ffn(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """
    Apply SwiGLU feed-forward network.
    """
    # Your code here
    swish = nn.SiLU()
    return (swish(x@W_gate.T)*(x@W_up.T))@W_down.T