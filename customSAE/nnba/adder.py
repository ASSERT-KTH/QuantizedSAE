import torch
import torch.nn as nn
from .logic import *
# from logic import *

class residual_adder(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits

        self.optimized_binary_adder = OptimizedBinaryAdder(n_bits)

    def forward(self, x, p=None, true_sum=None):

        len_x = x.shape[-2]

        powers = 2 ** torch.arange(self.n_bits, device=x.device)
        powers[-1] *= -1

        if len_x < 2:
            raise ValueError("Input x must have length at least 2")
        else:
            # Calculate the residual of the sum w.r.t. the input bits
            with torch.no_grad():
                x_sum = (x.sum(dim=-2) * powers).sum(dim=-1)
                x_int = (x * powers).sum(dim=-1)
                # x_residual = (((x_sum.unsqueeze(-1) - x_int).int().unsqueeze(-1) & powers.int()) > 0).float()
                x_residual = (x_sum.unsqueeze(-1) - x_int).unsqueeze(-1)

            return self.optimized_binary_adder(x.reshape(-1, x.shape[-1]), x_residual.reshape(-1, x_residual.shape[-1]), p.reshape(-1, p.shape[-1]), true_sum.reshape(-1, true_sum.shape[-1]))

class OptimizedBinaryAdderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, p, true_sum):
        # mask shpe: [batch, latent_dim, 1]
        # a, b shape: [batch, latent_dim * n_neurons, n_bits]
        N, n_bits = a.shape
        
        ctx.save_for_backward(a, b, p, true_sum)
        ctx.n_bits = n_bits
        
        # Create trigger tensor that's connected to the computation graph
        # Sum a small portion of the input to maintain gradient connection
        trigger = a.sum() * 0.0 + 1.0

        return trigger
    
    @staticmethod
    def backward(ctx, trigger):

        a, b, p, true_sum = ctx.saved_tensors
        n_bits = ctx.n_bits

        binary_weights = 2 ** torch.arange(n_bits, device=a.device)
        binary_weights[-1] *= -1
        
        residual = (true_sum * binary_weights).sum(dim=-1).unsqueeze(-1) - b

        carry = residual // 2 ** (n_bits-1)

        residual_remainder = residual - carry * 2 ** (n_bits-1)
        residual_bits = (residual_remainder.int() & binary_weights[:-1].int() > 0).float()

        residual_carry = carry.int()
        mask = (-residual_carry) == a[:, -1].unsqueeze(-1)

        grad_a = torch.zeros_like(a)
        grad_a[:, -1] = 1 - 2 * (residual_carry.squeeze(-1) < 0).float()
        mask = mask.expand(a[:, :-1].shape)
        alt_grad = (1 - 2 * (residual_carry.squeeze(-1) < 0).float()).unsqueeze(-1).expand(a[:, :-1].shape)
        grad_a[:, :-1] = torch.where(mask, 2 * residual_bits - 1, alt_grad)
        grad_a *= - p * (1 - p) * binary_weights

        return grad_a, None, None, None

class OptimizedBinaryAdder(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits
    
    def forward(self, a, b, p, true_sum):

        return OptimizedBinaryAdderFunction.apply(a, b, p, true_sum)