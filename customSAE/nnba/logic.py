import torch
import torch.nn as nn

class AND(nn.Module):

    def __init__(self, NOT: bool = False, ps: float = 1.0):
        super().__init__()

        self.NOT = NOT
        self.ps = ps
        self.eps = 1e-6  # Small epsilon for floating point comparison

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # Use approximate comparison with tolerance for floating point values
        a_is_one = torch.abs(a - 1.0) < self.eps
        b_is_one = torch.abs(b - 1.0) < self.eps
        condition = a_is_one | b_is_one  # | is bitwise OR operator

        if self.NOT:
            x_lin = torch.where(condition, 1 - a * b, 1 - a - b)
        else:
            x_lin = torch.where(condition, a * b, a + b)
        
        # Use straight-through estimator instead of round
        # This keeps x_lin in the computational graph for gradients
        # while forcing the forward value to be binary
        x_binary = torch.round(x_lin)
        return x_lin + (x_binary - x_lin).detach()

class OR(nn.Module):

    def __init__(self, NOT: bool = False, ps: float = 1.0):
        super().__init__()
        self.NOT = NOT
        self.ps = ps
        self.eps = 1e-6  # Small epsilon for floating point comparison

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # Use approximate comparison with tolerance for floating point values
        a_is_zero = torch.abs(a) < self.eps
        b_is_zero = torch.abs(b) < self.eps
        condition = a_is_zero | b_is_zero  # | is bitwise OR operator

        if self.NOT:
            x_lin = torch.where(condition, 1 - a - b + a * b, 2 - a - b)
        else:
            x_lin = torch.where(condition, a + b - a * b, a + b - 1)

        # Use straight-through estimator instead of round
        x_binary = torch.round(x_lin)
        return x_lin + (x_binary - x_lin).detach()

class XOR(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-6  # Small epsilon for floating point comparison

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        x_lin = a + b - 2 * a * b
        
        # Use straight-through estimator instead of round
        x_binary = torch.round(x_lin)
        return x_lin + (x_binary - x_lin).detach()