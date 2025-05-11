import torch
import torch.nn as nn

class AND(nn.Module):

    def __init__(self, NOT: bool = False, ps: float = 1.0):
        super().__init__()

        self.NOT = NOT
        self.ps = ps

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        condition = (a == 1) | (b == 1)  # | is bitwise OR operator

        if self.NOT:
            x_lin = torch.where(condition, 1 - a * b, 1 - a - b)
        else:
            x_lin = torch.where(condition, a * b, a + b)
        
        return x_lin

class OR(nn.Module):

    def __init__(self, NOT: bool = False, ps: float = 1.0):
        super().__init__()
        self.NOT = NOT
        self.ps = ps

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        condition = (a == 0) | (b == 0)  # | is bitwise OR operator

        if self.NOT:
            x_lin = torch.where(condition, 1 - a - b + a * b, 2 - a - b)
        else:
            x_lin = torch.where(condition, a + b - a * b, a + b - 1)

        return x_lin

class XOR(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        return a + b - 2 * a * b