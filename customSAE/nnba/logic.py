import torch
import torch.nn as nn

class AND(nn.Module):

    def __init__(self, NOT: bool = False, ps: float = 1.0):
        super().__init__()

        self.NOT = NOT
        self.ps = ps  # "pseudo-slope" used in the original implementation

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        if self.NOT:
            x_lin = - a - b + 1.5
        else:
            x_lin = a + b - 1.5

        x = torch.sigmoid(x_lin)

        with torch.no_grad():
            bool_x = (x >= 0.5).float()

        return x + (bool_x - x).detach()

class OR(nn.Module):

    def __init__(self, NOT: bool = False, ps: float = 1.0):
        super().__init__()
        self.NOT = NOT
        self.ps = ps

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        if self.NOT:
            x_lin = - a - b + 0.5
        else:
            x_lin = a + b - 0.5

        x = torch.sigmoid(x_lin)
        with torch.no_grad():
            bool_x = (x >= 0.5).float()

        return x + (bool_x - x).detach()

class XOR(nn.Module):

    def __init__(self):

        super().__init__()

        self.nand_gate = AND(NOT=True)
        self.and_gate = AND()
        self.or_gate = OR()
    
    def forward(self, a, b):

        out_nand = self.nand_gate(a, b)
        out_or = self.or_gate(a, b)

        return self.and_gate(out_or, out_nand)