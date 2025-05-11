import torch
import torch.nn as nn
from .logic import *
# from logic import *

class half_adder(nn.Module):

    def __init__(self):
        super().__init__()

        self.xor_gate = XOR()
        self.and_gate = AND()

    def forward(self, a, b):
        """Compute *a + b* (no carry-in).

        Parameters
        ----------
        a, b : torch.Tensor
            Tensors of shape ``(batch, 1)`` (or broadcast-compatible) holding
            the least-significant bit of each operand.
        Returns
        -------
        (sum, carry)
            `sum` and `carry` are tensors of shape ``(batch, 1)``.
        """
        output = self.xor_gate(a, b)
        c_out = self.and_gate(a, b)

        return output, c_out

class full_adder(nn.Module):

    def __init__(self):
        super().__init__()

        self.ha = half_adder()
        self.or_gate = OR()

    def forward(self, a, b, c_in):
        """Full adder: *a + b + c_in*.

        All inputs are tensors of shape ``(batch, 1)``.
        Returns ``(sum, carry)`` each of shape ``(batch, 1)``.
        """
        out_t, c_t1 = self.ha(a, b)
        output, c_t2 = self.ha(out_t, c_in)

        c_out = self.or_gate(c_t1, c_t2)

        return output, c_out

class ripple_carry_adder(nn.Module):

    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits

        self.ha = half_adder()
        self.fa = full_adder()
    
    # LSB first:
    def forward(self, x, y):

        output = torch.zeros_like(x)
        carry = torch.zeros_like(x)

        for i in range(self.n_bits):
            a = x[:, i].unsqueeze(1)
            b = y[:, i].unsqueeze(1)
            if i == 0:
                out_t, c_out = self.ha(a, b)
            else:
                out_t, c_out = self.fa(a, b, c_out)
            
            carry[:, i] = c_out.squeeze()
            
            output[:, i] = out_t.squeeze()
        
        return output, carry

class carry_save_adder(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits

        self.rca = ripple_carry_adder(n_bits)
        self.fa = full_adder()

    def forward(self, x, mask=None):

        len_x = x.shape[-2]

        if mask == None:
            mask = torch.ones(len_x)

        if len_x < 2:
            raise ValueError("Input x must have length at least 2")
        elif len_x == 2:
            return self.rca(x[:, 0], x[:, 1])
        else:
            # carry = torch.zeros_like(x[:, 0].unsqueeze(-1))
            carry = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device, requires_grad=True)

            in_0 = x[:, 0].unsqueeze(-1)
            t_carry = x[:, 1].unsqueeze(-1)

            for i in range(len_x-2):
                if not mask[i+2]:
                    continue

                in_1 = x[:, i+2].unsqueeze(-1)

                in_0, t_carry = self.fa(in_0, in_1, t_carry)
                carry = carry + t_carry[:, -1].squeeze(-1)
                # carry = carry + t_carry
                t_carry = torch.roll(t_carry, shifts=1, dims=-2)
                t_carry[:, 0] = 0

            output, t_carry = self.rca(in_0.squeeze(-1), t_carry.squeeze(-1))

            # carry = carry.squeeze(-1) + t_carry
            carry = carry + t_carry[:, -1]

            return output, carry