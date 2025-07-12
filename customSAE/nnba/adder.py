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
            # Initialize carry with zeros but don't track gradients for in-place operations
            carry = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

            in_0 = x[:, 0].unsqueeze(-1)
            t_carry = x[:, 1].unsqueeze(-1)

            for i in range(len_x-2):
                if not mask[i+2]:
                    continue

                in_1 = x[:, i+2].unsqueeze(-1)

                in_0, t_carry = self.fa(in_0, in_1, t_carry)
                # Use out-of-place addition to avoid in-place gradient issues
                carry = carry + t_carry[:, -1].squeeze(-1).detach()
                # carry = carry + t_carry
                t_carry = torch.roll(t_carry, shifts=1, dims=-2)
                t_carry[:, 0] = 0

            output, t_carry = self.rca(in_0.squeeze(-1), t_carry.squeeze(-1))

            # Final accumulation of carry
            carry = carry + t_carry[:, -1].detach()

            return output, carry

class residual_adder(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits

        # self.rca = ripple_carry_adder(n_bits)
        self.optimized_binary_adder = OptimizedBinaryAdder(n_bits)

    def forward(self, x, p=None, true_sum=None, true_carry=None):

        len_x = x.shape[-2]

        if len_x < 2:
            raise ValueError("Input x must have length at least 2")
        elif len_x == 2:
            # return self.rca(x[:, 0], x[:, 1])
            carry_n = torch.zeros_like(x[:, 0, 0])
            return self.optimized_binary_adder(x[:, 0], x[:, 1], carry_n, p, true_sum, true_carry)
        else:
            # Calculate the residual of the sum w.r.t. the input bits
            with torch.no_grad():
                powers = 2 ** torch.arange(self.n_bits, device=x.device)
                x_sum = (x.sum(dim=-2) * powers).sum(dim=-1)
                x_int = (x * powers).sum(dim=-1)
                x_residual = (((x_sum.unsqueeze(-1) - x_int).int().unsqueeze(-1) & powers.int()) > 0).float()
                carry_n = x_sum // 2 ** self.n_bits

            return self.optimized_binary_adder(x.view(-1, x.shape[-1]), x_residual.view(-1, x_residual.shape[-1]), carry_n, p, true_sum, true_carry)

class surrogate_gradient_adder_dense(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits

        self.rca = ripple_carry_adder(n_bits)
        # self.optimized_binary_adder = OptimizedBinaryAdder(n_bits)

    def forward(self, x, p=None):

        len_x = x.shape[-2]

        if len_x < 2:
            raise ValueError("Input x must have length at least 2")
        elif len_x == 2:
            return self.rca(x[:, 0], x[:, 1])
            # return self.optimized_binary_adder(x[:, 0], x[:, 1], p)
        else:
            # Calculate the residual of the sum w.r.t. the input bits
            with torch.no_grad():
                powers = 2 ** torch.arange(self.n_bits, device=x.device)
                x_sum = (x.sum(dim=-2) * powers).sum(dim=-1)
                x_int = (x * powers).sum(dim=-1)
                x_residual = (((x_sum.unsqueeze(-1) - x_int).int().unsqueeze(-1) & powers.int()) > 0).float()
                # carry_n = x_sum // 2 ** self.n_bits

            output, carry = self.rca(x.view(-1, x.shape[-1]), x_residual.view(-1, x_residual.shape[-1]))
            # self.optimized_binary_adder(x.view(-1, x.shape[-1]), x_residual.view(-1, x_residual.shape[-1]), carry_n, p)

            return output, carry

class OptimizedBinaryAdderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, cn, p, true_sum, true_carry):
        # mask shpe: [batch, latent_dim, 1]
        # a, b shape: [batch, latent_dim * n_neurons, n_bits]
        N, n_bits = a.shape
        
        ctx.save_for_backward(a, b, cn, p, true_sum, true_carry)
        ctx.n_bits = n_bits
        
        # Create trigger tensor that's connected to the computation graph
        # Sum a small portion of the input to maintain gradient connection
        trigger = a.sum() * 0.0 + 1.0

        return trigger
    
    @staticmethod
    def backward(ctx, trigger):

        a, b, cn, p, true_sum, true_carry = ctx.saved_tensors
        n_bits = ctx.n_bits

        b_reverse = torch.round(1 - b).to(torch.uint8)
        sum_binary = torch.round(true_sum).to(torch.uint8)
        
        residual_bits = torch.zeros_like(b_reverse)
        carry = torch.zeros_like(b_reverse).to(torch.int32)

        binary_weights = 2 ** torch.arange(n_bits, device=a.device)
        
        for i in range(n_bits):
            if i == 0:
                residual_bits[:, i] = b_reverse[:, i] ^ sum_binary[:, i] ^ 1
                carry[:, i] = b_reverse[:, i] | sum_binary[:, i]
            else:
                residual_bits[:, i] = b_reverse[:, i] ^ sum_binary[:, i] ^ carry[:, i-1]
                carry[:, i] = (b_reverse[:, i] & sum_binary[:, i]) | \
                              (sum_binary[:, i] & carry[:, i-1]) | \
                              (b_reverse[:, i] & carry[:, i-1])

        residual_carry = carry[:, -1] - 1 - cn + true_carry[:, -1]

        grad_a = 2 * residual_bits * binary_weights + 2 * residual_carry.unsqueeze(-1) * 2 ** n_bits - binary_weights
        grad_a *= - p * (1 - p)

        return grad_a, None, None, None, None, None

class OptimizedBinaryAdder(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits
    
    def forward(self, a, b, cn, p, true_sum, true_carry):

        return OptimizedBinaryAdderFunction.apply(a, b, cn, p, true_sum, true_carry)