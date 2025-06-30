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

class surrogate_gradient_adder_dense(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits

        # self.rca = ripple_carry_adder(n_bits)
        self.optimized_binary_adder = OptimizedBinaryAdder(n_bits)

    def forward(self, x, p=None):

        len_x = x.shape[-2]

        if len_x < 2:
            raise ValueError("Input x must have length at least 2")
        elif len_x == 2:
            # return self.rca(x[:, 0], x[:, 1])
            return self.optimized_binary_adder(x[:, 0], x[:, 1], p)
        else:
            # Calculate the residual of the sum w.r.t. the input bits
            with torch.no_grad():
                powers = 2 ** torch.arange(self.n_bits, device=x.device)
                x_sum = (x.sum(dim=-2) * powers).sum(dim=-1)
                x_int = (x * powers).sum(dim=-1)
                x_residual = (((x_sum.unsqueeze(-1) - x_int).int().unsqueeze(-1) & powers.int()) > 0).float()

            # output, carry = self.rca(x.view(-1, x.shape[-1]), x_residual.view(-1, x_residual.shape[-1]))
            output, carry = self.optimized_binary_adder(x.view(-1, x.shape[-1]), x_residual.view(-1, x_residual.shape[-1]), p)

            output = output.reshape(x.shape).float().mean(dim=-2)
            carry = carry.reshape(x.shape).float().sum(dim=-2)

            return output, carry

class OptimizedBinaryAdderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, p):
        # mask shpe: [batch, latent_dim, 1]
        # a, b shape: [batch, latent_dim * n_neurons, n_bits]
        N, n_bits = a.shape
        
        a_binary = a.round().to(torch.int8)
        b_binary = b.round().to(torch.int8)
        
        sum = torch.zeros_like(a_binary)
        carry = torch.zeros_like(a_binary)

        Ds = torch.zeros(N, n_bits, n_bits, device=a.device, dtype=torch.float32)
        Dc = torch.zeros_like(Ds)
        
        for i in range(n_bits):
            if i == 0:
                sum[:, i] = a_binary[:, i] ^ b_binary[:, i]  # XOR
                carry[:, i] = a_binary[:, i] & b_binary[:, i]     # AND

                Ds[:, i, i] = 1 - 2 * b_binary[:, i]
                # Dc[:, i, i] = a_binary[:, i]
                Dc[:, i, i] = b_binary[:, i]
            else:
                sum[:, i] = a_binary[:, i] ^ b_binary[:, i] ^ carry[:, i-1]
                carry[:, i] = (a_binary[:, i] & b_binary[:, i]) | \
                              (a_binary[:, i] & carry[:, i-1]) | \
                              (b_binary[:, i] & carry[:, i-1])
                
                Ds[:, i, i] = (1 - 2 * b_binary[:, i]) * (1 - 2 * carry[:, i-1])
                # Dc[:, i, i] = a_binary[:, i]
                Dc[:, i, i] = b_binary[:, i] + carry[:, i-1] - 2 * b_binary[:, i] * carry[:, i-1]

                ds_dcprev = (1 - 2 * a_binary[:, i]) * (1 - 2 * b_binary[:, i])
                # dc_dcprev = carry[:, i-1]
                dc_dcprev = a_binary[:, i] + b_binary[:, i] - a_binary[:, i] * b_binary[:, i]

                Ds[:, i, :i] = ds_dcprev.unsqueeze(-1) * Dc[:, i-1, :i]
                Dc[:, i, :i] = dc_dcprev.unsqueeze(-1) * Dc[:, i-1, :i]
        
        propagate = (a_binary ^ b_binary).bool()
        prop = b_binary.clone()
        if n_bits > 1:
            prop[:, 1:] ^= carry[:, :-1]    # bi xor c_{i‑1}.  For j=0, prop = b0.
        prop = prop.bool()

        mask_idx = torch.full_like(a_binary, n_bits, dtype=torch.int8)
        running_mask = torch.full((N,), n_bits, dtype=torch.int8)
        for i in reversed(range(n_bits)):
            mask_idx[:, i] = torch.where(prop[:, i], running_mask, torch.full_like(running_mask, i))
            running_mask = torch.where(propagate[:, i], running_mask, torch.full_like(running_mask, i))

        ctx.save_for_backward(a_binary, sum, carry, Ds, Dc, mask_idx, p)
        ctx.n_bits = n_bits
        
        return sum.float(), carry.float()
    
    @staticmethod
    def backward(ctx, err_sum, err_carry):

        a, sum, carry, Ds, Dc, mask_idx, p = ctx.saved_tensors
        a = a.float()
        Ds = Ds.float()
        Dc = Dc.float()
        mask_idx = mask_idx.float()
        n_bits = ctx.n_bits

        scale = 2 ** torch.arange(n_bits, device=a.device, dtype=torch.float32)
        scale /= scale.sum()
        carry_scale = scale * 2
        carry_scale[:-1] *= 0.5 ** n_bits
        
        true_sum = err_sum.to(torch.int8) ^ sum
        # true_carry = err_carry.to(torch.int8) ^ carry
        true_carry = err_carry.to(torch.int8)

        ori_grad_a = torch.zeros_like(a, dtype=torch.float32)
        # grad_b = torch.zeros_like(b)
        grad_b = None

        sum_matrix = (sum - true_sum.to(torch.float32)) * scale
        carry_matrix = (carry - true_carry.to(torch.float32)) * carry_scale
        # grad_a when a is the input
        # ori_grad_a = (
        #     torch.bmm(sum_matrix.unsqueeze(1), Ds) +          # (N,1,n_bits)
        #     torch.bmm(carry_matrix.unsqueeze(1), Dc)          # (N,1,n_bits)
        # ).squeeze(1)
        ori_grad_sum = torch.bmm(sum_matrix.unsqueeze(1), Ds)         # (N,1,n_bits)
        ori_grad_carry = torch.bmm(carry_matrix.unsqueeze(1), Dc)          # (N,1,n_bits)
        ori_grad_a = (ori_grad_sum + ori_grad_carry).squeeze(1)

        bits = torch.arange(n_bits)
        j_idx = bits.view(1, 1, n_bits)
        i_idx = bits.view(1, n_bits, 1)
        
        mask_idx = mask_idx.unsqueeze(-1)

        mask_sum = (j_idx >= i_idx) & (j_idx <= mask_idx)
        mask_carry = (j_idx >= i_idx) & (j_idx < mask_idx)

        alt_sum = mask_sum ^ sum.unsqueeze(1)
        alt_carry = mask_carry ^ carry.bool().unsqueeze(1)

        alt_sum_matrix = (alt_sum - true_sum.to(torch.float32).unsqueeze(1)) * scale
        alt_carry_matrix = (alt_carry.to(torch.float32) - true_carry.to(torch.float32).unsqueeze(1)) * carry_scale

        # alt_grad_a = (
        #     torch.einsum('bij,bji->bi', alt_sum_matrix, Ds).unsqueeze(1) +
        #     torch.einsum('bij,bji->bi', alt_carry_matrix, Dc).unsqueeze(1)
        # ).squeeze(1)

        alt_grad_sum = torch.einsum('bij,bji->bi', alt_sum_matrix, Ds).unsqueeze(1)
        alt_grad_carry = torch.einsum('bij,bji->bi', alt_carry_matrix, Dc).unsqueeze(1)
        alt_grad_a = (alt_grad_sum + alt_grad_carry).squeeze(1)

        if p is not None:
            ori_grad_a = torch.where(p > 0.5, p * ori_grad_a, (1 - p) * ori_grad_a)
            alt_grad_a = torch.where(p > 0.5, (1 - p) * alt_grad_a, p * alt_grad_a)

        grad_a = ori_grad_a + alt_grad_a

        return grad_a, grad_b, None

class OptimizedBinaryAdder(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits
    
    def forward(self, a, b, p):

        return OptimizedBinaryAdderFunction.apply(a, b, p)