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
        self.mask = None

    def forward(self, x):

        len_x = x.shape[-2]

        if len_x < 2:
            raise ValueError("Input x must have length at least 2")
        elif len_x == 2:
            # return self.rca(x[:, 0], x[:, 1])
            return self.optimized_binary_adder(x[:, 0], x[:, 1], self.mask)
        else:
            # Calculate the residual of the sum w.r.t. the input bits
            with torch.no_grad():
                powers = 2 ** torch.arange(self.n_bits, device=x.device)
                x_sum = (x.sum(dim=-2) * powers).sum(dim=-1)
                x_int = (x * powers).sum(dim=-1)
                x_residual = (((x_sum.unsqueeze(-1) - x_int).int().unsqueeze(-1) & powers.int()) > 0).float()

            # output, carry = self.rca(x.view(-1, x.shape[-1]), x_residual.view(-1, x_residual.shape[-1]))
            output, carry = self.optimized_binary_adder(x.view(-1, x.shape[-1]), x_residual.view(-1, x_residual.shape[-1]), self.mask)

            output = output.reshape(x.shape).float().mean(dim=-2)
            carry = carry.reshape(x.shape).float().sum(dim=-2)

            return output, carry

class OptimizedBinaryAdderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, mask):
        # mask shpe: [batch, latent_dim, 1]
        # a, b shape: [batch, latent_dim * n_neurons, n_bits]
        _, n_bits = a.shape
        
        a_binary = torch.round(a).to(torch.uint8)
        b_binary = torch.round(b).to(torch.uint8)
        
        sum_bits = torch.zeros_like(a_binary)
        carry = torch.zeros_like(a_binary)
        
        for i in range(n_bits):
            if i == 0:
                sum_bits[:, i] = a_binary[:, i] ^ b_binary[:, i]  # XOR
                carry[:, i] = a_binary[:, i] & b_binary[:, i]     # AND
            else:
                sum_bits[:, i] = a_binary[:, i] ^ b_binary[:, i] ^ carry[:, i-1]
                carry[:, i] = (a_binary[:, i] & b_binary[:, i]) | \
                              (a_binary[:, i] & carry[:, i-1]) | \
                              (b_binary[:, i] & carry[:, i-1])
        
        ctx.save_for_backward(a_binary, b_binary, carry, mask)
        ctx.n_bits = n_bits
        
        return sum_bits.float(), carry.float()
    
    @staticmethod
    def backward(ctx, grad_sum, grad_carry):

        a, b, carry, mask = ctx.saved_tensors
        n_bits = ctx.n_bits
        
        grad_a = torch.zeros_like(a, dtype=torch.float16)
        # grad_b = torch.zeros_like(b)
        grad_b = None
        
        # Calculate gradients for each bit position
        for i in range(n_bits):
            if i == 0:

                grad_a[:, i] += (1 - 2 * b[:, i]) * grad_sum[:, i]
                # grad_b[:, i] += (1 - 2 * a[:, i]) * grad_sum[:, i]
                
                grad_a[:, i] += b[:, i] * grad_carry[:, i]
                # grad_b[:, i] += a[:, i] * grad_carry[:, i]
            else:
                # Previous carry
                c_prev = carry[:, i-1]
                
                # For full adder:
                # ∂sum/∂a = (1 - 2b) * (1 - 2c_prev)   [from a⊕b⊕c_prev]
                # ∂sum/∂b = (1 - 2a) * (1 - 2c_prev)   [from a⊕b⊕c_prev]
                # ∂sum/∂c_prev = (1 - 2a) * (1 - 2b)   [from a⊕b⊕c_prev]
                
                # There are two options for the carry gradient:
                # ∂carry/∂a = b + c_prev - 2*b*c_prev  [from a&b | a&c_prev | b&c_prev]
                # ∂carry/∂b = a + c_prev - 2*a*c_prev  [from a&b | a&c_prev | b&c_prev]
                # ∂carry/∂c_prev = a + b - 2*a*b       [from a&b | a&c_prev | b&c_prev]
                # Or:
                # ∂carry/∂a = b + c_prev - b*c_prev  [from a&b | a&c_prev | b&c_prev]
                # ∂carry/∂b = a + c_prev - a*c_prev  [from a&b | a&c_prev | b&c_prev]
                # ∂carry/∂c_prev = a + b - a*b       [from a&b | a&c_prev | b&c_prev]
                # The second option is used here because 
                # it gives gradient when the other two operands are 1s.
                
                # Sum gradients
                grad_a[:, i] += (1 - 2 * b[:, i]) * (1 - 2 * c_prev) * grad_sum[:, i]
                # grad_b[:, i] += (1 - 2 * a[:, i]) * (1 - 2 * c_prev) * grad_sum[:, i]
                
                # Carry gradients for current bit
                grad_a[:, i] += (b[:, i] + c_prev - b[:, i] * c_prev) * grad_carry[:, i]
                # grad_a[:, i] += (b[:, i] + c_prev - 2 * b[:, i] * c_prev) * grad_carry[:, i]
                # grad_b[:, i] += (a[:, i] + c_prev - a[:, i] * c_prev) * grad_carry[:, i]
                
                # Propagate gradient to previous carry
                grad_sum_to_prev_carry = (1 - 2 * a[:, i]) * (1 - 2 * b[:, i]) * grad_sum[:, i]
                grad_carry_to_prev_carry = (a[:, i] + b[:, i] - a[:, i] * b[:, i]) * grad_carry[:, i]
                # grad_carry_to_prev_carry = (a[:, i] + b[:, i] - 2 * a[:, i] * b[:, i]) * grad_carry[:, i]
                
                # Add to the gradient of the previous bit's carry
                if i > 0:  # Only if not the first bit
                    grad_a[:, i-1] += grad_carry_to_prev_carry * b[:, i-1]
                    # grad_b[:, i-1] += grad_carry_to_prev_carry * a[:, i-1]
        
        if mask is None:
            mask = torch.ones(a.shape[0], 1, 1)

        batch_dim = mask.shape[0]
        feature_dim = mask.shape[1]
        neuron_dim = int(a.shape[0] / mask.shape[0] / mask.shape[1])
        mask = mask.unsqueeze(2).expand(-1, -1, neuron_dim, -1)
        mask = mask.reshape(batch_dim, feature_dim*neuron_dim, -1)
        mask = mask.expand(-1, -1, n_bits).reshape(-1, n_bits)
        reward_mask = (mask == 1) & (grad_a == 0)
        reward_factor = 0.01 * 2**torch.arange(a.shape[-1], device=a.device)
        reward_factor /= reward_factor.sum()
        reward_grad = reward_mask * reward_factor
        grad_a = grad_a + reward_grad * torch.sign(0.5 - a)
        # grad_b = grad_b * mask

        return grad_a, grad_b, None

class OptimizedBinaryAdder(nn.Module):

    def __init__(self, n_bits):

        super().__init__()
        self.n_bits = n_bits
    
    def forward(self, a, b, mask=None):

        return OptimizedBinaryAdderFunction.apply(a, b, mask)