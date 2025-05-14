import torch
import torch.nn as nn
import torch.autograd as autograd

# Binary operations with custom gradients using autograd.Function

class ANDFunction(autograd.Function):
    @staticmethod
    def forward(ctx, a, b, not_op=False):
        # Save inputs and parameters for backward
        ctx.save_for_backward(a, b)
        ctx.not_op = not_op
        
        # Compute the AND operation precisely
        result = a * b
        if not_op:
            result = 1.0 - result
            
        # Return exact binary values
        return torch.round(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        not_op = ctx.not_op
        
        # Calculate gradients for AND operation
        if not_op:
            grad_a = -b * grad_output
            grad_b = -a * grad_output
        else:
            grad_a = b * grad_output
            grad_b = a * grad_output
            
        # Return gradients for each input (and None for not_op parameter)
        return grad_a, grad_b, None

class ORFunction(autograd.Function):
    @staticmethod
    def forward(ctx, a, b, not_op=False):
        # Save inputs and parameters for backward
        ctx.save_for_backward(a, b)
        ctx.not_op = not_op
        
        # Compute the OR operation precisely
        result = a + b - a * b  # OR logic
        if not_op:
            result = 1.0 - result
            
        # Return exact binary values
        return torch.round(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        not_op = ctx.not_op
        
        # Calculate gradients for OR operation
        # For OR: c = a + b - a*b
        # dc/da = 1 - b
        # dc/db = 1 - a
        # For NOR (NOT OR): c = 1 - (a + b - a*b)
        # dc/da = -(1 - b)
        # dc/db = -(1 - a)
        if not_op:
            grad_a = -(1 - b) * grad_output
            grad_b = -(1 - a) * grad_output
        else:
            grad_a = (1 - b) * grad_output
            grad_b = (1 - a) * grad_output
        
        # Return gradients for each input (and None for not_op parameter)
        return grad_a, grad_b, None

class XORFunction(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save inputs for backward
        ctx.save_for_backward(a, b)
        
        # Compute the XOR operation precisely
        result = a + b - 2 * a * b
            
        # Return exact binary values
        return torch.round(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        # Calculate gradients for XOR operation
        # For XOR: c = a + b - 2*a*b
        # dc/da = 1 - 2*b
        # dc/db = 1 - 2*a
        grad_a = (1 - 2 * b) * grad_output
        grad_b = (1 - 2 * a) * grad_output
        
        # Return gradients for each input
        return grad_a, grad_b

# Helper functions for easier use (similar to functional API)
def binary_and(a, b, not_op=False):
    return ANDFunction.apply(a, b, not_op)

def binary_or(a, b, not_op=False):
    return ORFunction.apply(a, b, not_op)

def binary_xor(a, b):
    return XORFunction.apply(a, b)

# Keep nn.Module wrappers for backward compatibility if needed
class AND(nn.Module):
    def __init__(self, NOT: bool = False, ps: float = 1.0):
        super().__init__()
        self.NOT = NOT
        self.ps = ps
        
    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return binary_and(a, b, self.NOT)

class OR(nn.Module):
    def __init__(self, NOT: bool = False, ps: float = 1.0):
        super().__init__()
        self.NOT = NOT
        self.ps = ps
        
    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return binary_or(a, b, self.NOT)

class XOR(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return binary_xor(a, b)