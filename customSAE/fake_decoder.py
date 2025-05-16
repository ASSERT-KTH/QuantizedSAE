import torch
import torch.nn as nn
from nnba.adder import *
import torch.nn.functional as F

class FakeDecoder(nn.Module):
    def __init__(self, input_dim):
        super(FakeDecoder, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.sga = surrogate_gradient_adder_dense(input_dim)
        self.threshold = 0.5

        nn.init.normal_(self.weight, mean=0.5, std=0.2)
        self.weight.data.clamp_(0, 1)
        self.hook_handle = None
        self.register_hook()

    def register_hook(self):

        def weight_hook(grad):
            distance_from_threshold = abs(self.weight - (self.weight >= self.threshold).float())

            return grad * distance_from_threshold

        self.hook_handle = self.weight.register_hook(weight_hook)

    def forward(self, x):
        with torch.no_grad():
            hard_weight = (self.weight > self.threshold).float()
        
        hard_weight_with_gradient = self.weight + (hard_weight - self.weight).detach()

        # torch.repeat(sizes...) repeats tensor along each dimension specified in sizes
        # Here we repeat along:
        # - dim 0: x.shape[0] times to match batch size
        # - dim 1: 1 time (no repeat)
        # - dim 2: 1 time (no repeat)
        op_1 = hard_weight_with_gradient.unsqueeze(0).repeat(x.shape[0], 1, 1)
        op_2 = x.unsqueeze(1)

        op = torch.cat((op_1, op_2), dim=1)

        return self.sga(op)
    
class FakeDecoderTrainer():
    def __init__(self, n_dim, training_goal):
        self.model = FakeDecoder(n_dim)
        self.n_dim = n_dim
        self.training_goal = training_goal
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.sga = surrogate_gradient_adder_dense(n_dim)
        self.batch_size = 1

        self.scale_factor = (2 ** torch.arange(n_dim, device=self.model.weight.device)).float()
        self.scale_factor /= self.scale_factor.sum()

    def synthetic_data(self):
        return torch.bernoulli(torch.ones(self.batch_size, self.n_dim) * 0.5).float()

    def train(self):

        op_1 = self.training_goal.unsqueeze(0).repeat(self.batch_size, 1, 1)

        for epoch in range(100000):

            x = self.synthetic_data()

            with torch.no_grad():
                op_2 = x.unsqueeze(1)

                op = torch.cat((op_1, op_2), dim=1)

                target, carry = self.sga(op)

            output, carry = self.model(x)

            self.optimizer.zero_grad()

            print(f"Weight before update is {self.model.weight}")
            loss = torch.mean(((output - target) ** 2 * self.scale_factor).sum(dim=-1))
            # loss = F.mse_loss(output, target)
            loss.backward()

            self.optimizer.step()
            print(f"Target is {self.training_goal}, current weight is {self.model.weight}")

n_dim = 2
training_goal = torch.bernoulli(torch.ones(n_dim) * 0.5).float()
print(training_goal)

trainer = FakeDecoderTrainer(n_dim, training_goal)
trainer.train()