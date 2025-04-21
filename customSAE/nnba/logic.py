import torch
import torch.nn as nn

class AND(nn.Module):

    def __init__(self, NOT=False):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=True)
        ps = 16

        with torch.no_grad():
            if NOT:
                self.fc.weight.data = torch.tensor([[-1.*ps, -1.*ps]])
                self.fc.bias.data = torch.tensor([1.5*ps])
            else:
                self.fc.weight.data = torch.tensor([[1.*ps, 1.*ps]])
                self.fc.bias.data = torch.tensor([-1.5*ps])

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        with torch.no_grad():
            bool_x = (x >= 0.5).float()

        return x + (bool_x - x).detach()

class OR(nn.Module):

    def __init__(self, NOT=False):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=True)
        ps = 16

        with torch.no_grad():
            if NOT:
                self.fc.weight.data = torch.tensor([[-1.*ps, -1.*ps]])
                self.fc.bias.data = torch.tensor([0.5*ps])
            else:
                self.fc.weight.data = torch.tensor([[1.*ps, 1.*ps]])
                self.fc.bias.data = torch.tensor([-0.5*ps])

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        with torch.no_grad():
            bool_x = (x >= 0.5).float()

        return x + (bool_x - x).detach()

class XOR(nn.Module):

    def __init__(self):
        super().__init__()

        self.nand_gate = AND(NOT=True)
        self.and_gate = AND()
        self.or_gate = OR()
    
    def forward(self, x):
        out_nand = self.nand_gate(x)
        out_or = self.or_gate(x)

        in_and = torch.cat((out_or, out_nand), dim=-1)

        return self.and_gate(in_and)

# testing_cases = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
# 
# a = AND(NOT=True)
# print(a(testing_cases))
# 
# o = OR(NOT=True)
# print(o(testing_cases))

# xor = XOR()
# print(xor(testing_cases))