import torch
import torch.nn as nn
from logic import *

class half_adder(nn.Module):

    def __init__(self):
        super().__init__()

        self.xor_gate = XOR()
        self.and_gate = AND()

    def forward(self, input):
        output = self.xor_gate(input)
        c_out = self.and_gate(input)

        return output, c_out

class full_adder(nn.Module):

    def __init__(self):
        super().__init__()

        self.ha = half_adder()
        self.or_gate = OR()

    def forward(self, input, c_in):
        out_t, c_t1 = self.ha(input)
        output, c_t2 = self.ha(torch.concat((out_t, c_in), dim=1))

        c_out = self.or_gate(torch.concat((c_t1, c_t2), dim=1))

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

        for i in range(self.n_bits):
            input = torch.concat((x[:, i].unsqueeze(1), y[:, i].unsqueeze(1)), dim=1)
            if i == 0:
                out_t, c_out = self.ha(input)
            else:
                out_t, c_out = self.fa(input, c_out)
            
            output[:, i] = out_t.squeeze()
        
        return output

class carry_save_adder(nn.Module):

    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits

        self.rca = ripple_carry_adder(n_bits)
        self.fa = full_adder()

    def forward(self, x):

        len_x = len(x)
        if len_x < 2:
            raise ValueError("Input x must have length at least 2")
        elif len_x == 2:
            return self.rca(x[0].unsqueeze(0), x[1].unsqueeze(0))
        else:
            in_0 = x[0].unsqueeze(1)
            carry = x[1].unsqueeze(1)
            for i in range(len_x-2):
                in_1 = x[i+2].unsqueeze(1)

                in_0, carry = self.fa(torch.concat((in_0, in_1), dim=1), carry)
                carry = torch.cat((torch.zeros_like(carry[0:1]), carry[:-1]), dim=0)
            return self.rca(in_0.squeeze(1).unsqueeze(0), carry.squeeze(1).unsqueeze(0))

# testing_cases = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])

# adder_1 = half_adder()
# print(adder_1(testing_cases))

# adder_2 = full_adder()
# 
# carry_0 = torch.tensor([[0.], [0.], [0.], [0.]])
# carry_1 = torch.tensor([[1.], [1.], [1.], [1.]])
# print(adder_2(testing_cases, carry_0))
# print(adder_2(testing_cases, carry_1))

# testing_cases_x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
# testing_cases_y = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
# 
# rca = ripple_carry_adder(2)
# print(rca(testing_cases_x, testing_cases_y))

# test_values = [
#     # Edge cases
#     (0, 0),       # 0000 + 0000
#     (15, 15),     # 1111 + 1111
#     (8, 7),       # 1000 + 0111
#     (1, 15),      # 0001 + 1111
#     
#     # Random cases
#     (3, 5),       # 0011 + 0101
#     (10, 6),      # 1010 + 0110
#     (13, 9),      # 1101 + 1001
#     (7, 11)       # 0111 + 1011
# ]
# 
# # Convert to binary tensors (batch_size=8, bits=4)
# x_bin = torch.tensor([[int(b) for b in f"{x:04b}"[::-1]] for x, _ in test_values], dtype=torch.float32)
# y_bin = torch.tensor([[int(b) for b in f"{y:04b}"[::-1]] for _, y in test_values], dtype=torch.float32)
# 
# rca = ripple_carry_adder(4)
# res = rca(x_bin, y_bin)
# 
# for i, (x_dec, y_dec) in enumerate(test_values):
#     # Convert model output to decimal
#     sum_bits = res[i].tolist()[::-1]
#     sum_pred = int("".join(map(str, map(int, sum_bits))), 2)
#     
#     # Calculate expected result (including carry overflow)
#     expected_sum = (x_dec + y_dec) & 0b1111
#     expected_bits = [int(b) for b in f"{expected_sum:04b}"]  # 5-bit output
#     
#     print(f"Test case {i+1}:")
#     print(f"Input: {x_dec:04b} + {y_dec:04b}")
#     print(f"Predicted: {sum_pred:04b} ({sum_pred})")
#     print(f"Expected:  {expected_sum:04b} ({expected_sum})")
#     print("Status:", "PASS" if sum_pred == expected_sum else "FAIL")
#     print("-" * 40)

def construct_binary_list(num, n_bits):
    return [int(b) for b in f"{num:0{n_bits}b}"[::-1]]

testing_case = torch.tensor([construct_binary_list(2, 8), construct_binary_list(2, 8), construct_binary_list(2, 8), construct_binary_list(3, 8)], dtype=torch.float32)
print(testing_case)

csa = carry_save_adder(8)
print(csa(testing_case).tolist())