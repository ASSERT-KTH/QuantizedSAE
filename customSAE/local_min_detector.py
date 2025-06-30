import torch

torch.set_printoptions(profile="full")

class LocalMinDetector:
    def __init__(self):
        self.ctx = None

    def forward(self, a, b, p):
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
                Dc[:, i, i] = b_binary[:, i] + carry[:, i-1] - b_binary[:, i] * carry[:, i-1]

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

        self.ctx = (a_binary, sum, carry, Ds, Dc, mask_idx, p, n_bits)
        
        return sum.float(), carry.float()
    
    def backward(self, err_sum, err_carry):

        a, sum, carry, Ds, Dc, mask_idx, p, n_bits = self.ctx
        a = a.float()
        Ds = Ds.float()
        Dc = Dc.float()
        mask_idx = mask_idx.float()

        scale = 2 ** torch.arange(n_bits, device=a.device, dtype=torch.float32)
        scale /= scale.sum()
        carry_scale = scale * 2
        # carry_scale[:-1] *= 0.5 ** n_bits
        carry[:, :-1] = 0.
        carry_scale[:-1] = 0.
        
        true_sum = err_sum.to(torch.int8) ^ sum
        # true_carry = err_carry.to(torch.int8) ^ carry
        true_carry = err_carry.to(torch.int8) ^ carry

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

    def local_min_test(self, n_bits):
        rg = torch.arange(2 ** n_bits).unsqueeze(-1)
        shift_factor = torch.arange(n_bits)

        operands = (rg >> shift_factor) & 1

        addend = operands.repeat(operands.shape[0], 1, 1).view(-1, n_bits)
        op0 = operands.unsqueeze(1).repeat(1, operands.shape[0], 1).view(-1, n_bits)
        p = torch.ones_like(addend) * 0.5

        for i in range(2 ** n_bits):
            print(f"Current op1: {i}")
            op1 = (i >> shift_factor) & 1
            op1 = op1.repeat(addend.shape[0], 1)

            target_sum, target_carry = self.forward(op1, addend, p)

            true_sum, true_carry = self.forward(op0, addend, p)

            incorrect_bits = (target_sum != true_sum).float()

            target_carry[:, :-1] = true_carry[:, :-1]
            incorrect_carry = (true_carry != target_carry).float()

            grad_a, _, _ = self.backward(incorrect_bits, incorrect_carry)

            target_sign = torch.sign(((i >> shift_factor) & 1).repeat(2 ** n_bits, 1) - operands)
            target_sign = torch.where(target_sign == 0, 2 * ((i >> shift_factor) & 1).repeat(2 ** n_bits, 1) - 1, target_sign) * -1
            grad_a = grad_a.view(2 ** n_bits, 2 ** n_bits, n_bits)
            avg_a = grad_a.sum(dim=1)

            true_grad_a = torch.sign(avg_a)

            print((target_sign != true_grad_a).float().sum())

if __name__ == "__main__":
    detector = LocalMinDetector()
    detector.local_min_test(4)