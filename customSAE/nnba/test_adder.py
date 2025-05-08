import itertools, math, torch
from adder import ripple_carry_adder, carry_save_adder

# ----------------------------------------------------------------------
def int2bits(n: int, n_bits: int):
    """Return list of n_bits bits (LSB-first)."""
    return [(n >> i) & 1 for i in range(n_bits)]

def bits2tensor(bits):
    """1-D list[int] → column-vector float32 tensor."""
    return torch.tensor(bits, dtype=torch.float32).unsqueeze(0)  # shape (1, n_bits)

# ----------------------------------------------------------------------
def test_rca(n_bits: int):
    rca = ripple_carry_adder(n_bits)

    max_val = 1 << n_bits
    # For ≤ 4 bits test exhaustively, otherwise test 1 k random pairs
    pairs = (
        list(itertools.product(range(max_val), repeat=2))
        if n_bits <= 4
        else [(torch.randint(max_val, (1,)).item(),
               torch.randint(max_val, (1,)).item())
              for _ in range(1_000)]
    )

    for a_int, b_int in pairs:
        a_bits = bits2tensor(int2bits(a_int, n_bits))
        b_bits = bits2tensor(int2bits(b_int, n_bits))

        # Forward
        sum_bits, carries = rca(a_bits, b_bits)
        sum_bits = sum_bits.round().int().view(-1).tolist()
        carries  = carries.round().int().view(-1).tolist()

        # Expected results
        true_sum = (a_int + b_int) & (max_val - 1)
        true_cout = (a_int + b_int) >> n_bits
        exp_bits = int2bits(true_sum, n_bits)

        # -------- expected per-bit carries ------------------------------
        expected_carries = []
        c_prev = 0
        for bit_idx in range(n_bits):
            a_bit = (a_int >> bit_idx) & 1
            b_bit = (b_int >> bit_idx) & 1
            c_out = 1 if (a_bit + b_bit + c_prev) >= 2 else 0
            expected_carries.append(c_out)
            c_prev = c_out
        # ---------------------------------------------------------------

        assert sum_bits == exp_bits, \
            f"RCA {n_bits}-bit: {a_int}+{b_int} → wrong sum {sum_bits} exp {exp_bits}"
        assert carries == expected_carries, \
            f"RCA {n_bits}-bit: {a_int}+{b_int} → wrong carry {carries} exp {expected_carries}"

    print(f"ripple_carry_adder {n_bits:>2}-bit: PASS ({len(pairs)} cases)")

# ----------------------------------------------------------------------
def test_csa_two_operands(n_bits: int):
    """carry_save_adder behaves like RCA when len_x==2."""
    csa = carry_save_adder(n_bits)
    max_val = 1 << n_bits

    for _ in range(1_000):
        a_int = torch.randint(max_val, (1,)).item()
        b_int = torch.randint(max_val, (1,)).item()

        a_bits = bits2tensor(int2bits(a_int, n_bits))
        b_bits = bits2tensor(int2bits(b_int, n_bits))
        x = torch.stack((a_bits, b_bits), dim=1)     # shape (1, 2, n_bits)

        sum_bits, carries = csa(x)
        sum_bits = sum_bits.round().int().view(-1).tolist()   # always list
        carries   = carries.round().int().view(-1).tolist()

        true_sum = (a_int + b_int) & (max_val - 1)
        exp_bits = int2bits(true_sum, n_bits)

        # ---------- expected carries (identical to RCA) --------------------
        expected_carries = []
        c_prev = 0
        for bit_idx in range(n_bits):
            a_bit = (a_int >> bit_idx) & 1
            b_bit = (b_int >> bit_idx) & 1
            c_out = (a_bit + b_bit + c_prev) // 2
            expected_carries.append(c_out)
            c_prev = c_out
        # -------------------------------------------------------------------

        assert sum_bits == exp_bits, \
            f"CSA {n_bits}-bit: {a_int}+{b_int} → wrong sum {sum_bits} exp {exp_bits}"
        assert carries == expected_carries, \
            f"CSA {n_bits}-bit: {a_int}+{b_int} → wrong carry {carries} exp {expected_carries}"

    print(f"carry_save_adder {n_bits:>2}-bit (len_x=2): PASS (1000 random cases)")

# ----------------------------------------------------------------------
def test_csa_multi_operands(n_bits: int, n_ops: int):
    """Test carry_save_adder with *n_ops* (>2) operands.

    For > 2 operands we validate both the *sum* bits and the per-bit
    accumulated carries.  The carries are defined as

        C_i = ⌊(number_of_1s_at_bit_i + C_{i−1}) / 2⌋

    with C_{−1}=0.  This corresponds to the total number of carry events
    generated at each bit position while summing the operands.
    """

    assert n_ops >= 2, "n_ops must be >= 2"

    csa = carry_save_adder(n_bits)
    max_val = 1 << n_bits

    n_trials = 5_000 if n_ops <= 4 else 1_000

    for _ in range(n_trials):
        ints = [torch.randint(max_val, (1,)).item() for _ in range(n_ops)]

        bit_tensors = [bits2tensor(int2bits(v, n_bits)) for v in ints]
        # Shape: (1, n_ops, n_bits)
        x = torch.stack(bit_tensors, dim=1)

        sum_bits, carries = csa(x)
        sum_bits = sum_bits.round().int().view(-1).tolist()
        carries_list = carries.round().int().view(-1).tolist()

        true_sum = sum(ints) & (max_val - 1)
        exp_bits = int2bits(true_sum, n_bits)

        # ----- verify sum bits -----
        assert sum_bits == exp_bits, (
            f"CSA {n_bits}-bit: {'+'.join(map(str, ints))} → wrong sum {sum_bits} exp {exp_bits}")

        # ----- verify carry bits individually -----
        expected_carries = []
        c_prev = 0
        for bit_idx in range(n_bits):
            ones = sum((v >> bit_idx) & 1 for v in ints)
            c_out = (ones + c_prev) // 2
            expected_carries.append(c_out)
            c_prev = c_out

        assert carries_list == expected_carries, (
            f"CSA {n_bits}-bit: {'+'.join(map(str, ints))} → wrong carry {carries_list} exp {expected_carries}")

    print(
        f"carry_save_adder {n_bits:>2}-bit (len_x={n_ops}): PASS ({n_trials} random cases)")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    for n in range(1, 9):          # 1- … 8-bit
        test_rca(n)
        test_csa_two_operands(n)

        # test CSA with 3 and 4 operands as a basic sanity check
        for n_ops in (3, 4):
            test_csa_multi_operands(n, n_ops)

    print("All adder tests passed ✔")