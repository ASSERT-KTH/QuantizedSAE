import torch
from logic import AND, OR, XOR

def bool2tensor(arr):
    """Helper: list/tuple of 0 / 1 → column vector float32 tensor."""
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(1)

# ----------------------------------------------------------------------
# Truth tables  (a, b, expected)
truth_AND = [(0, 0, 0),
             (0, 1, 0),
             (1, 0, 0),
             (1, 1, 1)]

truth_OR  = [(0, 0, 0),
             (0, 1, 1),
             (1, 0, 1),
             (1, 1, 1)]

truth_XOR = [(0, 0, 0),
             (0, 1, 1),
             (1, 0, 1),
             (1, 1, 0)]

# ----------------------------------------------------------------------
def run_gate_test(name, gate, truth_table):
    a_vals, b_vals, exp_vals = zip(*truth_table)
    a = bool2tensor(a_vals)
    b = bool2tensor(b_vals)
    expected = bool2tensor(exp_vals)

    out = gate(a, b)      # forward with separate operands
    out = out.detach()    # no gradients needed for testing

    ok = torch.equal(out, expected)
    print(f"{name:<5}: {'PASS' if ok else 'FAIL'}  output = {out.squeeze().tolist()}")
    assert ok, f"{name} gate failed"

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Plain gates
    run_gate_test("AND", AND(), truth_AND)
    run_gate_test("OR",  OR(),  truth_OR)
    run_gate_test("XOR", XOR(), truth_XOR)

    # Negated versions (use NOT=True)
    run_gate_test("NAND", AND(NOT=True), [(a, b, 1 - y) for a, b, y in truth_AND])
    run_gate_test("NOR",  OR(NOT=True),  [(a, b, 1 - y) for a, b, y in truth_OR])

    print("All logic–gate tests passed ✔")