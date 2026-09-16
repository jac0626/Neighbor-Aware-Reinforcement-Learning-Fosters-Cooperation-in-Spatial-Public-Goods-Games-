"""Check supplied rounded means and a conditional SPGG identity, not RL results."""
from decimal import Decimal
from fractions import Fraction
from itertools import product
import json
from pathlib import Path


def main():
    means = {"baseline_clean": "0.959", "baseline_fault": "0.138",
             "e2e_clean": "0.773", "e2e_fault": "0.778"}
    m = {key: Decimal(value) for key, value in means.items()}
    differences = {
        "e2e_minus_baseline_fault": str(m["e2e_fault"] - m["baseline_fault"]),
        "e2e_minus_baseline_clean": str(m["e2e_clean"] - m["baseline_clean"]),
        "baseline_fault_minus_clean": str(m["baseline_fault"] - m["baseline_clean"]),
        "e2e_fault_minus_clean": str(m["e2e_fault"] - m["e2e_clean"]),
    }
    # Directly distribute group benefits and subtract each member's contribution.
    # Binary 1 denotes cooperation here; this is not work1's action encoding.
    side = 3
    groups = [[((x + dx) % side) * side + (y + dy) % side
               for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]]
              for x in range(side) for y in range(side)]
    cases = 0
    for r in [Fraction(22, 5), Fraction(24, 5)]:
        for cooperates in product([0, 1], repeat=side * side):
            payoffs = [Fraction(0) for _ in cooperates]
            for group in groups:
                benefit = r * sum(cooperates[i] for i in group) / len(group)
                for i in group:
                    payoffs[i] += benefit - cooperates[i]
            expected = 5 * (r - 1) * sum(cooperates)
            if sum(payoffs) != expected:
                raise AssertionError((r, cooperates, payoffs, expected))
            cases += 1
    result = {
        "source": "User-supplied interim presentation; rounded means, not raw runs",
        "reported_means": means,
        "differences_in_cooperation_fraction": differences,
        "welfare_identity": {
            "formula": "sum(P_i) = 5 * (r - 1) * N_C",
            "conditions": "periodic lattice; five groups per agent; cost 1 in every group; raw true payoffs",
            "arithmetic": "exact rational",
            "grid": [3, 3], "r": ["4.4", "4.8"],
            "enumerated_configurations": cases, "passed": True,
        },
        "not_validated": ["work2 raw results", "controller training", "statistical significance",
                          "all skill helper scripts", "writing improvement relative to no-skill control"],
    }
    output = Path(__file__).resolve().parents[1] / "build" / "skill-validation-checks.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
