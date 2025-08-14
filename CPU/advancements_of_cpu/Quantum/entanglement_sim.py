#!/usr/bin/env python3
"""
Quantum entanglement demo (Bell state). Saves plot & JSON under results/quantum/.
Runs only if qiskit is available; otherwise exits gracefully.
"""
import os, sys, json
from pathlib import Path

def main():
    try:
        from qiskit import QuantumCircuit, Aer, execute
        from qiskit.visualization import plot_bloch_vector
        import numpy as np
        import matplotlib
        if not (os.environ.get("DISPLAY") or sys.platform == "win32"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] Qiskit (and plotting libs) not available; skipping entanglement demo.")
        return

    RESULTS = Path("results/quantum"); RESULTS.mkdir(parents=True, exist_ok=True)

    # Build Bell state |Φ+> = (|00> + |11>)/sqrt(2)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0,1)
    qc.measure([0,1], [0,1])

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend=backend, shots=4096)
    counts = job.result().get_counts(qc)

    # crude "fidelity-like" score (probability of 00 or 11)
    bell_fidelity = float(counts.get('00',0) + counts.get('11',0)) / 4096.0

    # Simple Bloch for |0> as placeholder visual (keep demo concise)
    bloch_fig = plt.figure(figsize=(4,4))
    plot_bloch_vector([0,0,1], title="|0> Bloch")
    bloch_path = RESULTS / "entanglement.png"
    plt.savefig(bloch_path, dpi=200, bbox_inches="tight")
    plt.close(bloch_fig)

    out = {"counts": counts, "bell_fidelity": bell_fidelity}
    with open(RESULTS / "entanglement_result.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Quantum entanglement demo saved -> {bloch_path}")
    if os.environ.get("DISPLAY") or sys.platform == "win32":
        plt.show()

if __name__ == "__main__":
    main()

