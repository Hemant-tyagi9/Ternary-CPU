#!/usr/bin/env python3

import os, sys, json, random
from pathlib import Path

def main():
    try:
        from qiskit import QuantumCircuit, Aer, execute
        import matplotlib
        if not (os.environ.get("DISPLAY") or sys.platform == "win32"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] Qiskit not available; skipping quantum spiking demo.")
        return

    RESULTS = Path("results/quantum"); RESULTS.mkdir(parents=True, exist_ok=True)

    qc = QuantumCircuit(3,3)
    qc.h(0); qc.h(1); qc.cx(0,2); qc.cx(1,2)
    qc.measure([0,1,2],[0,1,2])

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend=backend, shots=2048)
    counts = job.result().get_counts(qc)

    # Toy "speedup" metric: skew toward high-correlation outcomes
    corr = sum(v for k,v in counts.items() if k.count('1') in (0,3)) / 2048.0
    estimated_speedup = 1.0 + 2.0 * corr  # 1x..3x

    # Plot bar of top outcomes
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]
    labels = [k for k,_ in items]; vals = [v for _,v in items]
    import numpy as np
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labels)
    ax.set_title("Quantum Spiking Toy Outcome Distribution")
    png = RESULTS / "quantum_spiking.png"
    fig.tight_layout(); fig.savefig(png, dpi=200, bbox_inches="tight"); plt.close(fig)

    with open(RESULTS / "quantum_spiking_result.json", "w") as f:
        json.dump({
            "top_counts": dict(items),
            "estimated_spike_speedup": estimated_speedup
        }, f, indent=2)

    print(f"[OK] Quantum spiking demo saved -> {png}")
    if os.environ.get("DISPLAY") or sys.platform == "win32":
        plt.show()

if __name__ == "__main__":
    main()

