# Variation-Quantum-Eigensolver

## What it does

Finds the smallest eigenvalue (ground-state energy) of a Hamiltonian by:
1. Preparing a parameterized quantum state (ansatz) on a classically simulated statevector.
2. Computing `⟨ψ(θ)|H|ψ(θ)⟩`.
3. Using a classical optimizer to adjust `θ` and minimize that expectation value.

Components

- Statevector simulator: applies gates directly to a 2^n-length complex vector via tensor reshaping, avoiding dense 2^n x 2^n gate matrices. Supports single-qubit gates (RY, RZ, X, Y, Z, I) and CNOT between two qubits.
- Hamiltonian: represents H as a weighted sum of Pauli strings, e.g. -1.05*II + 0.18*XX. Can build the dense matrix for verification, and can classically diagonalize itself to produce the exact ground-state energy as a ground truth to check VQE against.
- HardwareEfficientAnsatz: alternates layers of single-qubit rotations (RY, optionally RZ) with a CNOT entangling ladder connecting neighboring qubits. Depth and rotation set are configurable; produces the resulting statevector for a given parameter set.
- Expectation value function: computes the energy of a state under a Hamiltonian by applying each Pauli term to the state and taking the inner product, without constructing the full Hamiltonian matrix.
- VQE: ties the above together. Runs one optimization pass through SciPy's minimize (COBYLA by default), or repeats that pass from several random starting points and keeps the best result, since the cost landscape is non-convex. Each run returns the optimal energy, the optimal parameters, and the full energy history.

## Included example Hamiltonians

- `h2_hamiltonian()` — published 2-qubit tapered Hamiltonian for the H₂ molecule (STO-3G basis, Jordan-Wigner + parity tapering, bond length 0.735 Å).
- `random_hamiltonian(n_qubits, n_terms, seed)` — random Hermitian Hamiltonian built from random Pauli strings, for testing on arbitrary problems.

## Limitations

- Exact diagonalization (used only for verification) scales as `2^n`, so it's only practical for small qubit counts (~≤12).
- The ansatz's own simulation also scales as `2^n` since it's a full statevector simulator, not a real quantum device.
- No noise model — this simulates an ideal, noiseless quantum computer.
- No analytic gradients — COBYLA is gradient-free; a parameter-shift gradient method could be added for gradient-based optimizers (e.g. L-BFGS-B, Adam).
