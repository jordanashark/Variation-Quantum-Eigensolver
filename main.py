from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

_PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_string_to_matrix(pauli_string: str) -> np.ndarray:
    mats = [_PAULI[p] for p in pauli_string]
    full = mats[0]
    for m in mats[1:]:
        full = np.kron(full, m)
    return full


@dataclass
class Hamiltonian:
    terms: Sequence[tuple[float, str]]

    def __post_init__(self):
        lengths = {len(p) for _, p in self.terms}
        if len(lengths) != 1:
            raise ValueError("All Pauli strings must have the same length")
        self.n_qubits = lengths.pop()

    def matrix(self) -> np.ndarray:
        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for coeff, pstring in self.terms:
            H += coeff * pauli_string_to_matrix(pstring)
        return H

    def exact_ground_state_energy(self) -> float:
        eigvals = np.linalg.eigvalsh(self.matrix())
        return float(np.min(eigvals))


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    state = np.tensordot(gate, state, axes=([1], [qubit]))
    state = np.moveaxis(state, 0, qubit)
    return state.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    state = np.moveaxis(state, [control, target], [0, 1])
    out = state.copy()
    out[1, 0, ...] = state[1, 1, ...]
    out[1, 1, ...] = state[1, 0, ...]
    out = np.moveaxis(out, [0, 1], [control, target])
    return out.reshape(-1)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


@dataclass
class HardwareEfficientAnsatz:
    n_qubits: int
    depth: int = 2
    use_rz: bool = False

    @property
    def n_params(self) -> int:
        rots_per_layer = 2 if self.use_rz else 1
        return self.n_qubits * rots_per_layer * (self.depth + 1)

    def state(self, params: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0

        idx = 0
        rots_per_layer = 2 if self.use_rz else 1

        def rotation_layer(state):
            nonlocal idx
            for q in range(n):
                state = _apply_single_qubit_gate(state, ry(params[idx]), q, n)
                idx += 1
                if self.use_rz:
                    state = _apply_single_qubit_gate(state, rz(params[idx]), q, n)
                    idx += 1
            return state

        def entangling_layer(state):
            for q in range(n - 1):
                state = _apply_cnot(state, q, q + 1, n)
            return state

        for layer in range(self.depth):
            state = rotation_layer(state)
            if n > 1:
                state = entangling_layer(state)
        state = rotation_layer(state)
        return state


def expectation_value(state: np.ndarray, hamiltonian: Hamiltonian) -> float:
    n = hamiltonian.n_qubits
    total = 0.0 + 0.0j
    for coeff, pstring in hamiltonian.terms:
        pstate = state
        for q, p in enumerate(pstring):
            if p == "I":
                continue
            pstate = _apply_single_qubit_gate(pstate, _PAULI[p], q, n)
        total += coeff * np.vdot(state, pstate)
    return float(total.real)


@dataclass
class VQEResult:
    optimal_energy: float
    optimal_params: np.ndarray
    energy_history: list = field(default_factory=list)
    n_evaluations: int = 0


class VQE:
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        ansatz: HardwareEfficientAnsatz,
        optimizer: str = "COBYLA",
        maxiter: int = 500,
    ):
        if hamiltonian.n_qubits != ansatz.n_qubits:
            raise ValueError("Hamiltonian and ansatz qubit counts must match")
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.maxiter = maxiter

    def _cost(self, params: np.ndarray, history: list) -> float:
        state = self.ansatz.state(params)
        energy = expectation_value(state, self.hamiltonian)
        history.append(energy)
        return energy

    def run(self, initial_params: np.ndarray | None = None, seed: int | None = None) -> VQEResult:
        rng = np.random.default_rng(seed)
        if initial_params is None:
            initial_params = rng.uniform(0, 2 * np.pi, size=self.ansatz.n_params)

        history: list = []
        result = minimize(
            self._cost,
            initial_params,
            args=(history,),
            method=self.optimizer,
            options={"maxiter": self.maxiter},
        )

        return VQEResult(
            optimal_energy=result.fun,
            optimal_params=result.x,
            energy_history=history,
            n_evaluations=len(history),
        )

    def run_best_of(self, n_restarts: int = 5, seed: int = 0) -> VQEResult:
        best: VQEResult | None = None
        for i in range(n_restarts):
            res = self.run(seed=seed + i)
            if best is None or res.optimal_energy < best.optimal_energy:
                best = res
        return best


def h2_hamiltonian(bond_length: str = "0.735") -> Hamiltonian:
    if bond_length != "0.735":
        raise ValueError("Only the 0.735 Angstrom coefficient set is bundled")
    terms = [
        (-1.05237325, "II"),
        (0.39793742, "IZ"),
        (-0.39793742, "ZI"),
        (-0.01128010, "ZZ"),
        (0.18093119, "XX"),
    ]
    return Hamiltonian(terms)


def random_hamiltonian(n_qubits: int, n_terms: int, seed: int = 42) -> Hamiltonian:
    rng = np.random.default_rng(seed)
    paulis = ["I", "X", "Y", "Z"]
    terms = []
    seen = set()
    while len(terms) < n_terms:
        pstring = "".join(rng.choice(paulis) for _ in range(n_qubits))
        if pstring in seen:
            continue
        seen.add(pstring)
        coeff = rng.uniform(-1, 1)
        terms.append((coeff, pstring))
    return Hamiltonian(terms)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 70)
    print("VQE demo 1: H2 molecule (2-qubit tapered Hamiltonian, 0.735 A)")
    print("=" * 70)
    H = h2_hamiltonian()
    exact = H.exact_ground_state_energy()

    ansatz = HardwareEfficientAnsatz(n_qubits=H.n_qubits, depth=2, use_rz=True)
    vqe = VQE(H, ansatz, optimizer="COBYLA", maxiter=300)
    result = vqe.run_best_of(n_restarts=5, seed=0)

    print(f"Exact ground-state energy   : {exact:.8f} Ha")
    print(f"VQE   ground-state energy   : {result.optimal_energy:.8f} Ha")
    print(f"Absolute error              : {abs(result.optimal_energy - exact):.2e} Ha")
    print(f"Total cost-function calls   : {result.n_evaluations}")
    print(f"Optimal parameters          : {result.optimal_params}")

    print()
    print("=" * 70)
    print("VQE demo 2: random 3-qubit Hamiltonian (8 random Pauli terms)")
    print("=" * 70)
    H2_ = random_hamiltonian(n_qubits=3, n_terms=8, seed=7)
    exact2 = H2_.exact_ground_state_energy()

    ansatz2 = HardwareEfficientAnsatz(n_qubits=3, depth=3, use_rz=True)
    vqe2 = VQE(H2_, ansatz2, optimizer="COBYLA", maxiter=500)
    result2 = vqe2.run_best_of(n_restarts=6, seed=1)

    print(f"Exact ground-state energy   : {exact2:.8f}")
    print(f"VQE   ground-state energy   : {result2.optimal_energy:.8f}")
    print(f"Absolute error              : {abs(result2.optimal_energy - exact2):.2e}")
    print(f"Total cost-function calls   : {result2.n_evaluations}")
