from __future__ import annotations
from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

epsilon = 1e-12


class State:
    def __init__(
        self,
        num_qubits: int,
        vector: ArrayLike | Mapping[str, complex],
    ) -> None:
        self.num_qubits = num_qubits

        if isinstance(vector, Mapping):
            self.vector = np.zeros([2**num_qubits], dtype=complex)
            for basis_state, coeff in vector.items():
                assert len(basis_state) == self.num_qubits
                i = int(basis_state, base=2)
                assert 0 <= i < 2**self.num_qubits
                self.vector[i] = coeff
        else:
            self.vector = np.asarray(vector, dtype=complex)
        assert self.vector.shape == (2**num_qubits,)
        self.vector /= np.linalg.norm(self.vector)

    def tensor_product(self, other: State) -> State:
        return State(
            self.num_qubits + other.num_qubits,
            np.kron(self.vector, other.vector),
        )

    def coefficients(self) -> dict[str, complex]:
        result = {}
        for i in range(2**self.num_qubits):
            if abs(self.vector[i]) > epsilon:
                result[to_binary(i, digits=self.num_qubits)] = self.vector[i]
        return result

    def __repr__(self) -> str:
        return " + ".join(
            f"{coeff:.3}|{basis_state}⟩"
            for basis_state, coeff in self.coefficients().items()
        )

    def probabilities(self) -> dict[str, float]:
        return {
            basis_state: abs(coeff) ** 2
            for basis_state, coeff in self.coefficients().items()
        }

    def histogram(self, max_length: int = 80) -> str:
        partial = " ▏▎▍▌▋▊▉"
        lines = []
        for basis_state, prob in self.probabilities().items():
            scaled = round(prob * max_length * len(partial))
            bar = "█" * (scaled // len(partial)) + partial[scaled % len(partial)]
            lines.append(f"|{basis_state}⟩ {bar}")
        return "\n".join(lines)

    def measure(self) -> str:
        probabilities = self.probabilities()
        basis_state = np.random.choice(
            list(probabilities.keys()),
            p=list(probabilities.values()),
        )
        return cast(str, basis_state)


zero = State(num_qubits=1, vector=[1, 0])
one = State(num_qubits=1, vector=[0, 1])


def to_binary(n: int, digits: int) -> str:
    return f"{n:0{digits}b}"
