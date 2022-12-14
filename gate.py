from __future__ import annotations
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from state import State


class Gate:
    def __init__(
        self,
        name: str,
        num_qubits: int,
        matrix: ArrayLike,
        circuit_symbol: str = "",
    ) -> None:
        self.name = name
        self.num_qubits = num_qubits
        self.matrix = np.asarray(matrix, dtype=complex)
        assert self.matrix.shape == (2**num_qubits, 2**num_qubits)
        self.matrix /= np.sqrt(abs(np.linalg.det(self.matrix)))
        self.circuit_symbol = circuit_symbol or "╪" * num_qubits
        assert len(self.circuit_symbol) == num_qubits

    def named(self, name: str) -> Gate:
        return Gate(name, self.num_qubits, self.matrix, self.circuit_symbol)

    def __repr__(self) -> str:
        return f"{self.circuit_symbol} {self.name}"

    def apply(self, state: State) -> State:
        assert state.num_qubits == self.num_qubits
        return State(self.num_qubits, self.matrix @ state.vector)

    def inverse(self) -> Gate:
        return Gate(
            f"{self.name}†",
            self.num_qubits,
            self.matrix.T.conj(),
            self.circuit_symbol,
        )

    def control(self) -> Gate:
        return Gate(
            f"C{self.name}",
            self.num_qubits + 1,
            np.block(
                [
                    [np.eye(2**self.num_qubits), np.zeros_like(self.matrix)],
                    [np.zeros_like(self.matrix), self.matrix],
                ]
            ),
            "┿" + self.circuit_symbol,
        )

    def tensor_product(self, other: Gate) -> Gate:
        if self.name and other.name:
            name = f"{self.name} ⊗ {other.name}"
        else:
            name = self.name or other.name

        return Gate(
            name,
            self.num_qubits + other.num_qubits,
            np.kron(self.matrix, other.matrix),
            self.circuit_symbol + other.circuit_symbol,
        )

    def permute(self, permutation: Sequence[int]) -> Gate:
        assert len(permutation) == self.num_qubits
        assert set(permutation) == set(range(self.num_qubits))

        permutation_matrix = np.zeros_like(self.matrix)
        for old_basis_state in range(2**self.num_qubits):
            new_basis_state = 0
            for i, j in enumerate(reversed(permutation)):
                new_basis_state |= (old_basis_state >> j & 1) << i
            permutation_matrix[new_basis_state, old_basis_state] = 1

        circuit_symbols = [""] * self.num_qubits
        for i, j in enumerate(permutation):
            circuit_symbols[-1 - j] = self.circuit_symbol[i]

        return Gate(
            self.name,
            self.num_qubits,
            permutation_matrix.T @ self.matrix @ permutation_matrix,
            "".join(circuit_symbols),
        )
