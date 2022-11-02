from __future__ import annotations
from collections.abc import Collection, Iterable

import numpy as np
from numpy.typing import ArrayLike

from state import State, to_binary


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

    def permute(self, qubits: Collection[int]) -> Gate:
        assert len(qubits) == self.num_qubits
        assert set(qubits) == set(range(self.num_qubits))

        permutation_matrix = np.zeros_like(self.matrix)
        for i in range(2**self.num_qubits):
            i_binary = to_binary(i, digits=self.num_qubits)
            j_binary = permute_string(i_binary, qubits)
            j = int(j_binary, base=2)
            permutation_matrix[i, j] = 1

        return Gate(
            self.name,
            self.num_qubits,
            permutation_matrix.T.conj() @ self.matrix @ permutation_matrix,
            permute_string(self.circuit_symbol, qubits),
        )


def permute_string(string: str, new_indices: Iterable[int]) -> str:
    result = [""] * len(string)
    for old_index, new_index in enumerate(new_indices):
        result[new_index] = string[old_index]
    return "".join(result)
