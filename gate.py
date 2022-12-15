from __future__ import annotations
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from state import State


class Gate:
    """A quantum gate.

    Qubits are numbered from right to left, so qubit 0 is the rightmost digit in
    the binary representation of a basis state, and the rightmost wire in a
    vertical circuit diagram.

    Attributes:
        name: Name of the gate.
        num_qubits: Number of qubits.
        matrix: Unitary matrix of shape (2**num_qubits, 2**num_qubits).
        circuit_symbol: String of length num_qubits used to represent the gate
            in circuit diagrams. Each character represents a vertical wire.
    """

    def __init__(
        self,
        name: str,
        num_qubits: int,
        matrix: ArrayLike,
        circuit_symbol: str = "",
    ) -> None:
        """Initializes the gate.

        Args:
            name: Name of the gate.
            num_qubits: Number of qubits.
            matrix: Unitary matrix of shape (2**num_qubits, 2**num_qubits). Need
                not be normalized.
            circuit_symbol: String of length num_qubits used to represent the
                gate in circuit diagrams, or empty for the default ╪, which is
                meant to resemble to the box used in conventional circuit
                diagrams. Each character represents a vertical wire.
        """
        self.name = name
        self.num_qubits = num_qubits
        self.matrix = np.asarray(matrix, dtype=complex)
        assert self.matrix.shape == (2**num_qubits, 2**num_qubits)
        self.matrix /= np.sqrt(abs(np.linalg.det(self.matrix)))
        self.circuit_symbol = circuit_symbol or "╪" * num_qubits
        assert len(self.circuit_symbol) == num_qubits

    def named(self, name: str) -> Gate:
        """Returns a copy of the gate with a new name."""
        return Gate(name, self.num_qubits, self.matrix, self.circuit_symbol)

    def __repr__(self) -> str:
        """Returns a string representation of the gate as a vertical circuit diagram."""
        return f"{self.circuit_symbol} {self.name}"

    def apply(self, state: State) -> State:
        """Returns the result of applying the gate to a quantum register."""
        assert state.num_qubits == self.num_qubits
        return State(self.num_qubits, self.matrix @ state.vector)

    def inverse(self) -> Gate:
        """Returns the inverse of the gate."""
        return Gate(
            f"{self.name}†",
            self.num_qubits,
            self.matrix.T.conj(),
            self.circuit_symbol,
        )

    def control(self) -> Gate:
        """Returns a controlled version of the gate.

        The new leftmost qubit is the control, and is represented by ┿ in
        vertical circuit diagrams. This is meant to resemble the filled dot with
        a line connecting it to the box, used in conventional circuit diagrams.
        """
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

    def __matmul__(self, other: Gate) -> Gate:
        """Returns the composed gate that applies other then self."""
        assert self.num_qubits == other.num_qubits
        return Gate(
            f"{self.name} {other.name}",
            self.num_qubits,
            self.matrix @ other.matrix,
        )

    def tensor_product(self, other: Gate) -> Gate:
        """Returns a gate that applies self to qubits on the left and other to qubits on the right."""
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
        """Permutes the qubits that this gate operates on.

        Define inner index as the index that this gate assigns to a qubit, and
        outer index as the index used by the new gate, or the outer circuit.

        Args:
            permutation: Sequence of length num_qubits, ordered from right to
                left by inner index, where each element gives the outer index of
                the corresponding qubit.

        Returns:
            A new gate that maps each qubit from its outer index to its inner
            index, applies this gate, and maps back to the outer index.
        """
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
