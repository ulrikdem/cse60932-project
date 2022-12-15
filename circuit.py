from __future__ import annotations
from collections.abc import Collection

import numpy as np

from gate import Gate
from state import State


class Circuit(Gate):
    """A quantum circuit, which is a gate composed of other gates.

    Qubits are numbered from right to left, so qubit 0 is the rightmost digit in
    the binary representation of a basis state, and the rightmost wire in a
    vertical circuit diagram.

    Attributes:
        name: Name of the circuit.
        num_qubits: Number of qubits.
        matrix: Unitary matrix of shape (2**num_qubits, 2**num_qubits).
        gates: List of child quantum gates.
    """

    def __init__(self, name: str, num_qubits: int) -> None:
        """Initializes the circuit without any gates.

        Args:
            name: Name of the circuit.
            num_qubits: Number of qubits.
        """
        super().__init__(name, num_qubits, np.eye(2**num_qubits))
        self.gates: list[Gate] = []

    def __repr__(self) -> str:
        """Returns a string representation of the circuit as a vertical circuit diagram."""
        return "\n".join(map(repr, self.gates))

    def add(self, gate: Gate, qubits: Collection[int] = []) -> None:
        """Adds a gate after all previous gates.

        Args:
            gate: Gate to add.
            qubits: Sequence of qubit indices to use for the gate, or empty to
                use all qubits. Length must match the gate's number of qubits.
        """
        qubits = qubits or list(reversed(range(self.num_qubits)))
        assert gate.num_qubits <= self.num_qubits
        assert len(qubits) == gate.num_qubits
        assert len(set(qubits)) == gate.num_qubits
        assert all(0 <= i < self.num_qubits for i in qubits)

        extra_qubits = self.num_qubits - gate.num_qubits
        gate = gate.tensor_product(
            Gate("", extra_qubits, np.eye(2**extra_qubits), "│" * extra_qubits),
        )
        permutation = list(qubits) + list(set(range(self.num_qubits)) - set(qubits))
        gate = gate.permute(permutation)

        self.matrix = gate.matrix @ self.matrix
        self.gates.append(gate)

    def add_subcircuit(self, subcircuit: Circuit, qubits: Collection[int] = []) -> None:
        """Adds the gates from a subcircuit to this circuit.

        This is different from add(), which adds the whole subcircuit as a single gate.

        Args:
            subcircuit: Circuit containing the gates to add.
            qubits: Sequence of qubit indices to use for the subcircuit, or
                empty to use all qubits. Length must match the subcircuit's
                number of qubits.
        """
        for gate in subcircuit.gates:
            self.add(gate, qubits)

    def run(self, state: State | None = None) -> State:
        """Applies the circuit to a quantum register.

        Args:
            state: Initial state of the register, or None for the zero state.

        Returns:
            The final state of the register.
        """
        state = state or State(self.num_qubits, {"0" * self.num_qubits: 1})
        return self.apply(state)

    def inverse(self) -> Circuit:
        """Returns the inverse of the circuit."""
        circuit = Circuit(f"{self.name}†", self.num_qubits)
        for gate in reversed(self.gates):
            circuit.add(gate.inverse())
        return circuit
