from __future__ import annotations
from collections.abc import Collection

import numpy as np

from gate import Gate
from state import State


class Circuit(Gate):
    def __init__(self, name: str, num_qubits: int) -> None:
        super().__init__(name, num_qubits, np.eye(2**num_qubits))
        self.gates: list[Gate] = []

    def __repr__(self) -> str:
        return "\n".join(map(repr, self.gates))

    def add(self, gate: Gate, qubits: Collection[int] = []) -> None:
        qubits = qubits or range(self.num_qubits)
        assert gate.num_qubits <= self.num_qubits
        assert len(qubits) == gate.num_qubits
        assert len(set(qubits)) == gate.num_qubits
        assert all(0 <= i < self.num_qubits for i in qubits)

        extra_qubits = self.num_qubits - gate.num_qubits
        gate = gate.tensor_product(
            Gate("", extra_qubits, np.eye(2**extra_qubits), "│" * extra_qubits),
        )
        qubits = list(qubits) + list(set(range(self.num_qubits)) - set(qubits))
        gate = gate.permute(qubits)
        self.matrix = gate.matrix @ self.matrix
        self.gates.append(gate)

    def add_subcircuit(self, subcircuit: Circuit, qubits: Collection[int] = []) -> None:
        for gate in subcircuit.gates:
            self.add(gate, qubits)

    def run(self, state: State | None = None) -> State:
        state = state or State(self.num_qubits, {"0" * self.num_qubits: 1})
        return self.apply(state)

    def inverse(self) -> Circuit:
        circuit = Circuit(f"{self.name}†", self.num_qubits)
        for gate in reversed(self.gates):
            circuit.add(gate.inverse())
        return circuit
