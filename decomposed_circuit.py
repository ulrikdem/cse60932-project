from circuit import Circuit
from gate import Gate
from gates import X
from state import epsilon

import numpy as np
from numpy.typing import NDArray


class DecomposedCircuit(Circuit):
    def __init__(self, gate: Gate, name: str | None = None) -> None:
        super().__init__(gate.name if name is None else name, gate.num_qubits)
        adjoint_matrix = gate.matrix.T.conj()

        for i in range(2**self.num_qubits - 2):
            for j in range(i + 1, 2**self.num_qubits):
                a = adjoint_matrix[i, i]
                b = adjoint_matrix[j, i]
                if abs(b) > epsilon:
                    norm = np.sqrt(abs(a) ** 2 + abs(b) ** 2)

                    matrix = np.eye(2**self.num_qubits, dtype=complex)
                    matrix[i, i] = a.conj() / norm
                    matrix[i, j] = b.conj() / norm
                    matrix[j, i] = b / norm
                    matrix[j, j] = -a / norm

                    self._decompose_two_level_matrix(matrix, i, j)
                    adjoint_matrix = matrix @ adjoint_matrix

        self._decompose_two_level_matrix(
            matrix=adjoint_matrix.T.conj(),
            basis_state1=2**self.num_qubits - 2,
            basis_state2=2**self.num_qubits - 1,
        )

        assert abs(self.matrix - gate.matrix).max() < epsilon

    def _decompose_two_level_matrix(
        self,
        matrix: NDArray[np.complex_],
        basis_state1: int,
        basis_state2: int,
    ) -> None:
        matrix = np.array(
            [
                [
                    matrix[basis_state1, basis_state1],
                    matrix[basis_state1, basis_state2],
                ],
                [
                    matrix[basis_state2, basis_state1],
                    matrix[basis_state2, basis_state2],
                ],
            ]
        )
        gate = Gate(
            name=f"({matrix[0, 0]:.3}|0⟩⟨0| + {matrix[0, 1]:.3}|0⟩⟨1| + {matrix[1, 0]:.3}|1⟩⟨0| + {matrix[1, 1]:.3}|1⟩⟨1|)",
            num_qubits=1,
            matrix=matrix,
        )

        subcircuit = Circuit("", self.num_qubits)
        difference = basis_state1 ^ basis_state2
        while difference.bit_count() > 1:
            least_significant_difference = difference & -difference
            target_qubit = least_significant_difference.bit_length() - 1
            self._decompose_control_gate(X, target_qubit, basis_state1, subcircuit)
            basis_state1 ^= least_significant_difference
            difference ^= least_significant_difference
        target_qubit = difference.bit_length() - 1

        self.add_subcircuit(subcircuit)
        self._decompose_control_gate(gate, target_qubit, basis_state1, self)
        self.add_subcircuit(subcircuit.inverse())

    def _decompose_control_gate(
        self,
        gate: Gate,
        target_qubit: int,
        control_mask: int,
        circuit: Circuit,
    ) -> None:
        while gate.num_qubits < self.num_qubits:
            gate = gate.control()

        subcircuit = Circuit("", self.num_qubits)
        for i in range(self.num_qubits):
            if i != target_qubit and (control_mask & 2**i) == 0:
                subcircuit.add(X, [i])

        circuit.add_subcircuit(subcircuit)
        circuit.add(
            gate,
            [i for i in range(self.num_qubits) if i != target_qubit] + [target_qubit],
        )
        circuit.add_subcircuit(subcircuit.inverse())
