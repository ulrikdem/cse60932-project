from circuit import Circuit
from gate import Gate
from gates import CX, H, P, Ry, Rz, S, T, X
from state import epsilon

import numpy as np
from numpy.typing import NDArray


class DecomposedCircuit(Circuit):
    """A decomposition of a unitary matrix to CNOT and single-qubit gates.

    This is based on the decomposition described in Nielsen and Chuang's
    "Quantum Computation and Quantum Information". Section, figure and theorem
    numbers refer to the 10th Anniversary Edition.

    Attributes:
        name: Name of the circuit.
        num_qubits: Number of qubits.
        num_ancillas: Number of ancilla qubits added on the left of the register.
        matrix: Unitary matrix of shape (2**num_qubits, 2**num_qubits).
        gates: List of child quantum gates.
    """
    def __init__(self, gate: Gate, name: str | None = None) -> None:
        """Decomposes an arbitrary unitary matrix to CNOT and single-qubit gates.

        This function is based on Section 4.5.1 of Nielsen and Chuang.

        Args:
            gate: Gate to decompose.
            name: Name of the circuit, or None to copy from gate.
        """
        self.num_ancillas = max(gate.num_qubits - 2, 0)
        super().__init__(
            gate.name if name is None else name,
            gate.num_qubits + self.num_ancillas,
        )
        adjoint_matrix = gate.matrix.T.conj()

        if gate.num_qubits < 2:
            self.add(gate)
            return

        for i in range(2**gate.num_qubits - 2):
            for j in range(i + 1, 2**gate.num_qubits):
                a = adjoint_matrix[i, i]
                b = adjoint_matrix[j, i]
                if abs(b) > epsilon:
                    norm = np.sqrt(abs(a) ** 2 + abs(b) ** 2)

                    matrix = np.eye(2**gate.num_qubits, dtype=complex)
                    matrix[i, i] = a.conj() / norm
                    matrix[i, j] = b.conj() / norm
                    matrix[j, i] = b / norm
                    matrix[j, j] = -a / norm

                    self._decompose_two_level_matrix(matrix, i, j)
                    adjoint_matrix = matrix @ adjoint_matrix

        self._decompose_two_level_matrix(
            matrix=adjoint_matrix.T.conj(),
            basis_state1=2**gate.num_qubits - 2,
            basis_state2=2**gate.num_qubits - 1,
        )

        assert approx_equal(
            self.matrix[: 2**gate.num_qubits, : 2**gate.num_qubits],
            gate.matrix,
            epsilon=1e-4,
        )

    def _decompose_two_level_matrix(
        self,
        matrix: NDArray[np.complex_],
        basis_state1: int,
        basis_state2: int,
    ) -> None:
        """Decomposes a two-level matrix.

        This function is based on Section 4.5.2 of Nielsen and Chuang.

        Args:
            matrix: Two-level matrix to decompose.
            basis_state1: Index of the first vector component on which the
                matrix acts non-trivially.
            basis_state2: Index of the second vector component on which the
                matrix acts non-trivially.
        """
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

        subcircuit = Circuit("", self.num_qubits)
        difference = basis_state1 ^ basis_state2
        while difference.bit_count() > 1:
            least_significant_difference = difference & -difference
            target_qubit = least_significant_difference.bit_length() - 1
            self._decompose_multi_control_gate(
                X.matrix,
                basis_state1,
                target_qubit,
                subcircuit,
            )
            basis_state1 ^= least_significant_difference
            difference ^= least_significant_difference
        target_qubit = difference.bit_length() - 1

        self.add_subcircuit(subcircuit)
        self._decompose_multi_control_gate(matrix, basis_state1, target_qubit, self)
        self.add_subcircuit(subcircuit.inverse())

    def _decompose_multi_control_gate(
        self,
        matrix: NDArray[np.complex_],
        control_mask: int,
        target_qubit: int,
        circuit: Circuit,
    ) -> None:
        """Decomposes a multi-controlled gate.

        This function is based on Figures 4.10 and 4.11 of Nielsen and Chuang.

        Args:
            matrix: 2x2 unitary matrix of the (non-controlled) gate.
            control_mask: Bitmask with bits set for negative controls.
            target_qubit: Index of the target qubit.
            circuit: Circuit to add the decomposed gates to.
        """
        control_qubits = []
        subcircuit = Circuit("", self.num_qubits)

        for i in range(self.num_qubits - self.num_ancillas):
            if i != target_qubit:
                control_qubits.append(i)
                if (control_mask & 2**i) == 0:
                    subcircuit.add(X, [i])

        accumulator = control_qubits[0]
        ancilla = self.num_qubits - self.num_ancillas
        for i in control_qubits[1:]:
            _decompose_toffoli_gate(accumulator, i, ancilla, subcircuit)
            accumulator = ancilla
            ancilla += 1

        circuit.add_subcircuit(subcircuit)
        _decompose_control_gate(matrix, accumulator, target_qubit, circuit)
        circuit.add_subcircuit(subcircuit.inverse())


def _decompose_toffoli_gate(
    control1: int,
    control2: int,
    target: int,
    circuit: Circuit,
) -> None:
    """Decomposes a Toffoli gate.

    This function is based on Figure 4.9 of Nielsen and Chuang.

    Args:
        control1: Index of the first control qubit.
        control2: Index of the second control qubit.
        target: Index of the target qubit.
        circuit: Circuit to add the decomposed gates to.
    """
    circuit.add(H, [target])
    circuit.add(CX, [control2, target])
    circuit.add(T.inverse(), [target])
    circuit.add(CX, [control1, target])
    circuit.add(T, [target])
    circuit.add(CX, [control2, target])
    circuit.add(T.inverse(), [target])
    circuit.add(CX, [control1, target])
    circuit.add(T, [target])
    circuit.add(H, [target])
    circuit.add(T.inverse(), [control2])
    circuit.add(CX, [control1, control2])
    circuit.add(T.inverse(), [control2])
    circuit.add(CX, [control1, control2])
    circuit.add(T, [control1])
    circuit.add(S, [control2])


def _decompose_control_gate(
    matrix: NDArray[np.complex_],
    control: int,
    target: int,
    circuit: Circuit,
) -> None:
    """Decomposes a controlled gate.

    This function is based on Theorem 4.1, Corollary 4.2 and Figure 4.6 of
    Nielsen and Chuang.

    Args:
        matrix: 2x2 unitary matrix of the (non-controlled) gate.
        control: Index of the control qubit.
        target: Index of the target qubit.
        circuit: Circuit to add the decomposed gates to.
    """
    if approx_equal(matrix, X.matrix):
        circuit.add(CX, [control, target])
        return

    angles = np.angle(matrix)
    gamma = np.arccos(min(abs(matrix[0, 0]), 1)) * 2
    if abs(gamma - np.pi) > epsilon:
        alpha = (angles[1, 1] + angles[0, 0]) / 2
        beta = angles[1, 0] - angles[0, 0]
        delta = angles[1, 1] - angles[1, 0]
    else:
        alpha = (angles[1, 0] + angles[0, 1] - np.pi) / 2
        beta = angles[1, 0] - angles[0, 1] + np.pi
        delta = 0

    if not approx_equal(
        matrix,
        np.exp(1j * alpha) * (Rz(beta) @ Ry(gamma) @ Rz(delta)).matrix,
        epsilon=1e-6,
    ):
        print(alpha, beta, delta, gamma)
        print(matrix)
        print(np.exp(1j * alpha) * (Rz(beta) @ Ry(gamma) @ Rz(delta)).matrix)

    A = Rz(beta) @ Ry(gamma / 2)
    B = Ry(-gamma / 2) @ Rz(-(delta + beta) / 2)
    C = Rz((delta - beta) / 2)

    circuit.add(C, [target])
    circuit.add(CX, [control, target])
    circuit.add(B, [target])
    circuit.add(CX, [control, target])
    circuit.add(A, [target])
    circuit.add(P(alpha), [control])


def approx_equal(
    matrix1: NDArray[np.complex_],
    matrix2: NDArray[np.complex_],
    epsilon: float = epsilon,
) -> np.bool_:
    """Returns whether two matrices have an absolute difference less than epsilon."""
    return (abs(matrix1 - matrix2) < epsilon).all()
