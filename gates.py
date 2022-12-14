from gate import Gate

import numpy as np

H = Gate(name="H", num_qubits=1, matrix=[[1, 1], [1, -1]])

I = Gate(name="I", num_qubits=1, matrix=[[1, 0], [0, 1]], circuit_symbol="│")

X = Gate(name="X", num_qubits=1, matrix=[[0, 1], [1, 0]])
CX = X.control()
CCX = CX.control()

Y = Gate(name="Y", num_qubits=1, matrix=[[0, -1j], [1j, 0]])
CY = Y.control()
CCY = CY.control()

Z = Gate(name="Z", num_qubits=1, matrix=[[1, 0], [0, -1]])
CZ = Z.control()
CCZ = CZ.control()


def Rx(angle: float) -> Gate:
    return Gate(
        name=f"Rx({angle:.2f})",
        num_qubits=1,
        matrix=np.cos(angle / 2) * I.matrix - 1j * np.sin(angle / 2) * X.matrix,
    )


def Ry(angle: float) -> Gate:
    return Gate(
        name=f"Ry({angle:.2f})",
        num_qubits=1,
        matrix=np.cos(angle / 2) * I.matrix - 1j * np.sin(angle / 2) * Y.matrix,
    )


def Rz(angle: float) -> Gate:
    return Gate(
        name=f"Rz({angle:.2f})",
        num_qubits=1,
        matrix=np.cos(angle / 2) * I.matrix - 1j * np.sin(angle / 2) * Z.matrix,
    )


def P(angle: float) -> Gate:
    return Gate(
        name=f"P({angle:.2f})",
        num_qubits=1,
        matrix=[[1, 0], [0, np.exp(1j * angle)]],
    )


S = P(np.pi / 2).named("S")
T = P(np.pi / 4).named("T")

Swap = Gate(
    name="Swap",
    num_qubits=2,
    matrix=[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
)
