from decomposed_circuit import DecomposedCircuit
from gate import Gate
from gates import Ry, Rz

import numpy as np

# Iterate over (a sample of) all possible single-qubit unitary matrices
n = 15
for alpha in np.linspace(0, 2 * np.pi, n):
    for beta in np.linspace(0, 2 * np.pi, n):
        for delta in np.linspace(0, 2 * np.pi, n):
            for gamma in np.linspace(0, 2 * np.pi, n):
                matrix = np.exp(1j * alpha) * (Rz(beta) @ Ry(gamma) @ Rz(delta)).matrix

                # DecomposedCircuit() asserts that the final matrix matches the given one
                DecomposedCircuit(Gate(name="U", num_qubits=1, matrix=matrix))
