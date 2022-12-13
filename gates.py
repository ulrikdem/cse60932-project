from gate import Gate

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
