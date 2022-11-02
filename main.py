from circuit import Circuit
from gate import Gate


H = Gate(name="H", num_qubits=1, matrix=[[1, 1], [1, -1]])
HHH = H.tensor_product(H).tensor_product(H)

X = Gate(name="X", num_qubits=1, matrix=[[0, 1], [1, 0]])
XXX = X.tensor_product(X).tensor_product(X)

Z = Gate(name="Z", num_qubits=1, matrix=[[1, 0], [0, -1]])
CCZ = Z.control().control()

circuit = Circuit("Grover", 3)

circuit.add(HHH, range(3))

circuit.add(X, [1])
circuit.add(CCZ, range(3))
circuit.add(X, [1])

circuit.add(HHH, range(3))
circuit.add(XXX, range(3))
circuit.add(CCZ, range(3))
circuit.add(XXX, range(3))
circuit.add(HHH, range(3))

state = circuit.run()

print("Circuit:")
print(circuit)
print("\nFinal State:")
print(state)
print("\nProbability Distribution:")
print(state.histogram())
