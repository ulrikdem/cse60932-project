from circuit import Circuit
from decomposed_circuit import DecomposedCircuit
from gates import CCZ, H, X

HHH = H.tensor_product(H).tensor_product(H)
XXX = X.tensor_product(X).tensor_product(X)

circuit = Circuit("Grover", num_qubits=3)

circuit.add(HHH)

circuit.add(X, [1])
circuit.add(CCZ)
circuit.add(X, [1])

circuit.add(HHH)
circuit.add(XXX)
circuit.add(CCZ)
circuit.add(XXX)
circuit.add(HHH)

state = circuit.run()

print("Circuit:")
print(circuit)
print("\nFinal State:")
print(state)
print("\nProbability Distribution:")
print(state.histogram())

decomposed_circuit = DecomposedCircuit(circuit)
state = decomposed_circuit.run()

print("\nDecomposed Circuit:")
print(decomposed_circuit)
print("\nFinal State:")
print(state)
print("\nProbability Distribution:")
print(state.histogram())
