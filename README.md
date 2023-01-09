# CSE 60932 Project

The goal of this project was to create a tool that can convert a quantum circuit to a unitary matrix, and vice versa (decomposing the matrix to a circuit of CNOT and single-qubit gates).
It also acts as a quantum circuit simulator, which is a trivial matrix-vector multiplication after the circuit-to-matrix conversion.
See the [presentation slides](./presentation.pdf) for more information.

The project is implemented in Python and depends on NumPy.
It consists of the following modules:

- [`state.py`](state.py):
  Defines the `State` class, which wraps a complex vector to represent the state of a quantum register.
  It can be converted to or from a sparse dictionary representation, and to a probability distribution that can be used to print a histogram or randomly sampled to simulate a measurement.

- [`gate.py`](gate.py):
  Defines the `Gate` class, which wraps a unitary matrix.
  It has methods to create an inverted or controlled gate, the composition or tensor product of two gates, or a version of the gate with its inputs permuted.
  It can also apply itself to a `State`.

- [`gates.py`](gates.py):
  Contains definitions of some common `Gate`s.

- [`circuit.py`](circuit.py):
  Defines the `Circuit` class, which is a `Gate` composed of other `Gate`s.
  It can be represented as a rudimentary vertical circuit diagram, with vertical qubit wires numbered from right to left.
  Gates are indicated by two horizontal lines (`╪`) by default, meant to resemble opposite sides of the box used in conventional circuit diagrams.
  Control qubits are indicated by a single thick horizontal line (`┿`), meant to resemble the filled circle and line connecting it to the box.
  The name of the gate is shown to the right of the circuit.
  The following represents a circuit for Grover's algorithm:
  ```
  ╪╪╪ H ⊗ H ⊗ H
  │╪│ X
  ┿┿╪ CCZ
  │╪│ X
  ╪╪╪ H ⊗ H ⊗ H
  ╪╪╪ X ⊗ X ⊗ X
  ┿┿╪ CCZ
  ╪╪╪ X ⊗ X ⊗ X
  ╪╪╪ H ⊗ H ⊗ H
  ```

- [`decomposed_circuit.py`](decomposed_circuit.py):
  Defines the `DecomposedCircuit` class, which can decompose an arbitrary unitary matrix to CNOT and single-qubit gates, following sections 4.2, 4.3 and 4.5 of Nielsen and Chuang's "Quantum Computation and Quantum Information".

- [`main.py`](main.py):
  An example program that constructs a circuit (the Grover's algorithm circuit shown above), shows the result of simulating it, and decomposes it.

- [`test.py`](test.py):
  Samples the entire space of single-qubit matrices at regular intervals, and tests that the matrix of the decomposed circuit has an absolute difference of at most `1e-4` in each element.
