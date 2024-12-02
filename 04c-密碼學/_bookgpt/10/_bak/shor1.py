# Shor's Algorithm Implementation
# Shor’s algorithm is famous for factoring integers in polynomial time.

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
from qiskit_aer import Aer

# Print the message confirming successful imports
print("Imports Successful")

# 1. The Problem: Period Finding
# Example of periodic function with N = 35, a = 3
N = 35
a = 3

# Calculate the plotting data
xvals = np.arange(35)
yvals = [np.mod(a**x, N) for x in xvals]

# Use matplotlib to display it nicely
fig, ax = plt.subplots()
ax.plot(xvals, yvals, linewidth=1, linestyle='dotted', marker='x')
ax.set(xlabel='$x$', ylabel=f'${a}^x$ mod ${N}$',
       title="Example of Periodic Function in Shor's Algorithm")

try: 
    # Plot r on the graph
    r = yvals[1:].index(1) + 1
    plt.annotate('', xy=(0, 1), xytext=(r, 1),
                 arrowprops=dict(arrowstyle='<->'))
    plt.annotate(f'$r={r}$', xy=(r / 3, 1.5))
except ValueError:
    print('Could not find period, check a < N and have no common factors.')

# 2. The Solution: Quantum Phase Estimation
def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    # print('a=', a)
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2,4,7,8,11 or 13")
    
    U = QuantumCircuit(4)
    for _iteration in range(power):
        if a in [2, 13]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [7, 8]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    
    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U

def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

# 3. Constructing the Circuit for Shor's Algorithm
def shors_algorithm(a, N_COUNT=8):
    """Construct and simulate Shor's algorithm."""
    # Create QuantumCircuit with N_COUNT counting qubits + 4 qubits for U to act on
    qc = QuantumCircuit(N_COUNT + 4, N_COUNT)

    # Initialize counting qubits in state |+>
    for q in range(N_COUNT):
        qc.h(q)

    # Auxiliary register in state |1>
    qc.x(N_COUNT)

    # Do controlled-U operations
    for q in range(N_COUNT):
        qc.append(c_amod15(a, 2 ** q),
                 [q] + [i + N_COUNT for i in range(4)])

    # Do inverse-QFT
    qc.append(qft_dagger(N_COUNT), range(N_COUNT))

    # Measure circuit
    qc.measure(range(N_COUNT), range(N_COUNT))
    qc.draw(fold=-1)  # -1 means 'do not fold'
    return qc

# 4. Simulating the Circuit and Plotting Results
def simulate_circuit(qc):
    """Simulate the quantum circuit and plot the results."""
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, aer_sim)
    counts = aer_sim.run(t_qc).result().get_counts()
    plot_histogram(counts)
    return counts

# 5. Analyze Phase Measurement
def analyze_phases(counts, N_COUNT=8):
    """Analyze the phase measurements and estimate the period."""
    rows, measured_phases = [], []
    for output in counts:
        decimal = int(output, 2)  # Convert (base 2) string to decimal
        phase = decimal / (2 ** N_COUNT)  # Find corresponding eigenvalue
        measured_phases.append(phase)
        # Add these values to the rows in our table:
        rows.append([f"{output}(bin) = {decimal:>3}(dec)",
                     f"{decimal}/{2 ** N_COUNT} = {phase:.2f}"])

    # Print the rows in a table
    headers = ["Register Output", "Phase"]
    df = pd.DataFrame(rows, columns=headers)
    print(df)
    return measured_phases

# 6. Continued Fractions to Find Period
def continued_fractions_for_period(measured_phases, N):
    """Use continued fractions to estimate period."""
    rows = []
    for phase in measured_phases:
        frac = Fraction(phase).limit_denominator(N)
        rows.append([phase,
                     f"{frac.numerator}/{frac.denominator}",
                     frac.denominator])
    # Print as a table
    headers = ["Phase", "Fraction", "Guess for r"]
    df = pd.DataFrame(rows, columns=headers)
    print(df)

# Main execution flow
a = 7
qc = shors_algorithm(a)
counts = simulate_circuit(qc)
measured_phases = analyze_phases(counts)
continued_fractions_for_period(measured_phases, 15)

# 7. Repeated Squaring for Modular Exponentiation
def a2jmodN(a, j, N):
    """Compute a^{2^j} (mod N) by repeated squaring."""
    for _ in range(j):
        a = np.mod(a**2, N)
    return a

# 8. Quantum Phase Estimation for a mod 15
def qpe_amod15(a):
    """Performs quantum phase estimation on the operation a*r mod 15."""
    N_COUNT = 8
    qc = QuantumCircuit(4 + N_COUNT, N_COUNT)
    for q in range(N_COUNT):
        qc.h(q)  # Initialize counting qubits in state |+>
    qc.x(N_COUNT)  # Auxiliary register in state |1>

    for q in range(N_COUNT):  # Do controlled-U operations
        qc.append(c_amod15(a, 2 ** q),
                 [q] + [i + N_COUNT for i in range(4)])

    qc.append(qft_dagger(N_COUNT), range(N_COUNT))  # Do inverse-QFT
    qc.measure(range(N_COUNT), range(N_COUNT))

    # Simulate Results
    aer_sim = Aer.get_backend('aer_simulator')
    job = aer_sim.run(transpile(qc, aer_sim), shots=1, memory=True)
    readings = job.result().get_memory()
    print("Register Reading: " + readings[0])
    phase = int(readings[0], 2) / (2 ** N_COUNT)
    print(f"Corresponding Phase: {phase}")
    return phase

# Factoring using Shor's Algorithm
def factor_using_shors(a, N):
    """Attempt to find a factor of N using Shor's Algorithm."""
    FACTOR_FOUND = False
    ATTEMPT = 0
    while not FACTOR_FOUND:
        ATTEMPT += 1
        print(f"\nATTEMPT {ATTEMPT}:")
        phase = qpe_amod15(a)  # Phase = s/r
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        print(f"Result: r = {r}")
        if phase != 0:
            # Guesses for factors are gcd(x^{r/2} ±1 , 15)
            guesses = [gcd(a**(r // 2) - 1, N), gcd(a**(r // 2) + 1, N)]
            print(f"Guessed Factors: {guesses}")

# Example to factor 15
N = 15
a = randint(2, N)
print(f"Random a: {a}")
factor_using_shors(a, N)
