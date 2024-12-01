import random
import itertools
import math
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister #, execute
from qiskit_aer import Aer

# Classical step to find the period of x modulo N
def find_period_classical(x, N):
    n = 1
    t = x
    while t != 1:
        t *= x
        t %= N
        n += 1
    return n

# Sieve of Eratosthenes to generate prime numbers up to n
def sieve():
    D = {}
    yield 2
    for q in itertools.islice(itertools.count(3), 0, None, 2):
        p = D.pop(q, None)
        if p is None:
            D[q * q] = q
            yield q
        else:
            x = p + q
            while x in D or not (x & 1):
                x += p
            D[x] = p

# Create a list of prime numbers up to the given n
def get_primes_sieve(n):
    return list(itertools.takewhile(lambda p: p < n, sieve()))

# Generate a semiprime (product of two primes)
def get_semiprime(n):
    primes = get_primes_sieve(n)
    p = primes[random.randrange(len(primes))]
    q = primes[random.randrange(len(primes))]
    return p * q

# Function to implement Shor's algorithm classically
def shors_algorithm_classical(N):
    x = random.randint(1, N)
    if math.gcd(x, N) != 1:  # Step 1
        return x, 0, math.gcd(x, N), N // math.gcd(x, N)
    r = find_period_classical(x, N)  # Step 2
    while r % 2 != 0:  # Ensure the period is even
        r = find_period_classical(x, N)
    p = math.gcd(x**(r // 2) + 1, N)  # Step 3
    q = math.gcd(x**(r // 2) - 1, N)
    return x, r, p, q

# Quantum Fourier Transform
def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi / float(2**(j-k)), q[j], q[k])
        circ.h(q[j])

# Modular Exponentiation function
def mod_exp(a, b, N):
    result = 1
    base = a % N
    while b > 0:
        if b % 2 == 1:
            result = (result * base) % N
        base = (base * base) % N
        b //= 2
    return result

# Quantum Circuit for Shor's Algorithm
def shors_quantum_algorithm(N):
    # Initialize quantum registers
    n = 3  # Number of qubits for argument register
    q1 = QuantumRegister(n, 'q1')
    q2 = QuantumRegister(n, 'q2')
    c1 = ClassicalRegister(n, 'c1')
    c2 = ClassicalRegister(n, 'c2')
    circuit = QuantumCircuit(q1, q2, c1, c2)

    # Apply Hadamard gates to the argument register (q1)
    for i in range(n):
        circuit.h(q1[i])

    # Apply modular exponentiation (using x^a mod N)
    a = random.randint(2, N-1)
    for i in range(n):
        circuit.append(mod_exp(a, i, N), [q1[i]])

    # Apply Quantum Fourier Transform (QFT) to q1
    qft(circuit, q1, n)

    # Measure the quantum registers
    circuit.measure(q1, c1)
    circuit.measure(q2, c2)

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, simulator, shots=1024).result()
    counts = result.get_counts()

    return counts

# Generate a semiprime number N
N = get_semiprime(1000)
print(f"Semiprime N = {N}")

# Perform classical Shor's algorithm
x, r, p, q = shors_algorithm_classical(N)
print(f"Classical Shor's algorithm: x = {x}, period r = {r}, prime factors = {p}, {q}")

# Perform quantum Shor's algorithm
quantum_result = shors_quantum_algorithm(N)
print("Quantum Shor's algorithm result:", quantum_result)
