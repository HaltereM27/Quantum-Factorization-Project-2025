import math
import numpy as np
from QuantumRingsLib import QuantumRegister, ClassicalRegister, QuantumCircuit
from QuantumRingsLib import QuantumRingsProvider
from QuantumRingsLib import job_monitor
from matplotlib import pyplot as plt
from fractions import Fraction
import time  #for measuring execution time

#setting up the Quantum Rings provider
provider = QuantumRingsProvider(token='rings-200.JZCtyHx0gOLBXkPd1M9mEmUHOvvuBuq7',
                                name='heloisada.silva.1208@gmail.com')
backend = provider.get_backend("scarlet_quantum_rings")  #quantum backend for execution
shots = 1024  #the number of shots it will be used to sample the quantum circuit

#activate the provider account
provider.active_account()

#define a list of semiprimes to factorize
semiprimes_dict = [15, 35, 77, 221, 299, 323]

#variables to track the largest semiprime and factors
largest_bit_size = 0
largest_integer = 0
largest_factors = (None, None)

def iqft_cct(qc, q, n):
    """Applies the Inverse Quantum Fourier Transform (QFT)."""
    for i in range(n):
        for j in range(1, i + 1):
            qc.cu1(-math.pi / 2 ** (i - j + 1), q[j - 1], q[i])
        qc.h(q[i])  #apply hadamard operation to each qubit
    qc.barrier()
    return

def plot_histogram(counts, title=""):
    """Plots a histogram of the measurement results."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.xlabel("States")
    plt.ylabel("Counts")
    mylist = [key for key, val in counts.items() for _ in range(val)]
    unique, inverse = np.unique(mylist, return_inverse=True)
    bin_counts = np.bincount(inverse)
    plt.bar(unique, bin_counts)
    maxFreq = max(counts.values())
    plt.ylim(ymax=np.ceil(maxFreq / 10) * 10 if maxFreq % 10 else maxFreq + 10)
    plt.title(title)
    plt.show()
    return

def modular_exponentiation(qc, q, a, N, m, n):
    """Performs modular exponentiation."""
    for i in range(m):
        qc.h(q[i])  #apply hadamard to each qubit

    qc.barrier()

    for i in range(m):
        qc.cx(q[i], q[(i + 1) % n])  #apply controlled-x operation for modular exponentiation
    qc.barrier()
    return

def continued_fractions(binary_fraction):
    """extracts the period of a continued fraction"""
    frac = Fraction(int(binary_fraction, 2), 2 ** len(binary_fraction))
    period = frac.denominator
    return period

def shors_algorithm(n):
    """implements shors algorithm for factoring semiprimes"""
    a = np.random.randint(2, n - 1)

    if math.gcd(a, n) > 1:
        return math.gcd(a, n)

    number_of_qubits = 7
    q = QuantumRegister(number_of_qubits, 'q')  #quantum register
    c = ClassicalRegister(4, 'c')  #classical register
    qc = QuantumCircuit(q, c)

    #apply hadamard to each qubit in the register
    for i in range(number_of_qubits):
        qc.h(q[i])  #apply hadamard transform to individual qubits

    qc.barrier()

    #apply modular exponentiation
    modular_exponentiation(qc, q, a, n, number_of_qubits, number_of_qubits)

    #apply QPE followed by IQFT
    iqft_cct(qc, q, number_of_qubits)

    #measure the qubits and store the result in the classical register
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    qc.measure(q[2], c[2])

    #start the timer to track execution time
    start_time = time.time()

    #run the quantum circuit and monitor the job
    job = backend.run(qc, shots=shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()

    #end the timer and calculate execution time
    execution_time = time.time() - start_time

    #get the number of gate operations (count of gate types in the circuit)
    num_gate_operations = sum(count for gate, count in qc.count_ops().items())

    #get the number of qubits used
    num_qubits = qc.num_qubits

    #plot the histogram of the measurement results
    plot_histogram(counts, title=f"Factoring result for {n}")

    #extract the period from the most frequent result using continued fractions
    period_binary = max(counts, key=counts.get)
    period = continued_fractions(period_binary)

    #use the period to find the factors of the semiprime
    factors = find_factors(a, period, n)

    #track largest semiprime, bit size, and its factrs
    global largest_bit_size, largest_integer, largest_factors
    if n > largest_integer:
        largest_integer = n
        largest_bit_size = math.floor(math.log2(n)) + 1  #the bit size of the semiprime
        largest_factors = factors

    #print the number of qubits, gate operations, execution time, and largest semiprime information
    print(f"Number of qubits used: {num_qubits}")
    print(f"Number of gate operations: {num_gate_operations}")
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Largest Bit Size Factored: {largest_bit_size} bits")
    print(f"Largest Integer Factored: {largest_integer}")
    if largest_factors != (None, None):
        print(f"Factor 1: {largest_factors[0]}")
        print(f"Factor 2: {largest_factors[1]}")

    return factors

def find_factors(a, period, n):
    """finds the factors of the semiprime using the period"""
    if period % 2 != 0:
        return None

    factor1 = math.gcd(pow(a, period // 2) - 1, n)
    factor2 = math.gcd(pow(a, period // 2) + 1, n)

    if factor1 == 1 or factor1 == n:
        factor1 = None
    if factor2 == 1 or factor2 == n:
        factor2 = None

    return factor1, factor2

def factorize_semiprimes():
    """runs shor's algorithm to factorize a list of semiprimes"""
    for semiprime in semiprimes_dict:
        print(f"Factoring semiprime: {semiprime}")
        factors = shors_algorithm(semiprime)
        if factors:
            print(f"Factors of {semiprime} are: {factors}")
        else:
            print(f"Failed to factor {semiprime}. No factors found.")

#call the function to factorize a list of semiprimes
factorize_semiprimes()
