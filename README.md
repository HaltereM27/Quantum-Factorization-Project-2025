# Quantum Factorization Project 2025 

This code is designed to factor semiprimes using Shor's Algorithm. 
It's part of the challenge "Crack the Code with Quantum Factorization" from the Iquhack 2025 hackathon. 

### Here’s how it works: 

The code combines quantum and classical parts. 
The quantum side uses qubits to store data, while the classical side uses bits to hold the results after measurements are made. 
A big part of Shor’s algorithm is modular exponentiation, which helps find patterns in numbers. 
This is done using controlled-X gates in the quantum circuit. 

Next, the code uses Quantum Phase Estimation (QPE) to identify the period of the modular exponentiation, which is key to finding the factors of the semiprime. 
After QPE, the Inverse Quantum Fourier Transform (IQFT) is applied to get the period with high accuracy. 
From the binary results of QPE, the code uses continued fractions to figure out the period, which then helps with factoring the semiprime. 
Once the period is found, the classical part of the algorithm comes into play. 
The greatest common divisor (GCD) is calculated to help identify potential factors of the semiprime. 
The quantum circuit is run on a quantum processor, and measurements are taken. 
The frequency of these measured values helps extract the period, which is used to find the factors. 


This code uses the QuantumRingsLib, specifically the "scarlet_quantum_rings" backend, which is cool because it allows experimentation with quantum circuits without needing access to huge quantum machines. 
It runs on a simulated or small-scale quantum processor, making it accessible for testing. 
The code also tracks the largest semiprime factored and its bit size. 
This is useful for monitoring how the algorithm performs with different numbers. 
Additionally, the execution time and number of gate operations are measured, which helps optimize the quantum circuit or test different quantum backends. 

