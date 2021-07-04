##############################################
"""
This module generate a dataset

"""
##############################################
# preample
import numpy as np
from utilites import Pauli_operators, simulate, CheckNoise
from itertools import product
################################################
# meta parameters
name        = "S_2q_IX-XI-XX_IZ-ZI_N1-N6"
################################################
# quantum parameters
dim                   = 4                                                 # dimension of the system
Omega                 = [10,12]                                           # qubits energy gap
I                     = Pauli_operators[0]
static_operators      = [Omega[0]*np.kron(Pauli_operators[3], I), 
                         Omega[1]*np.kron(I, Pauli_operators[3])]         # drift Hamiltonian
dynamic_operators     = [np.kron(Pauli_operators[1], I), 
                         np.kron(I, Pauli_operators[1]), 
                         np.kron(Pauli_operators[1], Pauli_operators[1])] # control Hamiltonian 
noise_operators       = [np.kron(Pauli_operators[3], I), 
                         np.kron(I, Pauli_operators[3])  ]                # noise Hamiltonian
measurement_operators = [np.kron(m1, m2) for m1,m2 in list( product(Pauli_operators, Pauli_operators) )][1:] # measurement operators

initial_states_1q     = [np.array([[0.5,0.5],[0.5,0.5]]), np.array([[0.5,-0.5],[-0.5,0.5]]),
                         np.array([[0.5,-0.5j],[0.5j,0.5]]),np.array([[0.5,0.5j],[-0.5j,0.5]]),
                         np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ]
initial_states        = [np.kron(m1, m2) for m1,m2 in list( product(initial_states_1q, initial_states_1q ) )]    # intial states of the two qubits 
##################################################                          
# simulation parameters
T          = 1                                        # Evolution time
M          = 1024                                     # Number of time steps  
num_ex     = 10000                                    # Number of examples
batch_size = 50                                       # batch size for TF 
##################################################
# noise parameters
K               = 2000                                # Number of realzations
noise_profile   = [1,6]                               # Noise type
###################################################
# control parameters
pulse_shape     = "Square"                            # Control pulse shape
num_pulses      = 5                                   # Number of pulses per sequence
####################################################
# Generate the dataset
sim_parameters = dict( [(k,eval(k)) for k in ["name", "dim", "Omega", "static_operators", "dynamic_operators", "noise_operators", "measurement_operators", "initial_states", "T", "M", "num_ex", "batch_size", "K", "noise_profile", "pulse_shape", "num_pulses"] ])
CheckNoise(sim_parameters)
simulate(sim_parameters)
####################################################