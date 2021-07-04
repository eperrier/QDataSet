################################################################################
"""
This module implements a TF quantum simulator. It has these classes:
    HamiltonianConstruction: This is an internal class for constructing Hamiltonians
    QuantumCell            : This is an internal class required for implementing time-ordered evolution
    QuantumEvolution       : This is an internal class to implement time-ordered quantum evolution
    QuantumMeasurement     : This is an internal class to model coupling losses at the output
    VoLayer                : This is an internal class to calculate the Vo operator using the interaction picture
    quantumTFsim           : This is the main class that defines machine learning model for the qubit 
"""
###############################################################################
# Preamble
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,Model
###############################################################################
class HamiltonianConstruction(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the Hamiltonian parameters as input, and generates the
    Hamiltonain matrix as an output at each time step for each example in the batch
    """
    
    def __init__(self, dynamic_operators, static_operators, **kwargs):
        """
        Class constructor 
        
        dynamic_operators: a list of all operators that have time-varying coefficients
        static_operators : a list of all operators that have constant coefficients
        """
        
        self.dynamic_operators = [tf.constant(op, dtype=tf.complex64) for op in dynamic_operators]
        self.static_operators  = [tf.constant(op, dtype=tf.complex64) for op in static_operators]
        self.dim = dynamic_operators[0].shape[-1]   

        # this has to be called for any tensorflow custom layer
        super(HamiltonianConstruction, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 

        H = []
        # loop over the strengths of all dynamic operators
        
        for idx_op, op in enumerate(self.dynamic_operators):

            # select the particular strength of the operator
            h = tf.cast(inputs[:,:,:,idx_op:idx_op+1] ,dtype=tf.complex64)

            # construct a tensor in the form of a row vector whose elements are [d1,d2,d3, 1,1], where d1, d2, and d3 correspond to the
            # number of examples, number of time steps of the input, and number of realizations
            temp_shape = tf.concat( [tf.shape(inputs)[0:3],tf.constant(np.array([1,1],dtype=np.int32))],0 )

            # add two extra dimensions for batch, time, and realization
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            operator = tf.expand_dims(operator,0)
            
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            
            # repeat the pulse waveform to as dxd matrix
            temp_shape = tf.constant(np.array([1,1,1,self.dim,self.dim],dtype=np.int32))
            h = tf.expand_dims(h,-1)
            h = tf.tile(h, temp_shape)
            
            # Now multiply each operator with its corresponding strength element-wise and add to the list of Hamiltonians
            H.append( tf.multiply(operator, h) )
       
        # loop over the strengths of all static operators
        for op in self.static_operators:          
            # construct a tensor in the form of a row vector whose elements are [d1,d2,d3,1,1], where d1, d2, and d2 correspond to the
            # number of examples, number of time steps of the input, and number of realizations
            temp_shape = tf.concat( [tf.shape(inputs)[0:3],tf.constant(np.array([1,1],dtype=np.int32))],0 )

            # add two extra dimensions for batch and time
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            operator = tf.expand_dims(operator,0)
            
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            
            # Now add to the list of Hamiltonians
            H.append( operator )
        
        # now add all componenents together
        H =  tf.add_n(H)
                            
        return H    
###############################################################################
class QuantumCell(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces one step forward propagator
    """
    
    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
        delta_T: time step for each propagator
        """  
        
        # here we define the time-step including the imaginary unit, so we can later use it directly with the expm function
        self.delta_T= tf.constant(delta_T*-1j, dtype=tf.complex64)

        # we must define this parameter for RNN cells
        self.state_size = [1]
        
        # we must call thus function for any tensorflow custom layer
        super(QuantumCell, self).__init__(**kwargs)

    def call(self, inputs, states):        
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        states: The tensor representing the state of the cell. This is passed automatically by tensorflow.
        """         
        
        previous_output = states[0] 
        
        # evaluate -i*H*delta_T
        Hamiltonian = inputs * self.delta_T
        
        #evaluate U = expm(-i*H*delta_T)
        U = tf.linalg.expm( Hamiltonian )
        
        # accuamalte U to to the rest of the propagators
        new_output  = tf.matmul(U, previous_output)    
        
        return new_output, [new_output]
###############################################################################
class QuantumEvolution(layers.RNN):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces the time-ordered evolution unitary as output
    """
    
    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
              
        delta_T: time step for each propagator
        """  
        
        # use the custom-defined QuantumCell as base class for the nodes
        cell = QuantumCell(delta_T)

        # we must call thus function for any tensorflow custom layer
        super(QuantumEvolution, self).__init__(cell,  **kwargs)
      
    def call(self, inputs):          
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        """
        
        # define identity matrix with correct dimensions to be used as initial propagtor 
        dimensions = tf.shape(inputs)
        I          = tf.eye( dimensions[-1], batch_shape=[dimensions[0], dimensions[2]], dtype=tf.complex64 )
        
        return super(QuantumEvolution, self).call(inputs, initial_state=[I])         
###############################################################################    
class QuantumMeasurement(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the unitary as input, 
    and generates the measurement outcome probability as output
    """
    
    def __init__(self, initial_state, measurement_operator, **kwargs):
        """
        Class constructor
        
        initial_state       : The inital density matrix of the state before evolution.
        Measurement_operator: The measurement operator
        """          
        self.initial_state        = tf.constant(initial_state, dtype=tf.complex64)
        self.measurement_operator = tf.constant(measurement_operator, dtype=tf.complex64)
    
        # we must call thus function for any tensorflow custom layer
        super(QuantumMeasurement, self).__init__(**kwargs)
            
    def call(self, x): 
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
    
        # extract the different inputs of this layer which are the Vo and Uc
        Vo, Uc = x
        
        # construct a tensor in the form of a row vector whose elements are [d1,1,1,1], where d1 corresponds to the
        # number of examples of the input
        temp_shape = tf.concat( [tf.shape(Vo)[0:1],tf.constant(np.array([1,1,1],dtype=np.int32))],0 )

        # add an extra dimensions for the initial state and measurement tensors to represent batch and realization
        initial_state        = tf.expand_dims( tf.expand_dims(self.initial_state,0), 0)
        measurement_operator = tf.expand_dims( tf.expand_dims(self.measurement_operator,0), 0) 
        
        # repeat the initial state and measurment tensors along the batch dimensions
        initial_state        = tf.tile(initial_state, temp_shape )
        measurement_operator = tf.tile(measurement_operator, temp_shape)   
        
        # evolve the initial state using the propagator provided as input
        final_state = tf.matmul(tf.matmul(Uc, initial_state), Uc, adjoint_b=True )
        
        
        # tile along the realization axis
        temp_shape  = tf.concat( [tf.constant(np.array([1,],dtype=np.int32)), tf.shape(Vo)[1:2], tf.constant(np.array([1,1],dtype=np.int32))],0 )
        final_state = tf.tile(final_state, temp_shape)
        measurement_operator = tf.tile(measurement_operator, temp_shape)  

        # calculate the probability of the outcome
        expectation = tf.linalg.trace( tf.matmul( tf.matmul( Vo, final_state), measurement_operator) ) 
    
        return tf.expand_dims( tf.math.real(expectation), -1)
###############################################################################    
class VoLayer(layers.Layer):
    """
    This class defines a custom tensorflow layer that constructs the Vo operator using the interaction picture definition
    """
    
    def __init__(self, O, **kwargs):
        """
        Class constructor
        
        O: The observable to be measaured
        """
        # this has to be called for any tensorflow custom layer
        super(VoLayer, self).__init__(**kwargs)
    
        self.O = tf.constant(O, dtype=tf.complex64)         
        
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        
        # retrieve the two inputs: Uc and UI
        UI,Uc = x
        
        UI_tilde = tf.matmul(Uc, tf.matmul(UI,Uc, adjoint_b=True) )

        # expand the observable operator along batch and realizations axis
        O = tf.expand_dims(self.O, 0)
        O = tf.expand_dims(O, 0)
         
        temp_shape = tf.concat( [tf.shape(Uc)[0:2], tf.constant(np.array([1,1],dtype=np.int32))], 0 )
        O = tf.tile(O, temp_shape)

        # Construct Vo operator         
        VO = tf.matmul(O, tf.matmul( tf.matmul(UI_tilde,O, adjoint_a=True), UI_tilde) )

        return VO 
###############################################################################
class quantumTFsim():
    """
    This is the main class that defines machine learning model of the qubit.
    """    
      
    def __init__(self, T, M, dynamic_operators, static_operators, noise_operators, measurement_operators, initial_states, K=1):
        """
        Class constructor.

        T                : Evolution time
        M                : Number of time steps
        dynamic_operators: A list of arrays that represent the terms of the control Hamiltonian (that depend on pulses)
        static_operators : A list of arrays that represent the terms of the drifting Hamiltonian (that are constant)
        noise_operators  : A list of arrays that represent the terms of the classical noise Hamiltonians
        K                : Number of noise realizations
        """

        delta_T   = T/M

        # define tensorflow input layers for the pulse sequence and noise realization in time-domain
        pulse_time_domain = layers.Input(shape=(M, 1, len(dynamic_operators) ), name="control")
        noise_time_domain = layers.Input(shape=(M, K, len(noise_operators)  ), name="noise")                             

        # define the custom defined tensorflow layer that constructs the H0 part of the Hamiltonian from parameters at each time step
        H0 = HamiltonianConstruction(dynamic_operators=dynamic_operators, static_operators=static_operators, name="H0")(pulse_time_domain)

        # define the custom defined tensorflow layer that constructs the H1 part of the Hamiltonian from parameters at each time step
        H1 = HamiltonianConstruction(dynamic_operators=noise_operators, static_operators=[], name="H1")(noise_time_domain)
            
        # define the custom defined tensorflow layer that constructs the time-ordered evolution of H0 
        U0 = QuantumEvolution(delta_T, return_sequences=True, name="U0")(H0)
    
        # define Uc which is U0(T)
        Uc = layers.Lambda(lambda u0: u0[:,-1,:,:,:], name="Uc")(U0)
        
        # define custom tensorflow layer to calculate HI
        U0_ext = layers.Lambda(lambda x: tf.tile(x, tf.constant([1,1,K,1,1], dtype=tf.int32) ) )(U0)
        HI     = layers.Lambda(lambda x: tf.matmul( tf.matmul(x[0],x[1], adjoint_a=True), x[0] ), name="HI" )([U0_ext, H1])
    
        # define the custom defined tensorflow layer that constructs the time-ordered evolution of HI
        UI = QuantumEvolution(delta_T, return_sequences=False, name="UI")(HI)
        
        # construct the Vo operators
        Uc_ext = layers.Lambda(lambda x: tf.tile(x, tf.constant([1,K,1,1], dtype=tf.int32) ) )(Uc)        
        Vo     = [VoLayer(O, name="V%d"%idx_O)([UI,Uc_ext]) for idx_O, O in enumerate(measurement_operators)]
        
        # add the custom defined tensorflow layer that calculates the measurement outcomes
        expectations = [
                [QuantumMeasurement(rho,X, name="rho%dM%d"%(idx_rho,idx_X))([Vo[idx_X],Uc]) for idx_X, X in enumerate(measurement_operators)]
                for idx_rho,rho in enumerate(initial_states)]
       
        # concatenate all the measurement outcomes
        expectations = layers.Concatenate(axis=-1)(sum(expectations, [] ))
        
        # define now the tensorflow model
        self.model   = Model( inputs = [noise_time_domain, pulse_time_domain], outputs = [H0, H1, U0, Uc, UI, expectations] + Vo )
        
        # print a summary of the model showing the layers and their connections
        self.model.summary()
    
    def simulate(self, simulator_inputs, batch_size = 1):
        """
        This method is for predicting the measurement outcomes using the trained model. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is for the noise realization of shape (number of examples,number of time steps, number of realizations, number of axes), and the second is for the control pulses of dimensions (number of examples, number of time steps, 1, number of axes)
        
        batch_size:  The number of examples to process at each batch, chosen according to available memory
        
        returns a list of arrays representing H0,H1,U0,U0(T),VO,expectations respectively
        """        
        return self.model.predict(simulator_inputs, verbose=1, batch_size = batch_size)
#############################################################################
