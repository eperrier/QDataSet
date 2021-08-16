################################################################################
"""
This module implements a TF quantum simulator. It has these classes:
    Noise_Layer            : This is an inernal class for generation noise
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
from scipy.linalg import dft
from scipy.signal import cheby1
###############################################################################
class Noise_Layer(layers.Layer):
    """
    class for generating time-domain realizations of noise 
    """
    def __init__(self, T, M, K, profile, **kwargs):
        """
        class constructor

        T      : Total duration of the input signal 
        M      : Number of time steps
        K      : Number of realizations
        profile: Type of noise
        
        """
        super(Noise_Layer, self).__init__(**kwargs)

        # store class parameters
        self.T = T
        self.M = M
        self.K = K
        
        # define a vector of discrteized frequencies
        f = np.fft.fftfreq(M)*M/T 
    
        # define time step
        Ts = T/M
        
        # check the noise type, initialize required variables and define the correct "call" method
        if profile==0:   # No noise
            self.call = self.call_0
        elif profile==1: # PSD of 1/f + a bump
            alpha        = 1
            S_Z         = 1*np.array([(1/(fq+1)**alpha)*(fq<=15) + (1/16)*(fq>15) + np.exp(-((fq-30)**2)/50)/2 for fq in f[f>=0]])  
            self.P_temp = tf.constant( np.tile( np.reshape( np.sqrt(S_Z*M/Ts), (1,1,self.M//2) ), (1,self.K,1) ), dtype=tf.complex64)
            self.call = self.call_1
        elif profile==2: # Colored Gaussian Stationary Noise
            self.g      = 0.1
            self.color  = tf.ones([self.M//4, 1, 1], dtype=tf.float32 )
            self.call   = self.call_2
        elif profile==3: # Colored Gaussian Non-stationary Noise
            time_range  = [(0.5*T/M) + (j*T/M) for j in range(M)] 
            self.g      = 0.2
            self.color  = tf.ones([self.M//4, 1, 1], dtype=tf.float32 )
            self.non_stationary = tf.constant( np.reshape( 1-(np.abs(np.array(time_range)-0.5*T)*2), (1,M,1,1) ), dtype=tf.float32)
            self.call   = self.call_3
        elif profile==4: # Colored Non-Gaussian Non-stationary Noise
            time_range  = [(0.5*T/M) + (j*T/M) for j in range(M)] 
            self.g      = 0.01
            self.color  = tf.ones([self.M//4, 1, 1], dtype=tf.float32 )
            self.non_stationary = tf.constant( np.reshape( 1-(np.abs(np.array(time_range)-0.5*T)*2), (1,M,1,1) ), dtype=tf.float32)
            self.call   = self.call_4
        elif profile==5: # PSD of 1/f
            alpha       = 1
            S_Z         = 1*np.array([(1/(fq+1)**alpha) for fq in f[f>=0]])  
            self.P_temp = tf.constant( np.tile( np.reshape( np.sqrt(S_Z*M/Ts), (1,1,self.M//2) ), (1,self.K,1) ), dtype=tf.complex64)
            self.call   = self.call_1        
        elif profile==6: # correlated noise      
            self.g = 0.3
            self.call = self.call_6

        
    def call_0(self, inputs, training=False): # No noise
        """
        Method to generate type 0 noise
        
        """
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([self.M, self.K,1],dtype=np.int32))],0 )
        return tf.zeros(temp_shape, dtype=tf.float32)
    
    def call_1(self, inputs, training=False): # PSD of 1/f + a bump
        """
        Method to generate type 1 and type 5 noise
        
        """
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([1, 1],dtype=np.int32))],0 )
        P_temp     = tf.tile(self.P_temp, temp_shape)
 
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([self.K, self.M//2],dtype=np.int32))],0 )
        P_temp     = tf.multiply(P_temp, tf.exp(2*np.pi*1j* tf.cast(tf.random.uniform(temp_shape, dtype=tf.float32), dtype=tf.complex64) ) )
        
        noise      = tf.math.real( tf.signal.ifft( tf.concat( [P_temp, tf.reverse( tf.math.conj(P_temp), axis=tf.constant([2]) )], axis=2 ) ) )
        noise      = tf.transpose( tf.expand_dims(noise, axis=-1), perm=[0,2,1,3] )       
        
        return noise
    
    def call_2(self, inputs, training=False): # Colored Gaussian Stationary Noise
        """
        Method to generate type 2 noise
        
        """
        temp_shape = tf.concat( [self.K*tf.shape(inputs)[0:1], tf.constant(np.array([self.M+(self.M//4)-1,1],dtype=np.int32))],0 )
        noise      =  self.g * tf.nn.convolution( input=tf.random.normal(temp_shape), filters=self.color, padding="VALID")
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([self.K, self.M,1],dtype=np.int32))],0 )
        noise      = tf.transpose( tf.reshape( tf.transpose(noise, perm=[0,2,1]), temp_shape), perm=[0,2,1,3] )
        
        return noise
        
    def call_3(self, inputs, training=False): # Colored Gaussian Non-stationary Noise
        """
        Method to generate type 3 noise
        
        """
        temp_shape = tf.concat( [self.K*tf.shape(inputs)[0:1], tf.constant(np.array([self.M+(self.M//4)-1,1],dtype=np.int32))],0 )
        noise      =  self.g * tf.nn.convolution( input=tf.random.normal(temp_shape), filters=self.color, padding="VALID")
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([self.K, self.M,1],dtype=np.int32))],0 )
        noise      = tf.transpose( tf.reshape( tf.transpose(noise, perm=[0,2,1]), temp_shape), perm=[0,2,1,3] )
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([1,self.K,1],dtype=np.int32))],0 )
        non_stationary = tf.tile(self.non_stationary, temp_shape)
        
        return tf.multiply(noise, non_stationary)
    
    def call_4(self, inputs, training=False): # Colored Gaussian Non-stationary Noise
        """
        Method to generate type 4 noise
        
        """

        temp_shape = tf.concat( [self.K*tf.shape(inputs)[0:1], tf.constant(np.array([self.M+(self.M//4)-1,1],dtype=np.int32))],0 )
        noise      = tf.nn.convolution( input=tf.random.normal(temp_shape), filters=self.color, padding="VALID")
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([self.K, self.M,1],dtype=np.int32))],0 )
        noise      = tf.transpose( tf.reshape( tf.transpose(noise, perm=[0,2,1]), temp_shape), perm=[0,2,1,3] )
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([1,self.K,1],dtype=np.int32))],0 )
        non_stationary = tf.tile(self.non_stationary, temp_shape)

        return tf.square( tf.multiply(noise, non_stationary) )*self.g

    def call_6(self, inputs, training=False):  # correlated noise
        """
        Method to generate type 6 noise
        
        """
        return self.g*( tf.square(inputs) )
###############################################################################
class LTI_Layer(layers.Layer):
    """
    class for simulating the response of an LTI system
    """
    def __init__(self, T, M, **kwargs):
        """
        class constructor

        T  : Total duration of the input signal 
        M  : Number of time steps
        """
        super(LTI_Layer, self).__init__(**kwargs)

        #define filter coefficients
        num, den = cheby1(4,0.1,2*np.pi*20, analog=True)
        
        # define frequency vector
        f = np.reshape(np.fft.fftfreq(M)*M/T, (1,M))

        # evaluate the dft matrix
        F =  dft(M, 'sqrtn')
        
        # evaluate the numerator and denominator at each frequency 
        H_num = np.concatenate([(1j*2*np.pi*f)**s for s in range(len(num)-1,-1,-1)], axis=0)
        H_den = np.concatenate([(1j*2*np.pi*f)**s for s in range(len(den)-1,-1,-1)], axis=0)

        # evaluate the frequency response
        H = np.diag( (num@H_num) / (den@H_den) )
        
        # evaluate the full transformation and convert to a tensor of correct shape
        self.L = tf.constant( np.reshape( F.conj().T @ H @ F, (1,1,M,M) ), dtype=tf.complex64 )


    def call(self, inputs):
        """
        Method to evaluate the ouput of the layer which represents the response of the system to the input
        """

        # convert variables to complex
        x = tf.cast(tf.transpose(inputs, perm=[0,2,1,3]), tf.complex64)
        
        # repeat the transformation matrix
        temp_shape = tf.concat( [tf.shape(inputs)[0:1], tf.constant(np.array([1,1,1],dtype=np.int32))],0 )
        L = tf.tile( self.L, temp_shape )

        # apply the transformation
        y = tf.transpose( tf.math.real( tf.matmul(L , x) ), perm=[0,2,1,3])

        return y
###############################################################################
class SigGen(layers.Layer):
    """
    This class defines a custom tensorflow layer that generates a sequence of control pulse parameters
    
    """
    def __init__(self, T, M, n_max, waveform="Gaussian", **kwargs):
        """
        class constructor
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        n_max         : Maximum number of control pulses in the sequence
        waveform      : Waveform shape can either be "Gaussian", "Square", or "Zero"
        """
        # we must call thus function for any tensorflow custom layer
        super(SigGen, self).__init__(**kwargs)
        
        # store the parameters
        self.n_max           = n_max
        self.T               = T
        self.M               = M    
        self.time_range      = tf.constant( np.reshape( [(0.5*T/M) + (j*T/M) for j in range(M)], (1,M,1,1) ) , dtype=tf.float32)

        if waveform=="Gaussian":
            self.call = self.call_Gaussian
        elif waveform=="Square":
            self.call = self.call_Square
        else:
            self.call = self.call_0
       
        # define the constant parmaters to shift the pulses correctly
        self.pulse_width = (0.5*self.T/self.n_max)
        
        self.a_matrix    = np.ones((self.n_max, self.n_max))
        self.a_matrix[np.triu_indices(self.n_max,1)] = 0
        self.a_matrix    = tf.constant(np.reshape(self.a_matrix,(1,self.n_max,self.n_max)), dtype=tf.float32)
        
        self.b_matrix    = np.reshape([idx + 0.5 for idx in range(self.n_max)], (1,self.n_max,1) ) * self.pulse_width
        self.b_matrix    = tf.constant(self.b_matrix, dtype=tf.float32)
          
    def call_Square(self, inputs, training=False):
        """
        Method to generate square pulses
        
        """
        # generate randomly the signal parameters
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        a_matrix   = tf.tile(self.a_matrix, temp_shape)
        b_matrix   = tf.tile(self.b_matrix, temp_shape)
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([self.n_max,1],dtype=np.int32))],0 )     
        amplitude  = 100*tf.random.uniform(shape = temp_shape, minval=-1, maxval=1, dtype=tf.float32)
        position   = 0.5*self.pulse_width + tf.random.uniform(shape= temp_shape, dtype=tf.float32)*( ( (self.T - self.n_max*self.pulse_width)/(self.n_max+1) ) - 0.5*self.pulse_width)
        position   = tf.matmul(a_matrix, position) + b_matrix
        std        = self.pulse_width * tf.ones(temp_shape, dtype=tf.float32)
                
        # combine the parameters into one tensor
        signal_parameters = tf.concat([amplitude, position, std] , -1)
        
        # construct the signal
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1,1],dtype=np.int32))],0 )     
        time_range = tf.tile(self.time_range, temp_shape)
        tau   = [tf.reshape( tf.matmul(position[:,idx,:],  tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]
        A     = [tf.reshape( tf.matmul(amplitude[:,idx,:], tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]
        sigma = [tf.reshape( tf.matmul(std[:,idx,:]      , tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]    
        signal = [tf.multiply(A[idx], tf.cast( tf.logical_and( tf.greater(time_range, tau[idx] - 0.5*sigma[idx]), tf.less(time_range, tau[idx]+0.5*sigma[idx])), tf.float32) ) for idx in range(self.n_max)]
        signal = tf.add_n(signal)
        
        return signal_parameters, signal
        
    def call_Gaussian(self, inputs, training=False):
        """
        Method to generate Gaussian pulses
        
        """
        
        # generate randomly the signal parameters
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        a_matrix    = tf.tile(self.a_matrix, temp_shape)
        b_matrix    = tf.tile(self.b_matrix, temp_shape)
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([self.n_max,1],dtype=np.int32))],0 )     
        amplitude   = 100*tf.random.uniform(shape = temp_shape, minval=-1, maxval=1, dtype=tf.float32)
        position    = 0.5*self.pulse_width + tf.random.uniform(shape= temp_shape, dtype=tf.float32)*( ( (self.T - self.n_max*self.pulse_width)/(self.n_max+1) ) - 0.5*self.pulse_width)
        position    = tf.matmul(a_matrix, position) + b_matrix
        std         = self.pulse_width * tf.ones(temp_shape, dtype=tf.float32)/6
                
        # combine the parameters into one tensor
        signal_parameters = tf.concat([amplitude, position, std] , -1)

        # construct the signal
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1,1],dtype=np.int32))],0 )     
        time_range = tf.tile(self.time_range, temp_shape)
        tau   = [tf.reshape( tf.matmul(position[:,idx,:],  tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]
        A     = [tf.reshape( tf.matmul(amplitude[:,idx,:], tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]
        sigma = [tf.reshape( tf.matmul(std[:,idx,:]      , tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]
        signal = [tf.multiply(A[idx], tf.exp( -0.5*tf.square(tf.divide(time_range - tau[idx], sigma[idx])) ) ) for idx in range(self.n_max)] 
        signal = tf.add_n(signal)
        
        return signal_parameters, signal
    
    def call_0(self, inputs, training=False):
        """
        Method to generate the zero pulse sequence [for free evolution analysis]
        """
        
        # construct zero signal 
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([self.M,1,1],dtype=np.int32))],0 )
        signal     = tf.zeros(temp_shape, dtype=tf.float32)
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([self.n_max,3],dtype=np.int32))],0 )
        signal_parameters = tf.zeros(temp_shape, dtype=tf.float32)
        
        return signal_parameters,signal 
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
      
    def __init__(self, T, M, dynamic_operators, static_operators, noise_operators, measurement_operators, initial_states, K=1, waveform="Gaussian", num_pulses=5, distortion=False, noise_profile=0):
        """
        Class constructor.

        T                : Evolution time
        M                : Number of time steps
        dynamic_operators: A list of arrays that represent the terms of the control Hamiltonian (that depend on pulses)
        static_operators : A list of arrays that represent the terms of the drifting Hamiltonian (that are constant)
        noise_operators  : A list of arrays that represent the terms of the classical noise Hamiltonians
        K                : Number of noise realizations
        waveform         : The type of waveform [either "Zero", "Square", or "Gaussian"]
        num_pulses       : Number of pulses per control sequence
        distortion       : True for simulating distortions, False for no distortions 
        noise_profile    : The type of noise, a value chosen from [0,1,2,4,5,6]
        """

        delta_T   = T/M
        self.time_range = [(0.5*T/M) + (j*T/M) for j in range(M)]
        
        # define a dummy input layer needed to generate the control pulses and noise
        dummy_input = layers.Input(shape=(1,))
        
        # define the custom tensorflow layer that generates the control pulses for each direction and concatente if neccesary
        if len(dynamic_operators)>1:
            pulses            = [SigGen(T, M, num_pulses, waveform)(dummy_input) for _ in dynamic_operators]
            pulse_parameters  = layers.Concatenate(axis=-1)([p[0] for p in pulses])
            pulse_time_domain = layers.Concatenate(axis=-1)([p[1] for p in pulses])
        else:
            pulse_parameters, pulse_time_domain = SigGen(T, M, num_pulses, waveform)(dummy_input) 

        if distortion==True:
            distorted_pulse_time_domain = LTI_Layer(T, M)(pulse_time_domain)
        else:
            distorted_pulse_time_domain = pulse_time_domain
       
        # define the custom tensorflow layer that generates the noise realizations in time-domain and concatente if neccesary
        if len(noise_operators)>1:
            noise = []
            for profile in noise_profile:
                if profile!=6: #uncorrelated along different directions
                    noise.append( Noise_Layer(T, M, K, profile)(dummy_input) )
                else:    #correlated with the prevu=ious direction
                    noise.append( Noise_Layer(T, M, K, profile)(noise[-1]) )               
            noise_time_domain = layers.Concatenate(axis=-1)(noise)     
        else:
            noise_time_domain = Noise_Layer(T, M, K, noise_profile[0])(dummy_input)              

        # define the custom tensorflow layer that constructs the H0 part of the Hamiltonian from parameters at each time step
        H0 = HamiltonianConstruction(dynamic_operators=dynamic_operators, static_operators=static_operators, name="H0")(distorted_pulse_time_domain)

        # define the custom tensorflow layer that constructs the H1 part of the Hamiltonian from parameters at each time step
        H1 = HamiltonianConstruction(dynamic_operators=noise_operators, static_operators=[], name="H1")(noise_time_domain)
            
        # define the custom tensorflow layer that constructs the time-ordered evolution of H0 
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
        self.model   = Model( inputs = dummy_input, outputs = [pulse_parameters, pulse_time_domain, distorted_pulse_time_domain, noise_time_domain, H0, H1, U0, Uc, UI, expectations] + Vo )
        
        # print a summary of the model showing the layers and their connections
        self.model.summary()
    
    def simulate(self, simulator_inputs, batch_size = 1):
        """
        This method is for predicting the measurement outcomes using the trained model. Usually called after training.
        
        simulator inputs: A dummy numpy array of shape (number of examples to simulate, 1)
        
        batch_size:  The number of examples to process at each batch, chosen according to available memory
        
        returns a list of arrays representing H0,H1,U0,U0(T),VO,expectations respectively
        """        
        return self.model.predict(simulator_inputs, verbose=1, batch_size = batch_size)
#############################################################################
