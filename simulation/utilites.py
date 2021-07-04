###############################################################################
"""
utilities.py: This module inlcudes functions to generate noise and controls 
and generate the dataset by simulating the quantum system

"""
###############################################################################
#preample
import numpy as np
import pickle
from simulator import quantumTFsim
from scipy.signal import lsim, cheby1
import zipfile    
import os
import time
###############################################################################
Pauli_operators   = [np.eye(2), np.array([[0.,1.],[1.,0.]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0.],[0.,-1.]])]
###############################################################################
def GenerateGaussianPulses(amp, mean, std, time_range, distortion=None):
    """
    This function generates a Gaussian pulse sequence
    
    amp       : A numpy array that includes the amplitude of each pulse
    mean      : A numpy array that includes the position of each pulse
    std       : A numpy array that includes the standard deviation of each pulse
    time_range: A list that includes all the time steps over which the sequence is defined
    distortion: A tuple of 2 numpy arrays representing numerator and denominator of the analog transfer function modeling the distortions
    
    returns a (1, number of time steps, 1, 1) numpy array for the pulse sequence in time domain for direct use with the TF quantum simulator    
    """
    pulses =  [np.sum([p[0]*np.exp(-0.5*((t-p[1])/p[2])**2) for p in zip(amp,mean,std)]) for t in time_range] 
    if distortion!=None:
        distorted_pulses = lsim(tuple(distortion), pulses, time_range)[1]
    else:
        distorted_pulses = pulses
    return  np.reshape(pulses, (1,len(time_range),1,1)), np.reshape(distorted_pulses, (1,len(time_range),1,1))
###############################################################################
def GenerateSquarePulses(amp, pos, width, time_range, distortion=None):
    """
    This function generates a square pulse sequence
    
    amp       : A numpy array that includes the amplitude of each pulse
    pos       : A numpy array that includes the position of each pulse
    width     : A numpy array that includes the width of each pulse
    time_range: A list that includes all the time steps over which the sequence is defined
    distortion: A tuple of 2 numpy arrays representing numerator and denominator of the analog transfer function modeling the distortions
    
    returns a (1, number of time steps, 1, 1) numpy array for the pulse sequence in time domain for direct use with the TF quantum simulator    
    """
    pulses = sum( [np.array([p[0]*(t>(p[1]-0.5*p[2]))*(t<(p[1]+0.5*p[2])) for t in time_range]) for p in zip(amp,pos,width)] )
    if distortion!=None:
        distorted_pulses = lsim(tuple(distortion), pulses, time_range)[1]
    else:
        distorted_pulses = pulses
    return  np.reshape(pulses, (1,len(time_range),1,1)), np.reshape(distorted_pulses, (1,len(time_range),1,1))
        
###############################################################################
def GenerateNoise(T, M, K, profile, beta_1=None):
    """
    This function generates a set of realizations in time domain for a noise process
    
    T      : Maximum time
    M      : Number of time steps     
    K      : Number of realization required
    profile: An index that specifies the porfile of noise 
    beta_1 : optinal array of realizations of a noise process to generate another correlated one
    
    returns a (1, number of time steps, number of realizations, 1) numpy array for the noise realizations in time domain, for direct use with the TF quantum simulator
    """
    ###########################################################################
    # define a vector of discrteized frequencies
    f = np.fft.fftfreq(M)*M/T 
    
    # define time step
    Ts = T/M
    
    # check which profile to use and generate the niose
    if profile==0: # No noise
        return np.zeros((1,M,K,1))
    elif profile==1:# PSD of 1/f + a bump
        alpha  = 1
        S_Z    = 1*np.array([(1/(fq+1)**alpha)*(fq<=15) + (1/16)*(fq>15) + np.exp(-((fq-30)**2)/50)/2 for fq in f[f>=0]])  
    elif profile==2: # Colored Gaussian Stationary Noise 
        time_range  = [(0.5*T/M) + (j*T/M) for j in range(M)] 
        g           = 0.1
        color       = [1 for t in time_range[0:M//4]]
        noise       = [ [np.random.randn() for _ in range(M+(M//4)-1)] for _ in range(K)]
        noise       = [np.reshape( np.convolve(noise[idx_K],  color, 'valid'), (1,M,1,1) ) for idx_K in range(K)]
        return np.concatenate(noise, axis=2)*g
    elif profile==3: # Colored Gaussian Non-stationary Noise 
        time_range     = [(0.5*T/M) + (j*T/M) for j in range(M)] 
        non_stationary = 1-(np.abs(np.array(time_range)-0.5*T)*2)
        color          = [1 for t in time_range[0:M//4]]
        g              = 0.2
        noise          = [ [np.random.randn() for _ in range(M+(M//4)-1)] for _ in range(K)]
        noise          = [np.reshape( non_stationary * np.convolve(noise[idx_K],  color, 'valid'), (1,M,1,1) ) for idx_K in range(K)]
        return np.concatenate(noise, axis=2)*g 
    elif profile==4: # Colored Non-Gaussian Non-stationary Noise
        time_range     = [(0.5*T/M) + (j*T/M) for j in range(M)] 
        non_stationary = 1-(np.abs(np.array(time_range)-0.5*T)*2)
        color          = [1 for t in time_range[0:M//4]]
        g              = 0.02
        noise          = [ [np.random.randn() for _ in range(M+(M//4)-1)] for _ in range(K)]
        noise          = [np.reshape( non_stationary * np.convolve(noise[idx_K],  color, 'valid'), (1,M,1,1) ) for idx_K in range(K)]
        return (np.concatenate(noise, axis=2)**2)*g       
    elif profile==5: # PSD of 1/f
        alpha  = 1
        S_Z    = 1*np.array([(1/(fq+1)**alpha) for fq in f[f>=0]])      
    elif profile==6: # correlated noise
        g = 0.3
        return g*(beta_1**2)
    
    ###########################################################################
    # profiles with single-side band PSD require further processing 
    
    # define a list to store the different noise realizations
    beta  = []
    
    # generate different realizations
    for _ in range(K):
        
        #1) add random phase to the properly normalized PSD
        P_temp = np.sqrt(S_Z*M/Ts)*np.exp(2*np.pi*1j*np.random.rand(1,M//2))

        #2) add the symmetric part of the spectrum
        P_temp = np.concatenate( ( P_temp , np.flip(P_temp.conj()) ), axis=1 )

        #3) take the inverse Fourier transform
        x      = np.real(np.fft.ifft(P_temp))

        # store
        beta.append(np.reshape(x,(1,M,1,1)))
    
    # concatenate along axis 2 which encodes the realizations
    return np.concatenate(beta, axis=2)
###############################################################################
def simulate(sim_parameters):
    """
    This function generates the dataset and stores it based on the simulation parameters passed as a dictionary
    
    """
    ###########################################################################
    # 1) Define the simulator
    simulator = quantumTFsim(sim_parameters["T"], sim_parameters["M"], sim_parameters["dynamic_operators"], sim_parameters["static_operators"], sim_parameters["noise_operators"], sim_parameters["measurement_operators"], sim_parameters["initial_states"], sim_parameters["K"])
    
    fzip = zipfile.ZipFile("%s.zip"%sim_parameters["name"], mode='w', compression=zipfile.ZIP_DEFLATED)          
   
    ###########################################################################
    # 2) Generate the noise realizations and control pulses
    time_range         = [(0.5*sim_parameters["T"]/sim_parameters["M"]) + (j*sim_parameters["T"]/sim_parameters["M"]) for j in range(sim_parameters["M"])] 
    pulse_width        = 0.5*sim_parameters["T"]/sim_parameters["num_pulses"] # pusle width      
    
    if sim_parameters["pulse_shape"]=="Gaussian":
        GeneratePulses = GenerateGaussianPulses 
        sigma          = [pulse_width/6 for _ in range(sim_parameters["num_pulses"])]   
    else:
        GeneratePulses = GenerateSquarePulses
        sigma          = [pulse_width for _ in range(sim_parameters["num_pulses"])]
     
    pulse_parameters = np.zeros( (sim_parameters["batch_size"], sim_parameters["num_pulses"], 3*len(sim_parameters["dynamic_operators"])) )
    pulses           = np.zeros( (sim_parameters["batch_size"], sim_parameters["M"], 1, len(sim_parameters["dynamic_operators"])) )
    distortion       = [cheby1(4,0.1,2*np.pi*20, analog=True) for _ in sim_parameters["dynamic_operators"] ]     # Distortion filter coefficients
    distorted_pulses = np.zeros( (sim_parameters["batch_size"], sim_parameters["M"], 1, len(sim_parameters["dynamic_operators"])) )
    
    noise            = np.zeros( (sim_parameters["batch_size"], sim_parameters["M"], sim_parameters["K"], len(sim_parameters["noise_operators"])) )
    # 3) Run the simulator for pulses without distortions and collect the results
    print("Running the simulation for pulses without distortion\n")
    
    for idx_batch in range(sim_parameters["num_ex"]//sim_parameters["batch_size"]):
    ###########################################################################
        for idx_ex in range(sim_parameters["batch_size"]):
            for idx_direction in range(len(sim_parameters["dynamic_operators"])):
                # randomize the control pulse parameters
                A      = 5*( 2*np.random.rand(sim_parameters["num_pulses"])-1 )
                # generate the pulse postions randomly
                d = [0.5*pulse_width + np.random.rand()*( ((sim_parameters["T"]-sim_parameters["num_pulses"]*pulse_width)/(sim_parameters["num_pulses"]+1)) - 0.5*pulse_width) for _ in range(sim_parameters["num_pulses"]) ]
                pos = [d[0] + 0.5*pulse_width]
                for j in range(1,sim_parameters["num_pulses"]):
                    pos.append( pos[j-1] + pulse_width + d[j] )
                    
                # store the pulse parameters
                pulse_parameters[idx_ex, :, idx_direction*3:idx_direction*3 + 3] = np.concatenate([np.reshape( A, (1, sim_parameters["num_pulses"], 1) ), np.reshape( pos, (1, sim_parameters["num_pulses"], 1) ), np.reshape(sigma, (1, sim_parameters["num_pulses"], 1) ) ], axis = 2) 
            
                # store the pulse sequence in time domain
                pulses[idx_ex, :, 0:1, idx_direction:idx_direction+1], distorted_pulses[idx_ex, :, 0:1, idx_direction:idx_direction+1] =  GeneratePulses(A, pos, sigma, time_range, distortion[idx_direction]) 
            
            for idx_direction in range(len(sim_parameters["noise_operators"])):
                # generate and store the noise
                noise[idx_ex, :, :, idx_direction:idx_direction+1] =  GenerateNoise(sim_parameters["T"], sim_parameters["M"], sim_parameters["K"], sim_parameters["noise_profile"][idx_direction], noise[idx_ex,:,:,idx_direction-1:idx_direction]) 
        ###########################################################################
        simulation_results               = simulator.simulate([noise, pulses], batch_size = sim_parameters["batch_size"])
        expectations                     = simulation_results[0] 
        print("Storing the results for batch %d\n"%idx_batch)
        ###########################################################################
        # 4) Save the results in an external file and zip everything
        for idx_ex in range(sim_parameters["batch_size"]):          
            start                            = time.time()
            simulation_results               = simulator.simulate([noise, pulses], batch_size = sim_parameters["batch_size"])
            sim_parameters["elapsed_time"]   = time.time()-start
            H0, H1, U0, Uc, UI, expectations = simulation_results[0:6] 
            Vo                               = simulation_results[6:]
            Results = {"sim_parameters"  : sim_parameters,
                       "pulse_parameters": pulse_parameters[idx_ex:idx_ex+1, :],
                       "distortion"      : None, 
                       "time_range"      : time_range,
                       "pulses"          : pulses[idx_ex:idx_ex+1, :],
                       "distorted_pulses": pulses[idx_ex:idx_ex+1,:],
                       "expectations"    : np.average( expectations[idx_ex:idx_ex+1, :], axis=1),
                       "Vo_operator"     : [np.average( V[idx_ex:idx_ex+1, :], axis=1) for V in Vo],
                       "noise"           : noise[idx_ex:idx_ex+1, :],      
                       "H0"              : H0[idx_ex:idx_ex+1, :],
                       "H1"              : H1[idx_ex:idx_ex+1, :],
                       "U0"              : U0[idx_ex:idx_ex+1, :],
                       "UI"              : UI[idx_ex:idx_ex+1, :],
                       "Vo"              : [V[idx_ex:idx_ex+1, :] for V in Vo],  
                       "Eo"              : expectations[idx_ex:idx_ex+1, :]
                       }
            # open a pickle file
            fname = "%s_ex_%d"%(sim_parameters["name"],idx_ex + idx_batch*sim_parameters["batch_size"])
            f = open(fname, 'wb')
            # save the results
            pickle.dump(Results, f, -1)
            # close the pickle file
            f.close()
            #add the file to zip folder
            fzip.write(fname)
            # remove the pickle file
            os.remove(fname)
    ###########################################################################
    # close the zip file
    fzip.close()               
    ###########################################################################
    # 5) Run the simulator for pulses with distortions and collect the results
    print("Running the simulation for pulses with distortion\n")
    fzip = zipfile.ZipFile("%s_D.zip"%sim_parameters["name"], mode='w', compression=zipfile.ZIP_DEFLATED)          
    for idx_batch in range(sim_parameters["num_ex"]//sim_parameters["batch_size"]):
    ###########################################################################
        for idx_ex in range(sim_parameters["batch_size"]):
            for idx_direction in range(len(sim_parameters["dynamic_operators"])):
                # randomize the control pulse parameters
                A      = 5*( 2*np.random.rand(sim_parameters["num_pulses"])-1 )
                # generate the pulse postions randomly
                d = [0.5*pulse_width + np.random.rand()*( ((sim_parameters["T"]-sim_parameters["num_pulses"]*pulse_width)/(sim_parameters["num_pulses"]+1)) - 0.5*pulse_width) for _ in range(sim_parameters["num_pulses"]) ]
                pos = [d[0] + 0.5*pulse_width]
                for j in range(1,sim_parameters["num_pulses"]):
                    pos.append( pos[j-1] + pulse_width + d[j] )
                    
                # store the pulse parameters
                pulse_parameters[idx_ex, :, idx_direction*3:idx_direction*3 + 3] = np.concatenate([np.reshape( A, (1, sim_parameters["num_pulses"], 1) ), np.reshape( pos, (1, sim_parameters["num_pulses"], 1) ), np.reshape(sigma, (1, sim_parameters["num_pulses"], 1) ) ], axis = 2) 
            
                # store the pulse sequence in time domain
                pulses[idx_ex, :, 0:1, idx_direction:idx_direction+1], distorted_pulses[idx_ex, :, 0:1, idx_direction:idx_direction+1] =  GeneratePulses(A, pos, sigma, time_range, distortion[idx_direction]) 
            
            for idx_direction in range(len(sim_parameters["noise_operators"])):
                # generate and store the noise
                noise[idx_ex, :, :, idx_direction:idx_direction+1] =  GenerateNoise(sim_parameters["T"], sim_parameters["M"], sim_parameters["K"], sim_parameters["noise_profile"][idx_direction], noise[idx_ex,:,:,idx_direction-1:idx_direction]) 
        ###########################################################################
        simulation_results               = simulator.simulate([noise, pulses], batch_size = sim_parameters["batch_size"])
        expectations                     = simulation_results[0] 
        print("Storing the results for batch %d\n"%idx_batch)
        ###########################################################################
        # 4) Save the results in an external file and zip everything
        for idx_ex in range(sim_parameters["batch_size"]):          
            start                            = time.time()
            simulation_results               = simulator.simulate([noise, pulses], batch_size = sim_parameters["batch_size"])
            sim_parameters["elapsed_time"]   = time.time()-start
            H0, H1, U0, Uc, UI, expectations = simulation_results[0:6] 
            Vo                               = simulation_results[6:]
            Results = {"sim_parameters"  : sim_parameters,
                       "pulse_parameters": pulse_parameters[idx_ex:idx_ex+1, :],
                       "distortion"      : distortion, 
                       "time_range"      : time_range,
                       "pulses"          : pulses[idx_ex:idx_ex+1, :],
                       "distorted_pulses": distorted_pulses[idx_ex:idx_ex+1,:],
                       "expectations"    : np.average( expectations[idx_ex:idx_ex+1, :], axis=1),
                       "Vo_operator"     : [np.average( V[idx_ex:idx_ex+1, :], axis=1) for V in Vo],
                       "noise"           : noise[idx_ex:idx_ex+1, :],      
                       "H0"              : H0[idx_ex:idx_ex+1, :],
                       "H1"              : H1[idx_ex:idx_ex+1, :],
                       "U0"              : U0[idx_ex:idx_ex+1, :],
                       "UI"              : UI[idx_ex:idx_ex+1, :],
                       "Vo"              : [V[idx_ex:idx_ex+1, :] for V in Vo],  
                       "Eo"              : expectations[idx_ex:idx_ex+1, :]
                       }
            # open a pickle file
            fname = "%s_D_ex_%d"%(sim_parameters["name"],idx_ex + idx_batch*sim_parameters["batch_size"])
            f = open(fname, 'wb')
            # save the results
            pickle.dump(Results, f, -1)
            # close the pickle file
            f.close()
            #add the file to zip folder
            fzip.write(fname)
            # remove the pickle file
            os.remove(fname)
    ###########################################################################
    # close the zip file
    fzip.close()  
###############################################################################        
def CheckNoise(sim_parameters):
    """
    This function calculates the coherence measurements to check the noise behaviour, based on the simulation parameters passed as a dictionary
    
    """
    ###########################################################################
    # 1) Generate the noise realizations and zero control pulses

    pulses           = np.zeros( (1, sim_parameters["M"], 1, len(sim_parameters["dynamic_operators"])) )
    noise            = np.zeros( (1, sim_parameters["M"], sim_parameters["K"], len(sim_parameters["noise_operators"])) )
    
    for idx_direction in range(len(sim_parameters["noise_operators"])):
        # generate and store the noise
        noise[0, :, :, idx_direction:idx_direction+1] =  GenerateNoise(sim_parameters["T"], sim_parameters["M"], sim_parameters["K"], sim_parameters["noise_profile"][idx_direction], noise[0, : , :, idx_direction-1: idx_direction]) 
    ###########################################################################
    # 2) Define the simulator
    simulator = quantumTFsim(sim_parameters["T"], sim_parameters["M"], sim_parameters["dynamic_operators"], sim_parameters["static_operators"], sim_parameters["noise_operators"], sim_parameters["measurement_operators"], sim_parameters["initial_states"], sim_parameters["K"])
    ###########################################################################
    # 3) Run the simulator and collect the results
    print("Running the simulation\n")
    simulation_results               = simulator.simulate([noise, pulses], batch_size = 1)
    H0, H1, U0, Uc, UI, expectations = simulation_results[0:6] 
    Vo                               = simulation_results[6:]
    Vo                               = [np.average( V, axis=1) for V in Vo]
    print("Analyzing results\n")
    print("Measurement are:")
    print( np.average( expectations, axis=1) )
    print("The Vo operators are:")
    print(Vo)
    print("The distance measures are:")
    print([np.linalg.norm( V[0,:]-np.eye(sim_parameters["dim"]) , 2) for V in Vo])
