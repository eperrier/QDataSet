```python

```
![alt text](https://github.com/eperrier/QDataSet/blob/main/qdataset_clean.png?raw=true)

# QDataSet: Quantum Datasets for Machine Learning

## Overview 
This is the repository for the QDataSet introduced in [*QDataset: Quantum Datasets for Machine Learning* by Perrier, Youssry & Ferrie (2021)](https://arxiv.org/abs/2108.06661), a quantum dataset designed specifically to facilitate the training and development of QML algorithms. The QDataSet comprises 52 high-quality publicly available datasets derived from simulations of one- and two-qubit systems evolving in the presence and/or absence of noise.

The datasets are structured to provide a wealth of information to enable machine learning practitioners to use the QDataSet to solve problems in applied quantum computation, such as quantum control, quantum spectroscopy and tomography. Accompanying the datasets in this repository are a set of workbooks demonstrating the use of the QDataSet in a range of optimisation contexts.

### Link to QDataSet

The links to the QDataSet is here: [QDataSet Cloudstor Link](https://cloudstor.aarnet.edu.au/plus/s/rxYKXBS7Tq0kB8o). More details on cloud storage are included below.

### Links to examples notebooks

Example notebooks can be found in the 'examples' subfolder in this repository.

### Link to QDataSet simulation code

Links to the QDataSet simulation code can be found in the 'simulation' subfolder in this repository.

### Citing this repository and paper

Citation of the paper:

Perrier, E., Youssry, A. & Ferrie, C. QDataset: Quantum Datasets for Machine Learning. (2021). 	arXiv:2108.06661 [quant-ph].

Citation of this repository is recommended in the following formats (see the 'Cite this repository' link to the right):

Perrier E., Youssry A., Ferrie C. (2021). QDataSet: Quantum Datasets for Machine Learning (version 1.0.0). DOI: https://doi.org/10.5281/zenodo.5202814

For Bibtex:

@misc{Perrier_QDataSet_Quantum_Datasets_2021,
author = {Perrier, Elija and Youssry, Akram and Ferrie, Chris},
doi = {10.5281/zenodo.5202814},
month = {8},
title = {QDataSet: Quantum Datasets for Machine Learning},
url = {https://github.com/eperrier/QDataSet},
year = {2021}
}

## Summary
The QDataSet comprises 52 datasets based on simulations of one- and two-qubit systems evolving in the presence and/or absence of noise subject to a variety of controls. It has been developed to provide a large-scale set of datasets for the training, benchmarking and competitive development of classical and quantum algorithms for common tasks in quantum sciences, including quantum control, quantum tomography and noise spectroscopy. 

It has been generated using customised code drawing upon base-level Python packages in order to facilitate interoperability and portability across common machine learning and quantum programming platforms. Each dataset consists of 10,000 samples which in turn comprise a range of data relevant to the training of machine learning algorithms for solving optimisation problems. 

The data includes a range of information (stored in list, matrix or tensor format) regarding quantum systems and their evolution, such as: quantum state vectors, drift and control Hamiltonians and unitaries, Pauli measurement distributions, time series data, pulse sequence data for square and Gaussian pulses and noise and distortion data. The total compressed size of the QDataSet (using Pickle and zip formats) is around 14TB (uncompressed, around 100TB). 

Researchers can use the QDataSet in a variety of ways to design algorithms for solving problems in quantum control, quantum tomography and quantum circuit synthesis, together with algorithms focused on classifying or simulating such data. We also provide working examples of how to use the QDataSet in practice and its use in benchmarking certain algorithms. 

The associated paper provides in-depth detail on the QDataSet for researchers who may be unfamiliar with quantum computing, together with specifications for domain experts within quantum engineering, quantum computation and quantum machine learning.

# Description of datasets

## Dataset categories

The datasets in the QDataSet are set-out in Pickle-compressed lists and dictionaries. A taxonomy of each datasets is included below.

Each dataset can be categorised according to the number of qubits in the system and the noise profile to which the system was subject. [The table below] sets out a summary of such categories. For category 1 of the datasets, we created datasets with noise profiles N1, N2, N3, N4, together with the noiseless case. This gives a total of 5 datasets. 

For category 2,  the noise profiles for the X and Z respectively are chosen to be (N1,N5), (N1,N6), (N3,N6). Together with the noiseless case, this gives a total of 4 datasets. 

For category 3 (two-qubit system), we chose only the 1Z (identity on the first qubit, noise along the z-axis for the second) and Z1 (noise along the z-axis for the first qubit, identity along the second) noise to follow the (N1,N6) profile. This category simulates two individual qubit with correlated noise sources. 

For category 4, we generate the noiseless, (N1,N5), and (N1,N6) for the 1Z and Z1 noise. This gives 3 datasets. Therefore, the total number off datasets at this point is 13. Now, if we include the two types of control waveforms, this gives a total of 26. If we also include the cases of distortion and non-distorted control, then this gives a total of 52 datasets. Comprehensive detail on the noise profiles used to generate the datasets is contained in Appendix of the QDataSet paper.

## Naming convention

We chose a convention for the naming of the dataset to try delivering as much information as possible about the chosen parameters for this particular dataset. The name is partitioned into 6 parts, separated by an underscore sign '\_'. 

* The first part is either the letter 'G' or 'S' to denote whether the control waveform is Gaussian or square. 

* The second part is either '1q' or '2q' to denote the dimensionality of the system. 

* The third part denotes the control Hamiltonian. It is formed by listing down the Pauli operators we are using for the control for each qubit, and we separate between qubit by a hyphen '-'. For example, category 1 datasets will have 'X', while category 4 with have 'IX-XI-XX'. 

* The fourth part is optional and it encodes the noise Hamiltonian following the same convention of the third part. 

* The fifth which is also optional part contains the noise profiles following the same order of operators in the fourth part. If the dataset is for noiseless simulation, the the fourth and fifth parts are not included. 

* The sixth part denotes the presence of control distortions by the letter 'D', otherwise it is empty. 

For example, the dataset 'G\_2q\_IX-XI-XX\_IZ-ZI\_N1-N6' is two qubit, Gaussian pulses with no distortions, local X control on each qubit and an interacting XX control, there local Z-noise on each qubit with profile N1 and N6. Another example the dataset 'S\_1q\_XY\_D', is a single-qubit system with square distorted control pulses along X and Y axis, and there is no noise.

| Category | Qubits | Drift      | Control        | Noise      |
|--------------------|------------------|----------------------|--------------------------|----------------------|
| 1        | 1      | (*z*)      | (*x*)          | (*z*)      |
| 2        | 1      | (*z*)      | (*x*,*y*)        | (*x*,*z*)    |
| 3        | 2      | (*z*1, 1*z*) | (*x*1, 1*x*)     | (*z*1, 1*z*) |
| 4        | 2      | (*z*1, 1*z*) | (*x*1, 1*x*, *xx*) | (*z*1,1*z*) |



## QDataSet Parameters

A dictionary of specifications for each example in the QDataSet is set out in table below. Each of the 10,000 examples in each of the 52 datasets is encoded in one of these dictionaries.


| Item                     | Description                                                                                                                                                                                                                                                                                              |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *simulation_parameters* |  *name*: name of the dataset                                                                     |
|  |       *dim*: the dimension 2<sup>*n*</sup> of the Hilbert space for $n$ qubits (dimension 2 for single qubit, 4 for two qubits)                                                                                                                                                                                                                                                                                                                                                                                                             |
|  |       *Ω*: the spectral energy gap                                                                                                                                                                                                                                                                                             |
|  |  *static_operators*: a list of matrices representing the time-independent parts of the Hamiltonian (i.e. drift components)|
|  |  *dynamic_operators*: a list of matrices representing the time-dependent parts of the Hamiltonian (i.e. control components), without the pulses. So, if we have a term *f*(*t*)*σ*<sub>*x*</sub> + *g*(*t*)*σ*<sub>*y*</sub>, this list will be \[*σ*<sub>*x*</sub>,*σ*<sub>*y*</sub>\]|
|  |  *noise_operators*: a list of time-dependent parts of the Hamiltonian that are stochastic (i.e. noise components). so if we have terms like *β*<sub>1</sub>(*t*)*σ*<sub>*z*</sub> + *β*<sub>2</sub>(*t*)*σ*<sub>*y*</sub>, the list will be \[*σ*<sub>*z*</sub>,*σ*<sub>*y*</sub>\]|
|  |  *measurement_operators*: Pauli operators (including identity) (*I*, *σ*<sub>*x*</sub>, *σ*<sub>*y*</sub>, *σ*<sub>*z*</sub>)|
|  |  *initial_states*: the six eigenstates of the Pauli operators|
|  |  *T*: total time (normalised to unity)|
|  |  *num_ex*: number of examples, set to 10,000|
|  |  *batch_size*: size of batch used in data generation (default is 50)|
|  |  *K*: number of randomised pulse sequences in Monte Carlo simulation of noise (set to $K = 2000$)|
|  |  *noise_profile*: N0 to N6 (see paper for detail)|
|  |  *pulse_shape*: Gaussian or Square|
|  |  *num_pulses*: number of pulses per interval|
|  |  *elapsed_time*: time taken to generate the datasets|
| *pulse_parameters*      | The control pulse sequence parameters for the example:                                                                                                                                                                                                                                                            |
| | Square pulses: *A*<sub>*k*</sub> amplitude at time *t*<sub>*k*</sub>|
| | Gaussian pulses: *A*<sub>*k*</sub> (amplitude), *μ* (mean) and $*σ* (standard deviation)|
| *time_range*            | A sequence of time intervals *Δ*(*t*)<sub>*j*</sub> with *j* = 1, ..., *M*                                                                                                                                                                                                                                                      |
| *pulses*                 | Time-domain waveform of the control pulse sequence.                                                                                                                                                                                                                                                               |
| *distorted_pulses*      | Time-domain waveform of the distorted control pulse sequence (if there are no distortions, the waveform will be identical to the undistorted pulses).                                                                                                                                                             |
| *expectations*           | The Pauli expectation values 18 or 52 depending on whether one or two qubits (see above). For each state, the order of measurement is: *σ*<sub>*x*</sub>, *σ*<sub>*y*</sub>, *σ*<sub>*z*</sub> applied to the evolved initial states. As the quantum state is evolving in time, the expectations will range within the interval [1,-1]. |
| *V*<sub>*O*</sub> operator         | The *V*<sub>*O*</sub> operators corresponding to the three Pauli observables, obtained by averaging the operators *W*<sub>*O*</sub> over all noise realizations.                                                                                                                                                                          |
| *noise*                  | Time domain realisations of the relevant noise.                                                                                                                                                                                                                                                                   |
| *H*<sub>0</sub>                  | The system Hamiltonian *H*<sub>0</sub>(*t*) for time-step *j*.                                                                                                                                                                                                                                                                |
| *H*1                   | The noise Hamiltonian *H*<sub>1</sub>(*t*) for each noise realization at time-step *j*.                                                                                                                                                                                                                                       |
| *U*<sub>0</sub>                  | The system evolution matrix *U*<sub>0</sub>(*t*) in the absence of noise at time-step *j*.                                                                                                                                                                                                                                    |
| *U*<sub>*I*</sub>                  | The interaction unitary *U*<sub>*I*</sub>(*t*) for each noise realization at time-step *j*.                                                                                                                                                                                                                                     |
| *V*<sub>0</sub>                  | Set of 3 × 2000 expectation values (measurements) of the three Pauli observables for all possible states for each noise realization. For each state, the order of measurement is: *σ*<sub>*x*</sub>, *σ*<sub>*y*</sub>, *σ*<sub>*z*</sub> applied to the evolved initial states.                                                   |
| *E*<sub>0</sub>                  | The expectations values (measurements) of the three Pauli observables for all possible states averaged over all noise realizations. For each state, the order of measurement is: *σ*<sub>*x*</sub>, *σ*<sub>*y*</sub>, *σ*<sub>*z*</sub> applied to the evolved initial states.                                                           |


## QDataSet Generation

### Hardware specifications

Each dataset in the QDataSet consists of 10,000 examples. An example corresponds to a given control pulse sequence, associated with a set of noise realizations. Every dataset is stored as a compressed zip file, consisting of a number of Python *Pickle* files that stores the information. Each file is essentially a dictionary consisting of the elements described in the table below. The datasets were generated on the University of Technology (Sydney) high-performance computing cluster (iHPC). Each dataset was generated using Singularity containers with Python 3 installed, requiring standard packages including Tensorflow 2.5.0. 

The QDataSet was generated on using the iHPC Mars node (one of 30). The node consists of Intel Xeon Gold 6238R 2.2GHz 28cores (26 cores enabled) 38.5MB L3 Cache (Max Turbo Freq. 4.0GHz, Min 3.0GHz) 360GB RAM. We utilised GPU resources using a NVIDIA Quadro RTX 6000 Passive (3072 Cores, 384 Tensor Cores, 16GB Memory). It took around three months to generate over 2020-2021, coming to around 14TB of compressed quantum data. Single-qubit examples were relatively quick (between a few days and a week or so). The two-qubit examples took much longer, often several weeks.

### Simulation code

Links to the simulation code is set-out in Python scripts which can be found in the 'simulation' subfolder. To run a simulation, run the relevant code notebook. In more detail, the simulation code comrpises:
* *Dataset code*: each of the 52 datasets is generated in one of 26 Python scripts (each script generates the non-distorted and distored examples for the dataset). For example, the script 'dataset_G_1q_X.py' generates the single-qubit dataset with Gaussian control pulses via x-axis control (noise and distortion free)
* *utilities.py*: this script is called by each dataset code and contains general procedures for generating the dataset.
* *simulation.py*: this script is called by utilities.py and contains the details of the quantum simulation.

Please note, the two-qubit simulations tend to consume a lot of RAM. Using Singularity, wrapping a container with Python 3 and Tensorflow 2.5.0 required around 360GB of RAM running for between a few days and one week for most of the two-qubit simulation datasets.

## Dataset link and description

### Dataset Cloud storage

The QDataSet is stored using Cloudstor, a service provided via AARNet for UTS. We set-out the links below:

[QDataSet Cloudstor Link](https://cloudstor.aarnet.edu.au/plus/s/rxYKXBS7Tq0kB8o)

Each dataset is stored in its own separate subfolder. 

### Dataset description

The table below lists the name and description of each dataset.

| Dataset                   | Description                                                                                                                                                                          |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| G_1q_X                         | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: none; (iv) No distortion.                                                                                                     |
| G_1q_X_D                      | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: none; (iv) Distortion.                                                                                                        |
| G_1q_XY                         | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: none; (iv) No distortion.                                                                                                     |
| G_1q_XY_D                      | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: none; (iv) Distortion.                                                                                                        |
| G_1q_XY_XZ_N1N5              | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: N1 on x-axis, N5 on z-axis; (iv) No distortion.                                                              |
| G_1q_XY_XZ_N1N5_D           | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: N1 on x-axis, N5 on z-axis; (iv) No distortion.                                                              |
| G_1q_XY_XZ_N1N6              | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: N1 on x-axis, N6 on z-axis; (iv) Distortion.                                                                 |
| G_1q_XY_XZ_N1N6_D           | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: N1 on x-axis, N6 on z-axis; (iv) No distortion.                                                              |
| G_1q_XY_XZ_N3N6              | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: N1 on x-axis, N6 on z-axis; (iv) Distortion.                                                                 |
| G_1q_XY_XZ_N3N6_D           | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: N1 on x-axis, N6 on z-axis; (iv) No distortion.                                                              |
| G_1q_X_Z_N1                  | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N1 on z-axis; (iv) No distortion.                                                                                           |
| G_1q_X_Z_N1_D               | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N1 on z-axis; (iv) Distortion.                                                                                              |
| G_1q_X_Z_N2                  | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N2 on z-axis; (iv) No distortion.                                                                                           |
| G_1q_X_Z_N2_D               | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N2 on z-axis; (iv) Distortion.                                                                                              |
| G_1q_X_Z_N3                  | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N3 on z-axis; (iv) No distortion.                                                                                           |
| G_1q_X_Z_N3_D               | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N3 on z-axis; (iv) Distortion.                                                                                              |
| G_1q_X_Z_N4                  | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N4 on z-axis; (iv) No distortion.                                                                                           |
| G_1q_X_Z_N4_D               | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N4 on z-axis; (iv) Distortion.                                                                                              |
| G_2q_IX-XI_IZ-ZI_N1-N6       | (i) Qubits: two; (ii) Control: x-axis on both qubits, Gaussian; (iii)                                                                                                                       |
| G_2q_IX-XI_IZ-ZI_N1-N6_D    | (i) Qubits: two; (ii) Control: x-axis on both qubits, Gaussian; (iii) Noise: N1 and N6 z-axis on each qubit; (iv) Distortion.                                                             |
| G_2q_IX-XI-XX                  | (i) Qubits: two; (ii) Control: single x-axis control on both qubits and x-axis interacting control, Gaussian; (iii) Noise: none; (iv) No distortion.                                      |
| G_2q_IX-XI-XX_D               | (i) Qubits: two; (ii) Control: single x-axis control on both qubits and x-axis interacting control, Gaussian; (iii) Noise: none; (iv) Distortion.                                         |
| G_2q_IX-XI-XX_IZ-ZI_N1-N5    | (i) Qubits: two; (ii) Control: single x-axis control on both qubits and x-axis interacting control, Gaussian; (iii) Noise: N1 and N5 on z-axis noise on each qubit; (iv) No distortion. |
| G_2q_IX-XI-XX_IZ-ZI_N1-N5    | (i) Qubits: two; (ii) Control: single x-axis control on both qubits and x-axis interacting control, Gaussian; (iii) Noise: N1 and N5 on z-axis noise on each qubit; (iv) Distortion.    |
| S_1q_X                         | (i) Qubits: one; (ii) Control: x-axis, square; (iii) Noise: none; (iv) No distortion.                                                                                                       |
| S_1q_X_D                      | (i) Qubits: one; (ii) Control: x-axis, Gaussquaresian; (iii) Noise: none; (iv) Distortion.                                                                                                  |
| S_1q_XY                         | (i) Qubits: one; (ii) Control: x-axis and y-axis, square; (iii) Noise: none; (iv) No distortion.                                                                                                       |
| S_1q_XY_D                      | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussquaresian; (iii) Noise: none; (iv) Distortion.                                                                                                  |
| S_1q_XY_XZ_N1N5              | (i) Qubits: one; (ii) Control: x-axis and y-axis, square; (iii) Noise: N1 on x-axis, N5 on z-axis; (iv) No distortion.                                                                |
| S_1q_XY_XZ_N1N5_D           | (i) Qubits: one; (ii) Control: x-axis and y-axis, Gaussian; (iii) Noise: N1 on x-axis, N5 on z-axis; (iv) No distortion.                                                              |
| S_1q_XY_XZ_N1N6              | (i) Qubits: one; (ii) Control: x-axis and y-axis, square; (iii) Noise: N1 on x-axis, N6 on z-axis; (iv) Distortion.                                                                   |
| S_1q_XY_XZ_N1N6_D           | (i) Qubits: one; (ii) Control: x-axis and y-axis, square; (iii) Noise: N1 on x-axis, N6 on z-axis; (iv) No distortion.                                                                |
| S_1q_XY_XZ_N3N6              | (i) Qubits: one; (ii) Control: x-axis and y-axis, square; (iii) Noise: N1 on x-axis, N6 on z-axis; (iv) Distortion.                                                                   |
| S_1q_XY_XZ_N3N6_D           | (i) Qubits: one; (ii) Control: x-axis and y-axis, square; (iii) Noise: N1 on x-axis, N6 on z-axis; (iv) No distortion.                                                                |
| S_1q_X_Z_N1                  | (i) Qubits: one; (ii) Control: x-axis, square; (iii) Noise: N1 on z-axis; (iv) No distortion.                                                                                             |
| S_1q_X_Z_N1_D               | (i) Qubits: one; (ii) Control: x-axis, square; (iii) Noise: N1 on z-axis; (iv) Distortion.                                                                                                |
| S_1q_X_Z_N2                  | (i) Qubits: one; (ii) Control: x-axis, square; (iii) Noise: N2 on z-axis; (iv) No distortion.                                                                                             |
| G_1q_X_Z_N2_D               | (i) Qubits: one; (ii) Control: x-axis, Gaussian; (iii) Noise: N2 on z-axis; (iv) Distortion.                                                                                              |
| S_1q_X_Z_N3                  | (i) Qubits: one; (ii) Control: x-axis, square; (iii) Noise: N3 on z-axis; (iv) No distortion.                                                                                             |
| S_1q_X_Z_N3_D               | (i) Qubits: one; (ii) Control: x-axis, square; (iii) Noise: N3 on z-axis; (iv) Distortion.                                                                                                |
| S_1q_X_Z_N4                  | (i) Qubits: one; (ii) Control: x-axis, square; (iii) Noise: N4 on z-axis; (iv) No distortion.                                                                                             |
| S_1q_X_Z_N4_D               | (i) Qubits: one; (ii) Control: x-axis, square; (iii) Noise: N4 on z-axis; (iv) Distortion.                                                                                                |
| S_2q_IX-XI_IZ-ZI_N1-N6       | (i) Qubits: two; (ii) Control: x-axis on both qubits, square; (iii) Noise: N1 and N6 z-axis on each qubit; (iv) No distortion.                                                            |
| S_2q_IX-XI_IZ-ZI_N1-N6_D    | (i) Qubits: two; (ii) Control: x-axis on both qubits, square; (iii) Noise: N1 and N6 z-axis on each qubit; (iv) Distortion.                                                               |
| S_2q_IX-XI-XX                  | (i) Qubits: two; (ii) Control: single x-axis control on both qubits and x-axis interacting control, square; (iii) Noise: none; (iv) No distortion.                                        |
| S_2q_IX-XI-XX_D               | (i) Qubits: two; (ii) Control: single x-axis control on both qubits and x-axis interacting control, square; (iii) Noise: none; (iv) Distortion.                                           |
| S_2q_IX-XI-XX_IZ-ZI_N1-N5    | (i) Qubits: two; (ii) Control: x-axis on both qubits and x-axis interacting control, square; (iii) Noise: N1 and N5 z-axis on each qubit; (iv) No distortion.                           |
| S_2q_IX-XI-XX_IZ-ZI_N1-N5_D | (i) Qubits: two; (ii) Control: x-axis on both qubits and x-axis interacting control, square; (iii) Noise: N1 and N5 z-axis on each qubit; (iv) Distortion.                              |
| S_2q_IX-XI-XX_IZ-ZI_N1-N6    | (i) Qubits: two; (ii) Control: x-axis on both qubits and x-axis interacting control, square; (iii) Noise: N1 and N6 z-axis on each qubit; (iv) No distortion.                           |
| S_2q_IX-XI-XX_IZ-ZI_N1-N6_D | (i) Qubits: two; (ii) Control: x-axis on both qubits and x-axis interacting control, square; (iii) Noise: N1 and N6 z-axis on each qubit; (iv) Distortion.                              |



```python

```
