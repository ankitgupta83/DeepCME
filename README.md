# DeepCME
A deep learning framework for solving the Chemical Master Equation

This is the code accompanying the manuscript titled **DeepCME: A deep learning framework for solving the Chemical Master Equation** by *Ankit Gupta, Christoph Schwab* and *Mustafa Khammash*. It contains the following files:


1. **main.py**: This is the main python script which requires a configuration JSON file as input (see below). Based on this configuration file, a reaction network  example is selected (from ReactionNetworkExamples.py), a deep neural network is trained (if requested), simulation-based output and sensitivity values are estimated (if requested), and the desired plots are generated. 
  
2. **CME_Solver.py**: The file contains subroutines that construct and train the deep neural network (DNN) with a feedforward structure. The architecture of DNN is determined by the parameters defined in the configuration JSON file. The trained DNN is used for output estimation as well as the computation of its parameter sensitivities w.r.t. all reaction network parameters.

3. **ReactionNetworkClass.py**: This file contains a Python class called "ReactionNetworkDefinition" to describe a generic reaction network.

4. **ReactionNetworkExamples.py**: This file contains classes that encode reaction network examples by inheriting the "ReactionNetworkDefinition" class from (ReactionNetworkClass.py). Note that for each example, the output functions must be defined using TensorFlow operations. Currently the file includes the four examples from the manuscript: independent birth death, linear signalling cascade, nonlinear signalling cascade and linear signalling cascade with feedback. More examples can be easily added based on these examples.

5. **data_saving.py**: This file contains subroutines for saving and retrieving training/validation trajectories, saving training history and DeepCME estimated sensitivity values.

6. **simulation_validation.py**: This file contains subroutines for estimating outputs with Monte Carlo simulations (with the stochastic simulation algorithm (SSA)) and the parameter sensitivities (with the Bernoulli path algorithm (BPA)). See the manuscript for more details on these methods. The parameters required for these simulation-based methods are taken from the configuration JSON file. 

7. **plotting.py**: This file contains subroutines for plotting bar charts comparing DeepCME estimated outputs and sensitivity values with those obtained with simulation-based methods and also the exact values (if available). A subroutine for plotting the loss function trajectory is also provided. 

8. **ConfigFileDescription.txt**: This text file describes the configuration JSON file for an example. These configuration files are stored in the Configs subfolder. 

## Command Line Execution Example

```
python main.py independent_birth_death.json
```

## Dependencies

* [TensorFlow >=2.0](https://www.tensorflow.org/)



If you have any questions regarding the code, please contact Ankit Gupta at ankit.gupta@bsse.ethz.ch.
