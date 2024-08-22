This repository contains the code for "Stable recurrent dynamics in heterogeneous neuromorphic computing systems using excitatory and inhibitory plasticity." Maryada, Saray Soldado-Magraner, Martino Sorbaro, Rodrigo Laje, Dean V. Buonomano, Giacomo Indiveri bioRxiv 2023.08.14.553298; doi: https://doi.org/10.1101/2023.08.14.553298 

This python code create a neural network in accordance with the architecture of DYNAP-SE 2 chip. Briefly,the chip is a mixed-signal multi-core neuromorphic chip with 256 neurons-per-core. Each neuron has limited number of synapse connected to it (fan-in). 
# Dependancies
**This code will only work if the DYNAP-SE 2 chip with correct serial number is connected.**
Install `samna-0.14.25.0` using `pip install samna ==0.14.25.0`(todo: check version). Additional standard package required are: numpy,pandas and matplotlib. 

# How to run 
`python3 main.py ./bitfiles/Dynapse2Stack.bit 1 orange` (or green) where Dynapse2Stack.bit is the firmware file that runs on OpalKelly FPGA. orange/green is a name set for different chip to identify them. It's critical as each chip is different and each chip has a corrosponding json file (see CAMs folder). Check the chip for the colored marking and pass the label to load the correct file.

# main.py 

The following are the serial number of FPGAs connected to the chip and to the computer running this code. Change the serial numbers if new device is used. Check line 47

```
greenSN = '1911000P6X'
orangeSN = '2042000UO5'
```

Set the folder to save the data and plots `parent_dir`, see line 67 
There are two possibiliies:
1. Load exsiting csv for inital weights. (see folder EI, EI_init_weight.csv)
2. Randomly select new inital weights.

# /lib

The folder holds essential files essential to communicate with the chip and create netwojsork graphs that can be deployed on the chip.   

1. *dynapse2_network.py* : Functions to create network based on loaded json file for a specific chip.
2. *dynapse2_util.py*: Functions to set and get various parameters, improtant: get function read the loaded congif file on the computer running the code and not from the chip.
3. *dynapse2_spikegen.py* :  Functions to generate spike input to pass to the chip via FPGA. For input, Digital input neuron (0-1023) are created on FPGA.
4. *dynapse2_raster.py* : Function **get_events** read out the event set by the chip via FPGA. Other function are helper function to plot these events.
5. *dynapse2_obj.py* : DO NOT CHANGE.  Only of concern for hardware designers.
6. *dynapse2_init.py* : DO NOT CHANGE. Only of concern for hardware designers.

# /CAMs
Contains json files for different chips containins memory bit information. Please make sure correct file is loaded when a chip is connected. 

# /bitfiles
Contains firmware file for stack and dev board, stack boards are used in this work. 

# /experiments/helper.py
Wrapper functions to make it easier to use the internal functions of the chip. 

# /EI 

1. *network_config.py* : This is the configuration file that contains the details for a EI network and various test cases.
2. *network_configST.py*: This file is only for *sanity test* purpose.
3. *network_config_multi_attractor.py/_small.py*: This file is similar to the `network_config` but altered for multi_attractor test case.
4. *core_bias.py*: Neuron and synapse parameters based on neuron type.
5. *learning.py*: Implementation of Cross-homeostasis and Owen Mackwood, Laura B Naumann, Henning Sprekeler rule. For future extension, Adam and Momentum is implemented but not used.
6. *Input_Freq.py*: Experiment to generate *Input-Frequency* curve.
7. *EI_homeostasis.py/sWTA_post_training.py*: Contains all necessary functions for classic EI_homeostasis and soft-WTA experiments.
8.  
