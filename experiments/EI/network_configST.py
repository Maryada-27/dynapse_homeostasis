'''
Network parameters for sanity test
'''
from samna.dynapse2 import *

#TODO: add input part here as well
allocated_cores = dict(Pyr = 2, #  chip = 0
                       PV  = 1,#  chip = 0
                       SST = 3
                       )
allocated_chips = dict(Pyr = 0,
                       PV  = 0,
                       SST = 0
                       )
#-----------------------------------------------
population = dict(Pyr = 200,
                  PV = 200,
                  SST = 200,
                  CH = 2, # number of different Stimulus
                  CH_neurons = 1 # virtual neuron represent one stimulus
                )
neuronTypes = ['Pyr','PV']#,'SST']
conn_p = 0.1
ENABLE_DC = True
DC = (1,120)
connectivity = dict()
stimulus_FF = dict(StimulusON = 1., StimulusOFF = .001)# in seconds
stimulus_tau = dict(StimulusON = .1, StimulusOFF = .1)# in seconds
stimulus_adap = dict(StimulusON = 1., StimulusOFF = .001)# in seconds
input_stimuli =dict(CH_Type = ['poisson_gen','poisson_gen'],
                   rate_isi = [150.,180],
                   start_time = [0.,1.001],
                   StimulusON = [1.,1.],
                   Status = [True,True],
                   )# in start,duration (ON duration),isi in seconds and rate in Hz
IN_DEGREE: int = 64
N_PER_CORE: int = 256
N_CORES: int = 4
N_PER_CHIP: int = 1024  # N_PER_CORE * N_CORES
StimulusOFF = 1.
SYN_CONDUCTANCE = False
# ENABLE_NMDA_GATING_CORE = []
ENABLE_NMDA_GATING = False
# ENABLE_ADAPTATION_CORE = [3]
ENABLE_ADAPTATION = False
