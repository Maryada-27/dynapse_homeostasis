'''
Network parameters
'''
from samna.dynapse2 import Dendrite
import numpy as np
class netConfig:
  neuronTypes = ['Pyr','PV']#,'SST']
  conn_p = 0.1 # 
  memory_count = 0 # Change for memory
  memory_conn_p = 0.35 #Change for memory
  memory_size = 50 # Change for memory
  memory_topology = 's' # 's' for sequential, 'n' for non-sequential   
  cluster_count = 1 # For multicluster demonstration, do not change in this file

  #TODO: add input part here as well
  allocated_cores = dict(Pyr = 2, #  chip = 0
                        PV  = 1,#  chip = 0
                        # SST = 3
                        )
  allocated_chips = dict(Pyr = 0,
                         PV  = 0,
                        # SST = 0
                        )
  #-----------------------------------------------
  population = dict(Pyr = 200,
                    PV = 50,
                    # SST = 50,
                    CH = 1 + memory_count, # number of different Stimulus
                    CH_neurons = 1 # virtual neuron represent one stimulus Note: test the poisson process if gt 1
                  )
  
  # FAN-OUT CONVENTION
  # rec_cluster is for memory
  connectivity = dict(Pyr = dict(Pyr = dict(
                            weight = [False,True,False,False],
                            conn_type="prob",#'prob',#"fan_out",#"fan_in",'rec_cluster'
                            rW=1,cW=3,# weight bits for all option 
                            synapse = Dendrite.nmda,
                            p = conn_p, w_name = 'SYAM_W1_P', # weight updates with learning rule
                            ), 
                            PV = dict(
                            weight = [False,True,False,False],
                            rW=1, # same as weight bit vector
                            conn_type='prob',
                            synapse = Dendrite.ampa,
                            p = conn_p, w_name = 'SYAM_W1_P')
                          ),
                      PV = dict(Pyr = dict(
                            weight = [False,False,True,False],
                            rW=2, 
                            conn_type='prob',
                            synapse = Dendrite.shunt,
                            p = conn_p, w_name = 'SYAM_W2_P',
                            ),
                            PV = dict(
                            weight = [False,False,True,False],
                            rW=2, 
                            conn_type='prob',
                            synapse = Dendrite.shunt,
                            p = conn_p,  w_name = 'SYAM_W2_P')
                          )
                      )

  #-------------------------------------------SET CONFIGURATION-------------------------------------
  PARADOXICAL = False
  IMPLANT_MEMORY = False
  LEARNING_ON = True
  IMPLANT_MEMORY_at = 0 # trial number
  MEMORY_WEIGHT_BIAS =(3,65) #(3,65) #
  TRAIN_ITERATIONS = 400
  TEST_ITERATIONS = memory_count # SET TO 0 IF NO TEST
  REPEAT_COUNT = 5
  LEARNING_RATE = 0.005 * 1
  ENABLE_DC = False
  ENABLE_ADAPT = False
  ENABLE_NMDA_GATING = False
  DC = (1,120)
  KICK_ON  = .05
  KICK_ISI = 0.01
  MEMORY_STIMULUS_RATE = 0. #300 #Hz
  MEMORY_OFF_RATE = MEMORY_STIMULUS_RATE//5
  MEMORY_DURATION = 0#1.5 # in seconds. It's duration of memory stimulus not absoulte time
  EPOCH_DURATION = MEMORY_DURATION * memory_count # in seconds
  MEMORY_ONSET = np.linspace(0.01,EPOCH_DURATION,memory_count)#[0.1, 0.1+2.0, (0.01+2.0)+1.5 ]#
  DC_ON = KICK_ON
  input_stimuli =dict(CH_Type = ['spike_gen'] + memory_count * ['poisson_gen'],# in this order: KICK, PARADOXICAL or IMPLANT_MEMORY
                    rate_isi = [KICK_ISI] + memory_count * [MEMORY_STIMULUS_RATE], # PARADOXICAL is off
                    low_rate_isi = [0.] + memory_count * [MEMORY_OFF_RATE],
                    start_time = [0.] + list(MEMORY_ONSET),
                    StimulusON = [KICK_ON] + memory_count * [MEMORY_DURATION], #[2.,1.5,1.5],
                    Status = [True]+ memory_count * [False],
                    )# in start,duration (ON duration),isi in seconds and rate in Hz
  StimulusOFF = 1. # time of last event +  StimulusOFF =  simulation_duration
  IN_DEGREE: int = 64
  N_PER_CORE: int = 256
  N_CORES: int = 4
  N_PER_CHIP: int = 1024  # N_PER_CORE * N_CORES
  E_TARGET = 20 # 40#25#  200/100 neurons: 20, 64 neurons: 25 Hz and 32 neurons: 40 Hz
  I_TARGET = 2 * E_TARGET
  CONVERGENCE_STD = 1. # ms
  CROSSH = True
  STATIC_INH = False # if True, inhibitory synapse are not updated
  STATIC_EXC = False # if True, excitatory synapse are not updated