'''
Network parameters
'''
from samna.dynapse2 import Dendrite
class netConfig:
    
  #TODO: add input part here as well
  allocated_cores = dict(Pyr = 2, #  chip = 0
                        PV  = 1#  chip = 0
                        )
  allocated_chips = dict(Pyr = 0,
                        PV  = 0
                        )
  #-----------------------------------------------
  population = dict(Pyr = 48,
                    PV = 12,
                    CH = 2, # number of different Stimulus
                    CH_neurons = 1 # virtual neuron represent one stimulus Note: test the poisson process if gt 1
                  )
  neuronTypes = ['Pyr','PV']
  conn_p = 0.35 #0.1 #

  connectivity = dict(Pyr = dict(Pyr = dict(
                            weight = [False,True,False,False],
                            conn_type="prob",#'prob',#"fan_out",#"fan_in",'rec_cluster', 'cluster'
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
                            rW=2, # same as weight bit vector
                            conn_type='prob',
                            synapse = Dendrite.shunt,
                            p = conn_p, w_name = 'SYAM_W2_P',
                            ),
                            PV = dict(
                            weight = [False,False,True,False],
                            rW=2, # same as weight bit vector
                            conn_type='prob',
                            synapse = Dendrite.gaba,
                            p = conn_p,  w_name = 'SYAM_W2_P')
                          )
                      )

  #-----------------------------------------------SET CONFIGURATION-----------------------------------------------
  LEARNING_ON = True
  PARADOXICAL = False
  IMPLANT_MEMORY = False
  IMPLANT_MEMORY_at = None # trial number
  TRAIN_ITERATIONS = 200
  TEST_ITERATIONS = 0
  REPEAT_COUNT = 5
  LEARNING_RATE = 0.005 * 1
  ENABLE_DC = False
  ENABLE_ADAPT = False
  ENABLE_NMDA_GATING = False
  DC = (1,120)
  KICK_ON  = .04
  DC_ON = KICK_ON
  input_stimuli =dict(CH_Type = ['spike_gen','poisson_gen'],
                    rate_isi = [0.01,0.],
                    start_time = [0.,0.],
                    StimulusON = [KICK_ON,0.],
                    Status = [True,False],
                    )# in start,duration (ON duration),isi in seconds and rate in Hz
  StimulusOFF = 1.5 # time of last event +  StimulusOFF =  simulation_duration
  IN_DEGREE: int = 64
  N_PER_CORE: int = 256
  N_CORES: int = 4
  N_PER_CHIP: int = 1024  # N_PER_CORE * N_CORES
  E_TARGET = 40#25#20#   200/100 neurons: 20, 64 neurons: 25 Hz and 32 neurons: 35 Hz
  I_TARGET = 1.5 * E_TARGET