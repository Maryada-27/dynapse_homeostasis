'''
Network parameters
'''
from samna.dynapse2 import Dendrite
class netConfig:

  neuronTypes = ['Pyr','PV']#,'SST']
  conn_p = 0.35 #0.1 #
  
  cluster_conn_p = 0.35
  cluster_population = 32
  cluster_topology = 's' # 's' for sequential, 'n' for non-sequential   
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
  cluster_count = 5
  population = dict(Pyr = 50,
                    PV = 14,
                    # SST = 50,
                    CH = cluster_count, # number of different Stimulus
                    CH_neurons = 1 # virtual neuron represent one stimulus Note: test the poisson process if gt 1
                  )

  # FAN-OUT CONVENTION
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
  #-----------------------------------------------SET CONFIGURATION-----------------------------------------------
  PARADOXICAL = False
  IMPLANT_MEMORY = False
  LEARNING_ON = True
  IMPLANT_MEMORY_at = None # trial number
  TRAIN_ITERATIONS = 400
  TEST_ITERATIONS = 1 # SET TO 0 IF NO TEST
  REPEAT_COUNT = cluster_count
  LEARNING_RATE = 0.005
  ENABLE_DC = False
  ENABLE_ADAPT = False
  ENABLE_NMDA_GATING = False
  DC = (1,120)
  KICK_ON  = .04
  DC_ON = KICK_ON
  # to stimulate one cluster at a time, each repeat will stimulate one cluster
  input_stimuli =dict(CH_Type = cluster_count * ['spike_gen'],
                    rate_isi = cluster_count * [0.01],
                    start_time = cluster_count *[0.],
                    StimulusON = cluster_count * [KICK_ON],
                    Status = [False,False,False,False,False],
                    )# in start,duration (ON duration),isi in seconds and rate in Hz
  StimulusOFF = 1. # time of last event +  StimulusOFF =  simulation_duration
  IN_DEGREE: int = 64
  N_PER_CORE: int = 256
  N_CORES: int = 4
  N_PER_CHIP: int = 1024  # N_PER_CORE * N_CORES
  E_TARGET = 40#25# 20#  200/100 neurons: 20, 64 neurons: 25 Hz and 32 neurons: 35 Hz
  I_TARGET = 2 * E_TARGET
  CONVERGENCE_STD = 1. # ms
  CROSSH = True
  STATIC_INH = False # if True, inhibitory synapse are not updated
  STATIC_EXC = False # if True, excitatory synapse are not updated