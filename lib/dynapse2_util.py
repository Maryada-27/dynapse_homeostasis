import itertools
import re
from samna.dynapse2 import Dynapse2Destination
import time
from lib.dynapse2_obj import *
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def update_fine(parameters, name, fine):
    parameter = parameters[name]
    # parameter.coarse_value = coarse
    parameter.fine_value = fine
    return (parameter.coarse_value,parameter.fine_value)

def get_parameter(parameters, name):
    parameter = parameters[name]
    coarse = parameter.coarse_value 
    fine = parameter.fine_value
    return (coarse,fine)

def set_parameter(parameters, name, coarse, fine):
    parameter = parameters[name]
    parameter.coarse_value = coarse
    parameter.fine_value = fine

def set_multi_parameters(parameter,params_dict):
    for key,val in params_dict.items():
        set_parameter(parameter, key, val[0], val[1])

def set_dc_latches(config, neurons, cores, chips=range(1),dc = True):
    grid = dict(chip = chips,core = cores,neuron = neurons)
    for values in itertools.product(*grid.values()):
        temp = dict(zip(grid.keys(), values))
        config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_so_dc = dc
    
def set_nmda_latches(config, neurons, cores, chips=range(1),nmda=False):
    grid = dict(chip = chips,core = cores,neuron = neurons)
    for values in itertools.product(*grid.values()):
        temp = dict(zip(grid.keys(), values))
        config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_denm_nmda = nmda
        # config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_coho_ca_mem = True

def set_alpha_latches(config, neurons, cores, chips=range(1),alpha=False):
    grid = dict(chip = chips,core = cores,neuron = neurons)
    for values in itertools.product(*grid.values()):
        temp = dict(zip(grid.keys(), values))        
        config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_deam_alpha = alpha
        # config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_coho_ca_mem = True

def set_adap_latches(config, neurons, cores, chips=range(1),adaptation= True):
    grid = dict(chip = chips,core = cores,neuron = neurons)
    for values in itertools.product(*grid.values()):
        temp = dict(zip(grid.keys(), values))
        config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_so_adaptation = adaptation

#latch_de_conductance
def set_conductance_latches(config, neurons, cores, chips=range(1),conductance=False):
    grid = dict(chip = chips,core = cores,neuron = neurons)
    for values in itertools.product(*grid.values()):
        temp = dict(zip(grid.keys(), values))
        config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_de_conductance = conductance

def set_ntype_latches(config, neurons, cores, chips=range(1), ntype= 'threshold'):
    '''
    default: ntype= 'threshold'
    option: ntype=='not-threshold' or ntype= 'threshold'
    '''
    grid = dict(chip = chips,core = cores,neuron = neurons)
    if ntype=='not-threshold':
        print('not-threshold')        

        for values in itertools.product(*grid.values()): # 
            temp = dict(zip(grid.keys(), values))
            config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_soif_type = True
    else:
        print('threshold')        
        for values in itertools.product(*grid.values()): # 
            temp = dict(zip(grid.keys(), values))
            config.chips[temp['chip']].cores[temp['core']].neurons[temp['neuron']].latch_soif_type = False

def clear_srams(config, neurons, cores, chips=range(1), all_to_all=False, source_cores=None, monitor_cam=0):
    # an option to differentiate between source cores (cores that send out to other neurons on chip)
    if source_cores is None:
        source_cores = cores
    assert(not all_to_all or len(chips) <= 3)
    if all_to_all:
        assert (len(chips) <= 3)
        for h in chips:
            for c in cores:
                for n in neurons:
                    config.chips[h].cores[c].neurons[n].destinations = \
                        [Dynapse2Destination()] * monitor_cam + \
                        [DestinationConstructor(tag=c*256+n, core=[True]*4, x_hop=-7).destination] + \
                        [Dynapse2Destination()] * (3 - monitor_cam)
            for c in source_cores:
                for n in neurons:
                    config.chips[h].cores[c].neurons[n].destinations = \
                        [DestinationConstructor(tag=c*256+n, core=[True]*4, x_hop=-7).destination] + \
                        [DestinationConstructor(tag=c*256+n, core=[i in cores for i in range(4)], x_hop=t - h).destination for t in chips] + \
                        [Dynapse2Destination()] * (3 - len(chips))
    else:
        for h in chips:
            for c in cores:
                for n in neurons:
                    config.chips[h].cores[c].neurons[n].destinations = \
                        [DestinationConstructor(tag=c*256+n, core=[True]*4, x_hop=-7).destination] + \
                        [Dynapse2Destination()] * 3