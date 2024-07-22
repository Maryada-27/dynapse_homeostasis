
import itertools
import os
from pydoc import Helper
import re
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/experiments/')
sys.path.append(os.getcwd() + '/experiments/EI/')

from .core_bias import CoreBias 
import network_configST as  netConfig
import helper
from lib.dynapse2_network import Network
from lib.dynapse2_obj import *
from lib.dynapse2_raster import *
import lib.dynapse2_spikegen as DYN2spikegen
import lib.dynapse2_util as DYN2util
from samna.dynapse2 import *



def config_connections(network, virtual_group_size = 1, input_channels=1):
    #TODO: group vs channel
 
    input_groups = [network.add_virtual_group(size=virtual_group_size) for _ in range(input_channels)]
  
    groups = dict()
    for ntype in netConfig.neuronTypes: # neuronTypes is array that contains the neuron types that user intend to create
       
        group = network.add_group(chip=netConfig.allocated_chips[ntype], 
                                  core=netConfig.allocated_cores[ntype], 
                                  size=netConfig.population[ntype],
                                  name = ntype)
        groups.update({ntype : group})
        
    #---------------------------------SET  CONNECTIONS HERE-----------------------------    
    for input_group in input_groups:

        network.add_connection(source=input_group, target=groups["Pyr"], probability=1.,
                               dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=1,precise_delay=True)
        network.add_connection(source=input_group, target=groups["PV"], probability=1.,
                               dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=1,precise_delay=True)
        network.add_connection(source=input_group, target=groups["SST"], probability=1.,
                               dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=1,precise_delay=True)
    network.connect()    
    print('Connectivity done')

def init_network(myConfig,model,profile_path,number_of_chips,data_path,boards,tname):
    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips,boards=boards)
    #TODO: more input details to network_config.py
    config_connections(network=network, virtual_group_size=netConfig.population['CH_neurons'], input_channels=netConfig.population['CH'])
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    
    df_network = pd.DataFrame()
    for _,group in enumerate(network.groups):
        data = [[i,group.name] for i in group.ids]
        df_network = pd.concat([df_network,pd.DataFrame(data=data,columns=['Id','Type'])])
    
    for _,group in enumerate(network.virtual_groups):
        data = [[i,"Input"] for i in group.ids]
        df_network = pd.concat([df_network,pd.DataFrame(data=data,columns=['Id','Type'])])
    df_network.to_hdf(f'{data_path}/Network_{tname}.h5',key=f'Neurons')
    
    return network,df_network


def experiment_run(board,dir_path,board_names=["orange"], number_of_chips=1, profile_path="../profiles/",tname='Refactory_period'):
    # dir_path = f"./onchip_experimentaldata/board_{'-'.join(board_names)}/{date_label}/{tname}/{time_label}"
    # tname = "AMPA_timeconstant"#"Neuron_timeconstant"#"Refactory_period"#"conductance_based_syn"
    
    data_path = f'{dir_path}/data'
    imgdir = f"{dir_path}/img"
    os.makedirs(f"{data_path}",exist_ok=True)
    os.makedirs(f"{imgdir}",exist_ok=True)
    # input_type,rate,isi_spike,DC ="poisson_gen", 100.,0.,(0,0)
    input_type,isi_spike,rate,DC  ="spike_gen",.015e6,0.,(0,0)
    # input_type,isi_spike,rate,DC = "DC",0,0,(1,120)
    
    #TODO: call functions and remove inline code.
    #---------------------------reset_boardConfig------------------------------------------------
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(.1)
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(.1)

    print("Configuring parameters")
    core_specs = [CoreBias.get_module_params(neuron_type=nType) for nType in netConfig.neuronTypes]
    chipIds = [netConfig.allocated_chips[nType] for nType in netConfig.neuronTypes]
    coreIds = [netConfig.allocated_cores[nType] for nType in netConfig.neuronTypes]
    helper.config_parameters(myConfig=myConfig,chipIds=chipIds,coreIds=coreIds,core_specs=core_specs)
    model.apply_configuration(myConfig)
    time.sleep(0.2)
    #-------------------------------set neuron type and DC latch--------------------------------------------
    DYN2util.set_ntype_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),ntype='threshold')
    DYN2util.set_conductance_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),conductance=netConfig.SYN_CONDUCTANCE)
    DYN2util.set_dc_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),dc = netConfig.ENABLE_DC)
    DYN2util.set_adap_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),adaptation= netConfig.ENABLE_ADAPTATION)
    DYN2util.set_nmda_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),nmda= netConfig.ENABLE_NMDA_GATING)
    model.apply_configuration(myConfig)
    time.sleep(0.2)
    #--------------------------------------init_network---------------------------------
    network,df_network = init_network(myConfig,model,profile_path,number_of_chips,data_path,board_names,tname=tname)
    neuronE = df_network[df_network.Type == "Pyr"].Id.values
    neuronI = df_network[df_network.Type == "PV"].Id.values
    #---------------------------------------------------------------------------------------------------

    print([_.tag for _ in myConfig.chips[netConfig.allocated_chips['Pyr']].cores[netConfig.allocated_cores['Pyr']].neurons[131].synapses])
    print("\nAll configurations done!\n")
    helper.save_config(myConfig,number_of_chips,data_path,tname,list(netConfig.allocated_cores.values()))
    #-------------------------------------READ SPIKE DATA-----------------------------------------
    
    iteration_steps = 10
    netConfig.stimulus_duration = netConfig.stimulus_adap
    sim_duration = netConfig.stimulus_duration['StimulusON'] + netConfig.stimulus_duration['StimulusOFF']

    params = dict(Refactory_period = dict(pname = 'SOIF_REFR_N',
                                          coarse_range = np.arange(0,6,1), 
                                          fine_range = np.arange(10,200,iteration_steps)),
                  Neuron_tau = dict(pname = 'SOIF_LEAK_N',
                                    coarse_range = np.arange(0,6,1), 
                                    fine_range = np.arange(10,200,iteration_steps)),
                  AMPA_tau = dict(pname = 'DEAM_ETAU_P',
                                    coarse_range = np.arange(0,6,1),
                                    fine_range = np.arange(10,200,iteration_steps)),
                  conductance_based_synapse = dict(pname = 'DEAM_REV_N',
                                                    coarse_range = 2,
                                                    fine_range = np.arange(10,200,iteration_steps)),
                  adaptation = dict(pname = 'SOAD_PWTAU_N',
                                    coarse_range = np.arange(0,3,1),
                                    fine_range = np.arange(200,200+iteration_steps,iteration_steps)),
                )
    input_value = isi_spike
   
    for coarse in params[tname]['coarse_range']:
        for fine in params[tname]['fine_range']:
            print(f'Running {tname} = ({coarse},{fine})')
            for ntype in ['SST']:#netConfig.neuronTypes: 
                DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters,params[tname]['pname'],coarse,fine)
            model.apply_configuration(myConfig)
            time.sleep(0.1)
            firing_rate = pd.DataFrame()
            for repeat in range(5):
                time.sleep(.5)
                # set monitoring of one or more neurons
                for gid,ntype in enumerate(netConfig.neuronTypes):
                    nid = repeat#np.random.randint(0,netConfig.population[ntype])
                    print(f'Monitoring {ntype} :: {nid}')
                    myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].monitored_neuron = network.groups[gid].neurons[nid]
                    myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].neuron_monitoring_on = True
                model.apply_configuration(myConfig)
                time.sleep(0.1)
                #--------------- CLEAR BUFFER------------------------------------
                output_events = [[], []]
                DYN2spikegen.send_virtual_events(board=board, virtual_events=[])
                get_events(board=board, extra_time=100, output_events=output_events)
                #--------------- CLEARED BUFFER------------------------------------
                ts = DYN2spikegen.get_fpga_time(board=board) + (int)(.2e6) # get current time of FPGA
                input_events = []
                if input_type =='DC':
                    for ntype in netConfig.neuronTypes:
                        DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters,"SOIF_DC_P",DC[0],DC[1])
                    model.apply_configuration(myConfig)
                    time.sleep(netConfig.stimulus_duration['StimulusON']) #TODO: change to correct 
                    for ntype in netConfig.neuronTypes:
                        DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters,"SOIF_DC_P",0,0)
                    model.apply_configuration(myConfig)
                    time.sleep(0.1)
                else:
                    input_events = helper.gen_input_stim(network,netConfig.input_stimuli)
                #------------------- sending events --------------------------------------
                DYN2spikegen.send_virtual_events(board=board, virtual_events=input_events, offset=ts, min_delay=(int)(netConfig. StimulusOFF *1e6))
                #------------------------ READ and plot OUTPUT------------------------------
                output_events = [[], []]
                get_events(board=board, extra_time=100, output_events=output_events)# extra_time is to clear the buffer
                helper.reset_TAU(model, myConfig,netConfig)
                if len(output_events[0])> len(input_events):
                    print('Spike response of neurons ',len(output_events[0])-len(input_events))
                    # print(output_events)
                    df_events = pd.DataFrame.from_records(output_events).T
                    df_events = df_events.rename(columns={0: 'Id', 1: "Timestamp"}).astype({"Id": int})
                    df_events = df_events.set_index('Id').join(df_network.set_index('Id'))
                    df_events["Timestamp"] = df_events["Timestamp"] - output_events[1][0]
                    df_events["Input"] = int(input_value)
                    df_events["Repeat"] = int(repeat)
                    df_events['Coarse'] = int(coarse)
                    df_events['Fine'] = int(fine)
                    df_events['pname'] = params[tname]['pname']
                    df_events.to_hdf(f'{data_path}/NetworkActivity_{tname}.h5',append=True,key=f'{tname}')
                    plot_raster(output_events,imgdir,f'({coarse}_{fine}_{input_value})_{repeat}',sim_duration)
                    #---------------------get spike time and firing rate for both population
                    Emask =np.isin(output_events[0],neuronE,assume_unique=False)
                    E_spike_time= np.array(output_events[1])[Emask]
                    Imask =np.isin(output_events[0],neuronI,assume_unique=False)
                    I_spike_time= np.array(output_events[1])[Imask]
                    Ispike_count = len(I_spike_time)                
                    Espike_count = len(E_spike_time)
                    AvgE = Espike_count/(netConfig.population['Pyr'] * sim_duration)
                    AvgI = Ispike_count/(netConfig.population['PV'] * sim_duration)
                    print(f'{DYN2util.bcolors.OKGREEN} {input_type=}, {input_value=}  {fine=}| {AvgE=},{AvgI=}{DYN2util.bcolors.ENDC}')
                    E_df =pd.DataFrame(dict(Run = repeat,Input = input_value,FiringRate = AvgE, Type = 'E', InputType = input_type,coarse = coarse, pname = params[tname]['pname'],fine =fine), index=[0])
                    I_df = pd.DataFrame(dict(Run = repeat,Input = input_value,FiringRate = AvgI, Type = 'I', InputType = input_type,coarse = coarse, pname = params[tname]['pname'],fine =fine), index=[0])
                    firing_rate = pd.concat([firing_rate,E_df,I_df], ignore_index=True)
                    if AvgE > 200 or AvgI > 200:
                        print('Network reached saturation')
                else:
                    print(f'{DYN2util.bcolors.OKBLUE} {input_type=}, {input_value=} | {fine=}| No activity............{DYN2util.bcolors.ENDC}')
                    E_df = pd.DataFrame(dict(Run = repeat,Input = input_value,FiringRate = 0, Type = 'E', InputType = input_type,coarse = coarse, pname = params[tname]['pname'],fine =fine), index=[0])
                    I_df = pd.DataFrame(dict(Run = repeat,Input = input_value,FiringRate = 0, Type = 'I', InputType = input_type,coarse = coarse, pname = params[tname]['pname'],fine =fine), index=[0])
                    firing_rate = pd.concat([firing_rate,E_df,I_df], ignore_index=True)
            firing_rate.to_hdf(f'{data_path}/DC_Frequency.h5',key=f'{tname}')