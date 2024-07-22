
# import itertools
import os
# import re
import sys
import time
from datetime import datetime

import pandas as pd


sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/experiments/')
sys.path.append(os.getcwd() + '/experiments/EI/')


from lib.dynapse2_network import Network
from lib.dynapse2_obj import *
from lib.dynapse2_raster import *
import lib.dynapse2_spikegen as DYN2spikegen
import lib.dynapse2_util as DYN2util  
from samna.dynapse2 import *

from .core_bias import CoreBias 
import network_configST as  netConfig
import helper


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
    
    for input_group in input_groups:
        for ntype in netConfig.neuronTypes:
            network.add_connection(source=input_group, target=groups[ntype], probability=1.,
                                   dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=1,precise_delay=True)
        # network.add_connection(source=input_group, target=groups["Pyr"],dendrite=Dendrite.ampa, weight=None, repeat=1,matrix =Pyr_conn_matrix)
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


def experiment_run(board,dir_path,board_names=["orange"], number_of_chips=1, profile_path="../profiles/",tname="Input_FrequencyCurve" ):
    data_path = f'{dir_path}/data'
    imgdir = f"{dir_path}/img/{tname}"
    os.makedirs(f"{data_path}",exist_ok=True)
    os.makedirs(f"{imgdir}",exist_ok=True)
    # input_type,rate,isi_spike,DC ="poisson_gen", 100.,0.,(0,0)
    # input_type,isi_spike,rate,DC  ="spike_gen",.01e6,0.,(0,0)
    # input_type,isi_spike,rate,DC = "DC",0,0,(2,120)
    rate,isi_spike,DC = 0., .02e6, 0.
    input_type = "DC"#"spike_gen"#"poisson_gen" #
    sim_duration = netConfig.stimulus_FF['StimulusON'] + netConfig.stimulus_FF['StimulusOFF']
    
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
    helper.config_parameters(myConfig=myConfig,chipIds=chipIds, coreIds=coreIds,core_specs=core_specs)
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    #-----------------------------set neuron type and DC,adap,nmda,conductance latch--------------------------------------------
    DYN2util.set_ntype_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),ntype='threshold')
    DYN2util.set_conductance_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),conductance=netConfig.SYN_CONDUCTANCE)
    DYN2util.set_dc_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),dc = netConfig.ENABLE_DC)
    DYN2util.set_adap_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),adaptation= netConfig.ENABLE_ADAPTATION)
    DYN2util.set_nmda_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips),nmda= netConfig.ENABLE_NMDA_GATING)
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    #--------------------------------------init_network---------------------------------
    network,df_network = init_network(myConfig,model,profile_path,number_of_chips,data_path,board_names,tname)
    # For deugging
    neuronPyr = df_network[df_network.Type == "Pyr"].Id.values
    neuronPV = df_network[df_network.Type == "PV"].Id.values
    neuronSST = df_network[df_network.Type == "SST"].Id.values
    # set monitoring of one or more neurons
    for gid,ntype in enumerate(netConfig.neuronTypes):
        nid = 2#np.random.randint(0,netConfig.population[ntype])
        print(f'Monitoring {nid}')
        myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].monitored_neuron = network.groups[gid].neurons[nid]
        myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].neuron_monitoring_on = True
    # myConfig.chips[0].cores[2].enable_pulse_extender_monitor2 = True
    myConfig.chips[0].cores[2].enable_pulse_extender_monitor2 = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    #---------------------------------------------------------------------------------------------------
    print([_.tag for _ in myConfig.chips[netConfig.allocated_chips['Pyr']].cores[netConfig.allocated_cores['Pyr']].neurons[131].synapses])
    print("\nAll configurations done!\n")
    allocated_cores = list(netConfig.allocated_cores.values())
    allocated_chips = list(netConfig.allocated_chips.values())
    helper.save_config(myConfig,data_path,tname,allocated_chips,allocated_cores)
    #------------------------------------READ SPIKE DATA--------------------------------------
    DC_coarse = 3
    DC_start,isi_start,poisson_start = 10, 0.020e6, 100
    DC_end,isi_end,poisson_end = 250,.019e6,500
    DC_step, isi_step, poisson_step = 10,-0.001e6,100

    netConfig.stimulus_duration = netConfig.stimulus_FF
    for DC in np.arange(DC_start,DC_end,DC_step):
    # for rate in np.arange(poisson_start,poisson_end,poisson_step):# if poisson_gen
    # for isi_spike in np.arange(isi_start,isi_end,isi_step):# if spike_gen
        # set monitoring of one or more neurons
        input_value = DC #isi_spike#rate#
        start_time = time.time()
        firing_rate = pd.DataFrame()
        for repeat in range(1):
            time.sleep(0.5)
            #--------------- CLEAR BUFFER------------------------------------
            output_events = [[], []]
            DYN2spikegen.send_virtual_events(board=board, virtual_events=[])
            get_events(board=board, extra_time=100, output_events=output_events)
            #--------------- CLEARED BUFFER------------------------------------
            ts = DYN2spikegen.get_fpga_time(board=board) + (int)(.2e6) # get current time of FPGA
            input_events = []
            if input_type =='DC':
                for ntype in netConfig.neuronTypes:
                    DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters,"SOIF_DC_P",DC_coarse,DC)
                model.apply_configuration(myConfig)

                time.sleep(netConfig.stimulus_FF['StimulusON'])               
               
                for ntype in netConfig.neuronTypes:
                    DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters,"SOIF_DC_P",0,0)
                model.apply_configuration(myConfig)
                time.sleep(0.1)
            else:
                input_events = helper.stimulus(netConfig,input_type,ts,network,[rate],isi_spike)
            #------------------- sending events -----------------------------                
            # print(input_events)
            DYN2spikegen.send_virtual_events(board=board, virtual_events=input_events, offset=0, min_delay=(int)(netConfig.stimulus_FF['StimulusOFF']*1e6))
            #------------------------ READ and plot OUTPUT--------------------------
            output_events = [[], []]
            get_events(board=board, extra_time=100, output_events=output_events)# extra_time is to clear the buffer
            helper.reset_TAU(model,myConfig,netConfig)
            print(f'{DYN2util.bcolors.OKGREEN} Events are now available for further processing {repeat=}{DYN2util.bcolors.ENDC}')
            if len(output_events[0])> len(input_events):
                print('Spike response of neurons ',len(output_events[0])-len(input_events))
                # print(output_events)
                df_events = pd.DataFrame.from_records(output_events).T
                df_events = df_events.rename(columns={0: 'Id', 1: "Timestamp"}).astype({"Id": np.float64,"Timestamp": np.float64})
                df_events = df_events.set_index('Id').join(df_network.set_index('Id')).reset_index()
                df_events["Timestamp"] = df_events["Timestamp"] - output_events[1][0]
                df_events["Input"] = int(input_value)
                df_events["Repeat"] = int(repeat)
                min_itemsize = dict(Id= 1000,Timestamp =1000,Type=1000, Input = 1000,Repeat = 1000)
                df_events.to_hdf(f'{data_path}/NetworkActivity_{tname}.h5',append=True,min_itemsize=min_itemsize,key=f'{tname}')
                plot_raster(output_events,imgdir,f'({input_type}_{input_value})_{repeat}',sim_duration)
                #---------get spike time and firing rate for both population
                Emask =np.isin(output_events[0],neuronPyr,assume_unique=False)
                E_spike_time= np.array(output_events[1])[Emask]
                Imask =np.isin(output_events[0],neuronPV,assume_unique=False)
                I_spike_time= np.array(output_events[1])[Imask]
                Ispike_count = len(I_spike_time)
                Espike_count = len(E_spike_time)
                AvgE = Espike_count/(netConfig.population['Pyr'] * sim_duration)
                AvgI = Ispike_count/(netConfig.population['PV'] * sim_duration)
                print(f'{DYN2util.bcolors.OKGREEN} {input_type=}, {input_value=} | {AvgE=},{AvgI=}{DYN2util.bcolors.ENDC}')
                E_df =pd.DataFrame( dict (Repeat = repeat,Input = input_value,FiringRate = AvgE, Type = 'E', InputType = input_type), index=[0])
                I_df = pd.DataFrame(dict (Repeat = repeat,Input = input_value,FiringRate = AvgI, Type = 'I', InputType = input_type), index=[0])
                firing_rate = pd.concat([firing_rate,E_df,I_df], ignore_index=True)
                if AvgE > 200 or AvgI > 200:
                    print('Network reached saturation')
            else:
                print(f'{DYN2util.bcolors.OKBLUE} {input_type=}, {input_value=} | No activity............{DYN2util.bcolors.ENDC}')
                events = dict(Id= 2* [np.nan],Timestamp = 2* [np.nan],Type=['Pyr','PV'], Input = 2*[int(input_value)],Repeat = 2*[int(repeat)])
                
                df_events = pd.DataFrame.from_dict(events)
                min_itemsize = dict(Id= 1000,Timestamp =1000,Type=1000, Input = 1000,Repeat = 1000)
                df_events.to_hdf(f'{data_path}/NetworkActivity_{tname}.h5',append=True,min_itemsize=min_itemsize,key=f'{tname}')
                E_df = pd.DataFrame(dict(Repeat = repeat,Input = input_value,FiringRate = 0.0, Type = 'E', InputType = input_type), index=[0])
                I_df = pd.DataFrame(dict(Repeat = repeat,Input = input_value,FiringRate = 0.0, Type = 'I', InputType = input_type), index=[0])
                firing_rate = pd.concat([firing_rate,E_df,I_df], ignore_index=True)
        helper.reset_TAU(model,myConfig,netConfig)   
        firing_rate.to_hdf(f'{data_path}/DC_Frequency.h5',append=True,key=f'{tname}')
    end_time = time.time()
    print(f'{DYN2util.bcolors.WARNING}  {end_time - start_time}{DYN2util.bcolors.ENDC}')