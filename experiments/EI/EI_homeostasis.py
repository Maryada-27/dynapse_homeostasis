import itertools
import os
# import re
import sys
import time
import pandas as pd
import pickle
# pylint: disable=import-error,wildcard-import,wrong-import-position

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/experiments/EI')
sys.path.append(os.getcwd() + '/experiments/')

from .core_bias import CoreBias 
from network_config import netConfig
import helper
from lib.dynapse2_network import Network
from lib.dynapse2_obj import *
from lib.dynapse2_raster import *
import lib.dynapse2_spikegen as DYN2spikegen
from learning import Homeostasis
import samna.dynapse2 as dyn2
import lib.dynapse2_util as DYN2util

from tqdm import tqdm
            
def config_connections(network,input_channels=1, virtual_group_size = 1,input_types = ['pg'] ):
    #TODO: group vs channel
    #TODO: Variable group_size
    # set name of virtual group as poisson_gen or spike_gen
    
    input_groups = [network.add_virtual_group(size=virtual_group_size,name=name) for _,name in zip(range(input_channels),input_types)]
    groups = dict()
    neuron_clusters =[] # this make sense as an array if there are multiple clusters and each cluster has multiple memories
    for cluster_index in range(netConfig.cluster_count):  
        t_dict = dict()      
        for ntype in netConfig.neuronTypes: # neuronTypes is array that contains the neuron types that user intend to create
            group = network.add_group(chip=netConfig.allocated_chips[ntype],
                                    core=netConfig.allocated_cores[ntype],
                                    size=netConfig.population[ntype],
                                    name = f'Cluster_{cluster_index}_{ntype}')
            t_dict.update({ntype : group})
        groups[f'Cluster_{cluster_index}'] = t_dict
        for source, target_configs in netConfig.connectivity.items():
            for target,target_config in target_configs.items():
                target_count, source_count = netConfig.population[target], netConfig.population[source]
                neuron_cluster = helper.set_connection_by_type(netConfig,network, groups[f'Cluster_{cluster_index}'], target, source, target_config,target_count, source_count)
                if neuron_cluster:
                    neuron_clusters.append(neuron_cluster)
        #---------------------------------SET Input CONNECTIONS HERE-----------------------------
        input_group,pInput,target = input_groups[cluster_index],.8,'Pyr'
        target_count = netConfig.population[target]
        outdegree = round(pInput* target_count)
        
        conn_matrix= helper.get_connection_matrix_fan_out(input_group.size, target_count,out_degree=outdegree, W=[0])
        network.add_connection(source=input_group, target=groups[f'Cluster_{cluster_index}'][target],
                                dendrite=dyn2.Dendrite.ampa,weight=None,repeat=1,
                                matrix = conn_matrix,
                                mismatched_delay=True)
    print(f'{neuron_clusters=}, {netConfig.IMPLANT_MEMORY}')
    
    if netConfig.PARADOXICAL:
        for input_group,pInput,target in  zip([input_groups[1]],[1.],['PV']):
            cluster_index = 0
            network.add_connection(source=input_group, target=groups[f'Cluster_{cluster_index}'][target], probability=pInput,dendrite=dyn2.Dendrite.ampa, weight=[True,False,False,False], repeat=1,precise_delay=True)
    
    elif netConfig.IMPLANT_MEMORY:
        cluster_index = 0 # FIX: for now only one cluster
        input_group,pInput,target = input_groups[1:],0.5,f'Pyr'
        print(f'{DYN2util.bcolors.OKBLUE} Adding input stmulus channel to cluster 0, NeuronIds: {neuron_clusters} {DYN2util.bcolors.ENDC}')
        # the size of the conn vector should still be same as whole target "group" and not cluster
        target_population_count = netConfig.population[target]
        #IMPORTANT: fix it
        outdegree = round(pInput* netConfig.memory_size)
        for mem_index,inp_group in enumerate(input_group):
            stim_conn_matrix= helper.get_connection_matrix_fan_out(inp_group.size, target_population_count,out_degree=outdegree,W=[0,3],target_subsets=[neuron_clusters[cluster_index][mem_index]])
            network.add_connection(source=inp_group, target=groups[f'Cluster_{cluster_index}'][target],dendrite=dyn2.Dendrite.ampa, weight=None, repeat=1,matrix = stim_conn_matrix,mismatched_delay=True)
    print('Connecting.... ')
    network.connect()
    print('Connectivity done')

def init_network(myConfig,model,profile_path,number_of_chips,data_path,boards,tname='EI_homeostasis'):
    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips,boards=boards)
    config_connections(network=network, input_channels=netConfig.population['CH'],virtual_group_size=netConfig.population['CH_neurons'], input_types =netConfig.input_stimuli['CH_Type'])
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
    # IMPORTANT: Save connectivity as well
    return network,df_network

def experiment_run(board,dir_path,board_names=["orange"], number_of_chips=1, profile_path="../profiles/",tname='EI_homeostasis',weight_config=dict()):
    data_path = f'{dir_path}/data'
    imgdir = f"{dir_path}/img/{tname}"
     
    run_config = netConfig()    
    with open(f'{data_path}/params.pickle', 'wb') as fobj:
        pickle.dump(run_config, fobj)
    
    if weight_config:
        with open(f'{data_path}/weight_init.pickle', 'wb') as fobj:
                pickle.dump(weight_config, fobj)
    
    stablization_window = 0.002
    #---------------------------reset_boardConfig------------------------------------------------
    model = board.get_model()
    model.reset(dyn2.ResetType.PowerCycle, 0b1)
    time.sleep(0.1)
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(.1)
    # --------------------Configure the parameters for all the cores--------------------
    print("Configuring parameters")
    core_specs = [CoreBias.get_module_params(neuron_type=nType) for nType in netConfig.neuronTypes]
    chipIds = [netConfig.allocated_chips[nType] for nType in netConfig.neuronTypes]
    coreIds = [netConfig.allocated_cores[nType] for nType in netConfig.neuronTypes]
    helper.config_parameters(myConfig=myConfig,chipIds=chipIds,coreIds=coreIds,core_specs=core_specs)
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    #-------------------------------------------------------------------------------------------
    if netConfig.ENABLE_DC:
        DYN2util.set_dc_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=range(netConfig.N_CORES), chips=range(number_of_chips))
        model.apply_configuration(myConfig)
        time.sleep(.1)
    if netConfig.ENABLE_ADAPT: # ONLY FOR SST neurons
        DYN2util.set_adap_latches(config=myConfig, neurons=range(netConfig.N_PER_CORE), cores=netConfig.allocated_cores['SST'], chips=range(number_of_chips))
        model.apply_configuration(myConfig)
        time.sleep(.1)
    
    #--------------------------------------init_network---------------------------------
    network,df_network = init_network(myConfig,model,profile_path,number_of_chips,data_path,board_names,tname)
    # set monitoring of one or more neurons
    for gid,ntype in enumerate(netConfig.neuronTypes):
        nid = np.random.randint(0,netConfig.population[ntype])
        print(f'Monitoring {nid}')
        myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].monitored_neuron = network.groups[gid].neurons[nid]
        myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].neuron_monitoring_on = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    #--------------------------------------------------------------------------------------------
    # for nn in range(10):
    #     ntags = [_.tag for _ in myConfig.chips[netConfig.allocated_chips['Pyr']].cores[netConfig.allocated_cores['Pyr']].neurons[nn].synapses]
    print("\nAll configurations done!\n")
    allocated_cores = list(netConfig.allocated_cores.values())
    allocated_chips = list(netConfig.allocated_chips.values())
    helper.save_config(myConfig,data_path,tname,allocated_chips,allocated_cores)
    #------------------learning setup------------------------------
    
    homeostasis= Homeostasis(netConfig.LEARNING_RATE,netConfig.E_TARGET,netConfig.I_TARGET)
    input_stimuli = netConfig.input_stimuli
    start_time = time.time()
    sim_duration = netConfig.input_stimuli['StimulusON'][0] + netConfig.StimulusOFF
    # test_count = np.array([netConfig.PARADOXICAL, netConfig.IMPLANT_MEMORY]).sum()

    #------------------initialize weights------------------------------ 
    for wconfig in weight_config:
        source,target = wconfig['source'],wconfig['target']
        w_name = netConfig.connectivity[source][target]['w_name']
        chip_id,core_id = netConfig.allocated_chips[target],netConfig.allocated_cores[target]
        DYN2util.set_parameter(myConfig.chips[chip_id].cores[core_id].parameters,w_name,wconfig['coarse_weight'],wconfig['fine_weight'])
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    
    for iteration in tqdm(range(netConfig.TRAIN_ITERATIONS + netConfig.TEST_ITERATIONS ),desc="Iteration Progress"):
        AvgE_iteration = []
        AvgI_iteration = []
        df_weight = pd.DataFrame()
        df_Firingrate = pd.DataFrame({'Iteration': pd.Series(dtype='int'),
                                      'Repeat_count' : pd.Series(dtype='int'),
                                      'AvgE': pd.Series(dtype='float'),
                                      'AvgI': pd.Series(dtype='float')})
        #BUG:  Fix it for multicluster case

        for repeat in tqdm(range(netConfig.REPEAT_COUNT),desc="Repetition Progress"):
            if netConfig.cluster_count == 1:
                cluster_index = 0
            else:
                cluster_index = repeat
            neuronPyr = df_network[df_network.Type == f'Cluster_{cluster_index}_Pyr'].Id.values #Cluster_{cluster_index}_
            neuronPV = df_network[df_network.Type == f'Cluster_{cluster_index}_PV'].Id.values #Cluster_{cluster_index}_
            time.sleep(0.5)
            #--------------- CLEAR BUFFER------------------------------------
            output_events = [[], []]
            DYN2spikegen.send_virtual_events(board=board, virtual_events=[])
            get_events(board=board, extra_time=100, output_events=output_events)
            #--------------- CLEARED BUFFER------------------------------------
            ts = DYN2spikegen.get_fpga_time(board=board) + (int)(.2e6) # get current time of FPGA
            input_events = []
            if netConfig.ENABLE_DC:
                #-----------------------enable monitoring and DC------------------------------------
                DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips['Pyr']].cores[netConfig.allocated_cores['Pyr']].parameters,"SOIF_DC_P",netConfig.DC[0],netConfig.DC[1])
                model.apply_configuration(myConfig)
                time.sleep(netConfig.DC_ON)
                DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips['Pyr']].cores[netConfig.allocated_cores['Pyr']].parameters,"SOIF_DC_P",0,0)
                model.apply_configuration(myConfig)
                time.sleep(0.1)

            min_delay = (int)(netConfig.StimulusOFF*1e6)
            # for no input, set status line netConfig as False
            if iteration >= netConfig.TRAIN_ITERATIONS:
                # print(f'{DYN2util.bcolors.OKBLUE} TESTING MEMORY RECALL {DYN2util.bcolors.ENDC}')
                input_stimuli['Status'] = [True for _ in range(len(input_stimuli['Status']))]
                # the first one is the kick
                input_stimuli['Status'][0] = False
                # min_delay = 0.5*1e6 # randomly adding a  delay 
                sim_duration = None 
            elif netConfig.cluster_count > 1:
                input_stimuli['Status'] = [False for _ in range(len(input_stimuli['Status']))]
                input_stimuli['Status'][cluster_index] = True

            input_events = helper.gen_input_stim(network,input_stimuli)
            ts = DYN2spikegen.get_fpga_time(board=board) + (int)(.2e6) # get current time of FPGA
            # #------------------- sending events --------------------------------------
            ts = helper.check_timer_overflow(board, ts, input_events)
            
            DYN2spikegen.send_virtual_events(board=board, virtual_events=input_events, offset=ts, min_delay=min_delay)
            #------------------------ READ and plot OUTPUT------------------------------
            output_events = [[], []]
            get_events(board=board, extra_time=100, output_events=output_events)# extra_time is to clear the buffer
            helper.reset_TAU(model, myConfig,netConfig) # Move on top of the if condition
            AvgI_Trialwise = 0.
            AvgE_Trailwise = 0.
            if len(output_events[0])> len(input_events):
                # Record spike activty
                df_events = pd.DataFrame.from_records(output_events).T
                df_events = df_events.rename(columns={0: 'Id', 1: "Timestamp"}).astype({"Id": int})
                df_events = df_events.set_index('Id').join(df_network.set_index('Id')).reset_index()
                df_events["Timestamp"] = df_events["Timestamp"] - output_events[1][0]
                df_events["Iteration"] = int(iteration)
                df_events["Repeat"] = int(repeat)
                min_itemsize = dict(Id= 1000,Timestamp =1000,Type=1000, Iteration = 1000,Repeat = 1000)
                df_events.to_hdf(f'{data_path}/NetworkActivity_{tname}.h5',complevel=9,append=True,min_itemsize=min_itemsize,key=f'{tname}')                
                plot_raster(output_events,f'{imgdir}/Iter_{iteration}',f'repeat_{repeat}',sim_duration,max(df_network.Id.values))
                #----------------------get spike time and firing rate for both population
                seed_timestamp = ts*1e-6 + netConfig.KICK_ON + stablization_window                
                Pyrmask =np.isin(output_events[0],neuronPyr,assume_unique=False)
                E_spike_time= np.array(output_events[1])[Pyrmask]
                Espike_OFF_stim = E_spike_time[E_spike_time>=seed_timestamp]
                PVmask =np.isin(output_events[0],neuronPV,assume_unique=False)
                I_spike_time= np.array(output_events[1])[PVmask]
                Ispike_OFF_stim = I_spike_time[I_spike_time>=seed_timestamp]             
                Ispike_count = len(Ispike_OFF_stim)        
                Espike_count = len(Espike_OFF_stim)
                spike_last_timestamp = max(max(Espike_OFF_stim,default=0),max(Ispike_OFF_stim,default=0))
                #---------------------------------Excitatory neurons firing rate---------------------
                if Espike_count > 0:
                    Espike_last_timestamp = spike_last_timestamp #max(Espike_OFF_stim)
                    firing_rate_time_window =  Espike_last_timestamp - seed_timestamp
                    firing_rate_time_window = np.around(firing_rate_time_window,4)
                    if firing_rate_time_window > 0.001: # to avoid divison by very very small number
                        AvgE_Trailwise = Espike_count/(netConfig.population['Pyr']*firing_rate_time_window)
                #-------------------------------Inhibitory neurons ----------------------------
                if Ispike_count > 0:
                    Ispike_last_timestamp = spike_last_timestamp #max(Ispike_OFF_stim)
                    firing_rate_time_window = Ispike_last_timestamp - seed_timestamp
                    firing_rate_time_window = np.around(firing_rate_time_window,4)
                    if firing_rate_time_window > 0.001: # to avoid divison by very very small number
                        AvgI_Trialwise = Ispike_count/(netConfig.population['PV']*firing_rate_time_window)
                
                AvgE_iteration.append(np.around(AvgE_Trailwise,4))
                AvgI_iteration.append(np.around(AvgI_Trialwise,4))
            else:
                print('No activity............')
                AvgE_iteration.append(0)
                AvgI_iteration.append(0)

        mu_E = np.mean(AvgE_iteration)
        mu_I = np.mean(AvgI_iteration)
        print(f'{DYN2util.bcolors.OKBLUE}Iteration: {iteration} :: {AvgE_iteration=} Hz = {mu_E} Hz, {AvgI_iteration=} Hz = {mu_I} Hz, {DYN2util.bcolors.ENDC}')
       
        AvgFr_dict = dict(Iteration = netConfig.REPEAT_COUNT * [iteration],Repeat_count = np.arange(0,netConfig.REPEAT_COUNT,1),AvgE=AvgE_iteration, AvgI=AvgI_iteration)
        df_Firingrate = pd.concat([df_Firingrate,pd.DataFrame(AvgFr_dict)],ignore_index=True)
        #-----------------------WEIGHT UPDATE Averaged over firing rates-------------------------
        # if netConfig.LEARNING_ON and iteration < netConfig.TRAIN_ITERATIONS:
        #     if abs(max(np.subtract(AvgE_iteration,netConfig.E_TARGET),key=abs)) > netConfig.CONVERGENCE_STD or abs(max(np.subtract(AvgI_iteration,netConfig.I_TARGET),key=abs)) > netConfig.CONVERGENCE_STD:
        #         homeostasis.calculate_update_weights(myConfig,mu_E,mu_I,netConfig.STATIC_INH,netConfig.STATIC_EXC)
        #         model.apply_configuration(myConfig)
        #         time.sleep(0.1)
        #     else:
        #         print(f'{DYN2util.bcolors.OKGREEN}SUCCESS: !! Network Converged, re-running to confirm!!{DYN2util.bcolors.ENDC}')
        # #-----------------------WEIGHT UPDATE for each repeat and averaged-------------------------
        if netConfig.LEARNING_ON and iteration < netConfig.TRAIN_ITERATIONS:
            if abs(max(np.subtract(AvgE_iteration,netConfig.E_TARGET),key=abs)) > netConfig.CONVERGENCE_STD or abs(max(np.subtract(AvgI_iteration,netConfig.I_TARGET),key=abs)) > netConfig.CONVERGENCE_STD:
                
                if netConfig.CROSSH:
                    homeostasis.calculate_update_weights_avg_over_repeat(myConfig,AvgE_iteration,AvgI_iteration,netConfig.STATIC_INH,netConfig.STATIC_EXC)
                else:
                    print(f'{DYN2util.bcolors.FAIL}WARNING!!!: OLH learning rule{DYN2util.bcolors.ENDC}')
                    homeostasis.calculate_update_weights_OLH(myConfig,AvgE_iteration,AvgI_iteration)
                
                model.apply_configuration(myConfig)
                time.sleep(0.1)
            else:
                print(f'{DYN2util.bcolors.OKGREEN}SUCCESS: !! Network Converged, re-running to confirm!!{DYN2util.bcolors.ENDC}')
        #-------------------------------MEMORY IMPLANTATION-------------------------------
        if netConfig.IMPLANT_MEMORY and iteration == netConfig.IMPLANT_MEMORY_at:
            print(f'{DYN2util.bcolors.OKGREEN}INFO: Memory Implantation at {iteration} !!{DYN2util.bcolors.ENDC}')
            # SYAM_W3_P set weight
            ntype = 'Pyr'
            DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters,"SYAM_W3_P",netConfig.MEMORY_WEIGHT_BIAS[0],netConfig.MEMORY_WEIGHT_BIAS[1]) #TODO: make it configurable
           
        for source, target_configs in netConfig.connectivity.items():
            for target,target_config in target_configs.items():
                w_val= DYN2util.get_parameter(myConfig.chips[netConfig.allocated_chips[target]].cores[netConfig.allocated_cores[target]].parameters,target_config['w_name'])
                weight_dict = dict(Iteration = iteration,source=source, target=target,  coarse=w_val[0],fine=w_val[1])
                df_weight = pd.concat([df_weight,pd.DataFrame(weight_dict,index=[0])],ignore_index=True)
    
        if not df_weight.empty:
            df_weight.to_hdf(f'{data_path}/weight_trend.h5',append=True,key=f'{tname}')
        if not df_Firingrate.empty:
            df_Firingrate.to_hdf(f'{data_path}/firing_rate.h5',append=True,key=f'{tname}')
        
    end_time = time.time()
    print(f'{DYN2util.bcolors.WARNING}  {(end_time-start_time)/60} {DYN2util.bcolors.ENDC}')
    