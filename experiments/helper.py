
import sys
import os
import pandas as pd
import numpy as np
import itertools
import re
import time
sys.path.append(os.getcwd() + '/..')
# from lib.dynapse2_network import Network
from lib.dynapse2_obj import *
import lib.dynapse2_util as DYN2util
import lib.dynapse2_spikegen as DYN2spikegen
from samna.dynapse2 import *

def check_timer_overflow(board, ts, input_events):
    if len(input_events) > 0 and ts + input_events[-1][3]>= 2**32 - 60000000:                
        print(f'{DYN2util.bcolors.WARNING}------Timer overflow!!! {ts + input_events[-1][3]}--------------{DYN2util.bcolors.ENDC}')
        time.sleep(2*60)
        ts = DYN2spikegen.get_fpga_time(board=board) # get current time of FPGA
        while ts > 2**31:
            ts = DYN2spikegen.get_fpga_time(board=board) # get current time of FPGA
            time.sleep(20)
        ts = ts + (int)(.2e6)
        print('Modified offset: ',ts)
    return ts

def config_parameters(myConfig,coreIds,chipIds,core_specs,n_chips= 1):
    # set neuron parameters
    #TODO: move somewhere else
    for chip_id in range(n_chips):
        DYN2util.set_parameter(myConfig.chips[chip_id].shared_parameters01, "PG_BUF_N", 0, 100)
        DYN2util.set_parameter(myConfig.chips[chip_id].shared_parameters23, "PG_BUF_N", 0, 100)
    
    for core_spec,core_id,chip_id in zip(core_specs,coreIds,chipIds):
        DYN2util.set_multi_parameters(myConfig.chips[chip_id].cores[core_id].parameters,core_spec)

def enable_monitoring(myConfig,enable_monitoring):
    # set monitoring of one or more neurons
    for chip_id,core_id,neuron_id in enable_monitoring:
        myConfig.chips[chip_id].cores[core_id].monitored_neuron = neuron_id
        myConfig.chips[chip_id].cores[core_id].neuron_monitoring_on = True
    # myConfig.chips[0].cores[2].enable_pulse_extender_monitor2 = True
    # myConfig.chips[0].cores[2].enable_pulse_extender_monitor2 = True

def set_connection_by_type(netConfig,network, groups, target, source, target_config,target_count, source_count):
    cluster_neurons = None
    if target_config['conn_type']=='prob':
        network.add_connection(source=groups[source], target=groups[target], probability=target_config['p'], dendrite=target_config['synapse'], weight=target_config['weight'], repeat=1,precise_delay=True)
    else:
        conn_matrix = None
        if target_config['conn_type']=='fan_out':
            outdegree = round(target_config['p'] * target_count)
            conn_matrix= get_connection_matrix_fan_out(source_count, target_count,out_degree=outdegree,W=[target_config['rW']])
        elif target_config['conn_type']=='fan_in':
            indegree = round(target_config['p'] * source_count)
            conn_matrix= get_connection_matrix_fan_in(source_count, target_count,in_degree=indegree,W=target_config['rW'])
        elif 'cluster' in target_config['conn_type']: #=='rec_cluster':
            print(f'{DYN2util.bcolors.FAIL} Creating memory with 0 weights, enable it by setting weight bias {DYN2util.bcolors.ENDC}')
            # print(f'{DYN2util.bcolors.WARNING} memory topology is sequential {DYN2util.bcolors.ENDC}')
            random_conn_dict = dict(in_degree = round(target_config['p'] * source_count),W = target_config['rW'])
            memory_conn_dict = dict(memory_conn_p = netConfig.memory_conn_p,W = target_config['cW'],
                                     memory_count=netConfig.memory_count,memory_size=netConfig.memory_size,
                                     memory_topology=netConfig.memory_topology, conn_type = target_config['conn_type'])
            conn_matrix,cluster_neurons= get_connection_matrix_recurrent(source_count,\
                                                        target_count,
                                                        random_conn_dict = random_conn_dict,\
                                                        memory_conn_dict = memory_conn_dict)
            print(f'{DYN2util.bcolors.WARNING}Add input stmulus channel to cluster neurons: {cluster_neurons} {DYN2util.bcolors.ENDC}')
        if conn_matrix:
            network.add_connection(source=groups[source], target=groups[target],\
                            dendrite=target_config['synapse'], weight=None,\
                            repeat=1,matrix =conn_matrix,precise_delay=True)
    return cluster_neurons

def get_connection_matrix_fan_out(source_count, target_count, out_degree, W=[0],target_subsets = [[]]):
    """
    Returns a weight matrix in a dynapse-compatible format.
    The weight is constant and each presynaptic neuron has the same out degree.
    
    source_count: (int) N. of neurons in the presynaptic population
    target_count: (int) N. of neurons in the postsynaptic population
    out_degree: (int) N. of synapses per source neuron
    W: (0-3) Value of the weight, can be a list of 4 values
    target_subsets: (list of lists) Each list contains the indices of the neurons in the cluster population
    Returns:
        connectivity matrix: list of length target_count. For each target, contains a
        list of (source, weight) tuples.
    """
    wval = [False] * 4
    for w in W: 
        wval[w] = True
    conn = [[] for _ in range(target_count)]
    for sindex,target_subset in itertools.zip_longest(range(source_count),target_subsets):
        if len(target_subset)>0:
            population_targets = np.random.choice(target_subset, out_degree, replace=False)
        else:
            population_targets = np.random.choice(target_count, out_degree, replace=False)
        for tg in population_targets:
            conn[tg].append((sindex, wval))
    return conn

def get_connection_matrix_fan_in(source_count, target_count, in_degree, W):
    wval = [False] * 4
    wval[W] = True
    conn = [
        [(sindex, wval) for sindex in np.random.choice(source_count, in_degree, replace=False)]
        for _ in range(target_count)
    ]
    return conn

def get_connection_matrix_recurrent(source_count, target_count,random_conn_dict,memory_conn_dict,conn_type = 'rec_cluster'):
    '''
    ONLY Fan-in connectivity,
    for no lateral connectivity, use empty random_conn_dict
    '''
    clusters_selection = np.array_split(range(target_count), memory_conn_dict['memory_count'])#,dtype=int)
    clusters = []
    
    if memory_conn_dict['memory_topology'] == 's':
        clusters +=[cluster[:memory_conn_dict['memory_size']] for cluster in clusters_selection]
    else:
        clusters +=[np.random.choice(cluster,memory_conn_dict['memory_size'],replace=False) for cluster in clusters_selection]
    cluster_conn = [[] for _ in range(target_count)]
    conn = [[] for _ in range(target_count)]
    c_indegree = int(memory_conn_dict['memory_conn_p'] * memory_conn_dict['memory_size'])
    if random_conn_dict:
        conn = get_connection_matrix_fan_in(source_count, target_count, random_conn_dict['in_degree'], random_conn_dict['W'])
    for cluster_index,cluster in enumerate(clusters): # over-write for clusters 
        for target_neuron in cluster:
            if conn_type =='rec_cluster':
                cluster_source = np.random.choice(cluster,c_indegree,replace=False)
            elif conn_type =='cluster': # one to many feed-forward, not recurrent and not tested
                cluster_source = cluster_index
            wval = [False] * 4
            wval[memory_conn_dict['W']] = True
            W1 = len(cluster_source)*([np.array(wval)])
            cluster_conn[target_neuron] = list(zip(cluster_source,W1))
            df_t = pd.DataFrame(conn[target_neuron]+ cluster_conn[target_neuron],columns=['source','W']).groupby('source')['W'].sum().reset_index()
            conn[target_neuron] = df_t.to_numpy()
    return conn,clusters

def stimulus(netConfig,input_type,ts,network,rate,isi_spike):
    '''
    Depreciated, use gen_input_stim
    Workes for single input stimulus
    '''
    #-------------- SET poisson gen---------------------------
    input_events = []
    if input_type =="poisson_gen":
        #TODO: for multiple input stimulus, re-write the code
        # print(rates)
        input_events += [event for event in DYN2spikegen.poisson_gen(start=ts,duration=netConfig.stimulus_duration['StimulusON']*1e6,virtual_groups=network.virtual_groups,rates=[rate])]
    #------------------------SET ISI gen Input----------------------
    # sending spikes at precise timing for all virtual neurons
    elif input_type =="spike_gen":
        # timestamps = np.sort(np.random.rand(35)) * .1e6 + ts #for random time events
        timestamps = np.arange(ts,ts+netConfig.stimulus_duration['StimulusON']*1e6,isi_spike)
        timestamp_array = np.repeat(timestamps,len(network.virtual_groups[0].ids))
        inp_nodes = np.tile(network.virtual_groups[0].ids,len(timestamps))
        input_events = DYN2spikegen.isi_gen(virtual_group=network.virtual_groups[0], neurons= inp_nodes, timestamps = timestamp_array)
    return input_events

def gen_input_stim(network,input_stimuli):
    CH_Type = np.array(input_stimuli['CH_Type'])
    start_times = np.array(input_stimuli['start_time'])
    durations = np.array(input_stimuli['StimulusON'])
    status = np.array(input_stimuli['Status'])
    rate_isi= np.array(input_stimuli['rate_isi'])
    low_rate_isi= np.array(input_stimuli['low_rate_isi'])
    pg_index = np.where(CH_Type=='poisson_gen')[0]
    sg_index = np.where(CH_Type=='spike_gen')[0]
    df_events,pg_df_events,sg_df_events = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    input_events = []
    if len(pg_index) > 0:
        poisson_input_stimuli = list(zip(np.array(network.virtual_groups)[pg_index],start_times[pg_index],durations[pg_index],rate_isi[pg_index],low_rate_isi[pg_index],status[pg_index]))
        pg_df_events = DYN2spikegen.poisson_gens(poisson_input_stimuli,min(start_times),max(start_times),durations[-1])
    #NOTE: virtual_groups only works for single input channel per stimulus
    if len(sg_index) > 0:
        spikegen_input_stimuli = list(zip(np.array(network.virtual_groups)[sg_index],start_times[sg_index],durations[sg_index],rate_isi[sg_index],status[sg_index]))
        sg_df_events =DYN2spikegen.isi_gens(spikegen_input_stimuli) 
        
    if not pg_df_events.empty and not sg_df_events.empty:
        df_events = pd.concat([pg_df_events,sg_df_events],ignore_index=True)
        df_events = df_events.sort_values(['timestamp'])
        input_events = list(df_events.to_records(index=False))
    elif not pg_df_events.empty:
        input_events = list(pg_df_events.to_records(index=False))
    elif not sg_df_events.empty:
        input_events = list(sg_df_events.to_records(index=False))
    return input_events

def save_config(myConfig,data_path,tname,allocated_chips,allocated_cores):
    df_config = pd.DataFrame()
    for chip,core in zip(allocated_chips,allocated_cores):            
        df_core_config = pd.DataFrame(myConfig.chips[chip].cores[core].parameters,index=[0]).T.reset_index()
        df_core_config = df_core_config.rename(columns={'index': 'paramName', 0: "param_detail"},inplace=False).astype({"param_detail": str})
        df_core_config['Chip'] = chip
        df_core_config['Core'] = core
        df_core_config[['Coarse','Fine']] = df_core_config['param_detail'].str.split(re.compile("\(([^\)]+)\)"),regex=True,expand=True).rename(columns={0: 'neglect', 1: "val1",2: 'val2'},inplace=False)['val1'].str.split(',',expand=True).rename(columns={0: 'coarse', 1: "fine",2: 'neglect'},inplace=False)[['coarse','fine']]
        df_core_config.drop(['param_detail'],axis=1,inplace=True)
        df_config = pd.concat((df_config,df_core_config),ignore_index=True)
  
    df_config.to_hdf(f'{data_path}/board_Config.h5',key=f'{tname}')

def reset_TAU(model, myConfig,netConfig):
    
    for ntype in netConfig.neuronTypes:
        biases_name = ['SOIF_LEAK_N','DEAM_ETAU_P','DEGA_ITAU_P','DENM_ETAU_P','DESC_ITAU_P']
        biases_value = []
        for bias_name in biases_name:
            #----------------------------------------------Save bias settings-----------------------------------
            bias_val= DYN2util.get_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters,bias_name)
            biases_value.append(bias_val)
            #---------------Set to highest so everything leaks to direct current------------------------------------
            DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters, bias_name, 5, 255)
        model.apply_configuration(myConfig)
        time.sleep(.1)

        for bias_name,bias_val in zip(biases_name,biases_value):
            DYN2util.set_parameter(myConfig.chips[netConfig.allocated_chips[ntype]].cores[netConfig.allocated_cores[ntype]].parameters, bias_name, bias_val[0], bias_val[1])
        model.apply_configuration(myConfig)
        time.sleep(.1)
