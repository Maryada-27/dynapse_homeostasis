import time
from numpy import random
from samna.dynapse2 import *
from lib.dynapse2_obj import *
import numpy as np
# import units
import pandas as pd


def get_fpga_time(board):
    while True:
        board.input_interface_write_events(0,
                                           [AerConstructor(
                                               DestinationConstructor(tag=1024,
                                                                      core=[True]*4, x_hop=-1, y_hop=-1).destination,
                                               0).aer])
        for timeout in range(1000):
            evs = board.read_events()
            if len(evs) > 0:
                return evs[-1].timestamp


def send_events(board, events, min_delay=0):
    if len(events) > 0:
        ts = events[-1].timestamp
    else:
        ts = get_fpga_time(board=board)
    board.input_interface_write_events(0, events + [AerConstructor(DestinationConstructor(tag=2047,core=[True]*4, x_hop=-1, y_hop=-1).destination,ts + min_delay).aer]*2)

    # can also send through the grid bus directly
    # board.grid_bus_write_events(events + [AerConstructor(DestinationConstructor(tag=2047,
    #                                                                    core=[True]*4, x_hop=-1, y_hop=-1).destination,
    #                                             ts + min_delay).aer]*32)

    # can also send through the grid bus directly
    # board.grid_bus_write_events(events + [AerConstructor(DestinationConstructor(tag=2047, core=[True]*4, x_hop=-1, y_hop=-1).destination,ts + min_delay).aer]*32)


def send_virtual_events(board, virtual_events, offset=0, min_delay=0):
    # 1 << core is same as 1 * (2**core)
    input_events = []
    for event in virtual_events:
        
        input_events += [AerConstructor(
            destination=DestinationConstructor(tag=event[0], core=[True]*4, x_hop=-7).destination,
            timestamp=event[3] + offset).aer] + \
                        [AerConstructor(
                            destination=DestinationConstructor(
                                tag=event[1], core=[event[2][1] & 1 << core > 0 for core in range(4)],
                                x_hop=event[2][0]).destination,
                            timestamp=(int)(event[3] + offset)).aer]
    
    if len(virtual_events) > 0:
        ts = input_events[-1].timestamp
    else:
        ts = get_fpga_time(board=board)
    _dummyTag = 2047

    input_events += [AerConstructor(
        destination=DestinationConstructor(tag=_dummyTag, core=[True]*4, x_hop=-1, y_hop=-1).destination,timestamp=ts + min_delay).aer] * 2 # sent twice to avoid possible infinte loop, if the event is not readout properly.
    
    board.input_interface_write_events(0, input_events)


def poisson_gen(start, duration, virtual_groups, rates):
    # random.seed(3400)
    events = []
    rates_weighted = [sum(rate) for rate in rates]
    rates_weighted_sum = sum(rates_weighted)
    
    if rates_weighted_sum:
        scale = 1 / rates_weighted_sum
        p = [rate * scale for rate in rates_weighted]
        scale *= 1e6
        t = start
        while t < start + duration:
            t += random.exponential(scale=scale)
            group_id = random.choice(range(len(virtual_groups)), p=p)
            ids = virtual_groups[group_id].get_ids()
            # print(group_id,ids)
            i = random.choice(range(virtual_groups[group_id].size),p=rates[group_id]/sum(rates[group_id]))            
            for chip_core, tags in virtual_groups[group_id].get_destinations().items():
                events += [(ids[i], tags[i], chip_core, int(t))]
    return events

def poisson_gens(input_stimuli,sim_start=None,sim_end=None,dur=None):
    '''
    strictly 1 channel
    input_stimuli = [(virtual_group,start,duration,rate,status),(virtual_group,start,duration,rate,status)]
    '''
    df_events = pd.DataFrame()
    for stimulus in input_stimuli:
        if stimulus[-1]: #check status of stimulus, to set or reset edit network_config file
            rate = stimulus[3]     # spike rate
            low_rate = stimulus[4]     # spike rate
            start = stimulus[1]*1e6    # the total lenght of the spike train
            duration = stimulus[2]*1e6
            scale = 1 / (rate-low_rate) # to compansate for the low_rate
            # p = rate * scale
            scale *= 1e6
            t = start
            ids = stimulus[0].get_ids()
            i = 0
            while t < start + duration:
                t += random.exponential(scale=scale)            
                for chip_core, tags in stimulus[0].get_destinations().items():                
                    dict_event = dict(Id = ids[i],tag = tags[i],chip_core=[chip_core],timestamp = int(t))
                    df_events = pd.concat([df_events,pd.DataFrame.from_dict(dict_event)],ignore_index=True)
            low_t = sim_start*1e6
            low_scale = 1 / low_rate
            # p = rate * scale
            low_scale *= 1e6
            print(low_rate,sim_end*1e6, dur*1e6)
            while low_t < sim_end*1e6 + dur*1e6: 
                low_t += random.exponential(scale=low_scale)            
                for chip_core, tags in stimulus[0].get_destinations().items():                
                    dict_event = dict(Id = ids[i],tag = tags[i],chip_core=[chip_core],timestamp = int(low_t))
                    df_events = pd.concat([df_events,pd.DataFrame.from_dict(dict_event)],ignore_index=True)
    if not df_events.empty:
        df_events = df_events.sort_values(['timestamp'])
        # events = list(df_events.to_records(index=False))
    return df_events

def isi_gen(virtual_group, neurons, timestamps):
    '''
    return event: [(ids[n], tags[n], chip_core, int(t))]
    ids : FPGA loop back for plotting
    tags: internal rounting tag 
    chip_core: (chip, core) tuple, note: core is assigned as 1,2,4,8
    t: timestamp in micosecond
    '''
    events = []
    ids = virtual_group.get_ids()
    for n, t in zip(neurons, timestamps):
        for chip_core, tags in virtual_group.get_destinations().items():
            events += [(ids[n], tags[n], chip_core, int(t))]
    return events

def isi_gens(input_stimuli):
    '''
    parameters: 
    input_stimuli = [(virtual_group,start,duration,isi_spike),(virtual_group,start,duration,isi_spike)]
    where start, duration, isi_spike in seconds 

    return: events [(ids[n], tags[n], chip_core, int(t))]
    '''
    df_events = pd.DataFrame()
    # events =[]
    for stimulus in input_stimuli:
        if stimulus[-1]: #check status of stimulus, to set or reset edit network_config file
            timestamps = np.arange(stimulus[1] * 1e6,(stimulus[1] + stimulus[2]) * 1e6,stimulus[3] * 1e6)
            ids = stimulus[0].get_ids()
            n = 0
            for t in timestamps:
                for chip_core, tags in stimulus[0].get_destinations().items():
                    dict_event = dict(Id = ids[n],tag = tags[n],chip_core=[chip_core],timestamp = int(t))
                    df_events = pd.concat([df_events,pd.DataFrame.from_dict(dict_event)],ignore_index=True)
    if not df_events.empty:
        df_events = df_events.sort_values(['timestamp'])
        # events = list(df_events.to_records(index=False))
    return df_events