# stack board run with: python3 main.py ./bitfiles/Dynapse2Stack.bit 1 orange (or green)
# dev board run with: python3 main.py -d ./bitfiles/Dynapse2DacTestboard.bit

# if self-built samna (no need if installed through pip)
# export PYTHONPATH=$PYTHONPATH:~/Documents/git/samna/build/src

import optparse
import random

import pandas as pd
import samna
import lib.dynapse2_init as DYN2init
from datetime import datetime
from experiments.EI import Input_Freq,EI_homeostasis,sWTA_post_training,NeuronSanityCheck #, WMM
import os
import numpy as np
import itertools
from network_config import netConfig
import time

def create_result_dir(parent_dir,tname,board_names):
    date_label = datetime.today().strftime('%Y-%m-%d')
    time_label = datetime.today().strftime('%H-%M')
    dir_path = f"{parent_dir}/board_{'-'.join(board_names)}/{date_label}/{tname}/{time_label}"
    data_path = f'{dir_path}/data'
    imgdir = f"{dir_path}/img/{tname}"
    os.makedirs(f"{data_path}",exist_ok=True)
    os.makedirs(f"{imgdir}",exist_ok=True)
    return dir_path,data_path,imgdir

def main():
    parser = optparse.OptionParser()
    parser.set_usage("Usage: test_sadc.py [options] bitfile [number_of_chips]")
    parser.add_option("-d", "--devboard", action="store_const", const="devboard", dest="device",
                      help="use first XEM7360 found together with DYNAP-SE2 DevBoard")
    parser.add_option("-s", "--stack", action="store_const", const="stack", dest="device",
                      help="use first XEM7310 found together with DYNAP-SE2 Stack board(s)")
    parser.set_defaults(device="stack")
    opts, args = parser.parse_args()
    print(args)
    if len(args) > 2:
        number_of_chips = int(args[1])
    else:
        number_of_chips = 1
    board = None
    board_names = [args[2]] #["green"]#["orange"]#
    greenSN = '1911000P6X'#'1911000P6X'#'1739000JB7'
    orangeSN = '2042000UO5'#'2042000UP2'
    purpleSN = '1911000P8A'
    if board_names[0] == "green":
        SN = greenSN
    elif board_names[0] == "orange":
        SN = orangeSN
    elif board_names[0] == "purple":
        SN = purpleSN
    deviceInfos = samna.device.get_unopened_devices()   
    print(f'{deviceInfos=}')
    for device in deviceInfos:
        print(f'{device.serial_number=}')
        if device.serial_number == SN:
            board = samna.device.open_device(device)
    #samna-0.14.25.0:
    board.reset_fpga()
    board = DYN2init.dynapse2board(board=board, args=args)
    init_w = False # enable this to use initial weights from file instead of random weights
    # parent_dir = '/media/mb/Data/DYNAP/EI_homeostasis/results/onchip_experimentaldata'
    parent_dir = '/media/mb/Dev/workspace/dynapse/onchip_experimentaldata'
    tname = "EI_homeostasis_highall"#EI_homeostasis | TEST_adapt | IF_curve | EI_homeostasis_OLH

    if init_w:
        initial_weights_fname = "/media/mb/Data/DYNAP/initial_weights_scatter.h5"
        df_w = pd.read_hdf(initial_weights_fname)
        coarse_fine_list = dict(
                    Pyr_Pyr = df_w[df_w.type =='PyrPyr'][['coarse','fine']].values, #E-to-E
                    PV_PV = df_w[df_w.type =='PVPV'][['coarse','fine']].values, #I-to-I
                    Pyr_PV = df_w[df_w.type =='PVPyr'][['coarse','fine']].values,  #I-to-E
                    PV_Pyr = df_w[df_w.type =='PyrPV'][['coarse','fine']].values, #E-to-I
            )
        combinations_count = coarse_fine_list['Pyr_Pyr'].shape[0]
        
    else:
        coarse_fine_grid = dict(
            Pyr_Pyr = list(itertools.product([2,4],np.arange(20,200,10))), #E-to-E
            PV_PV = list(itertools.product([2,4],np.arange(20,200,10))), #I-to-I
            Pyr_PV = list(itertools.product([2,4],np.arange(20,200,10))), #I-to-E
            PV_Pyr = list(itertools.product([2,4],np.arange(20,200,10))), #E-to-I
        )
        param_grid= list(itertools.product(*coarse_fine_grid.values()))
        random.shuffle(param_grid)
        combinations_count = len(param_grid)

    print(f"Total number of combinations: {combinations_count}")
    
    # uncomment for running the experiment for IF curve
    
    # NeuronSanityCheck.experiment_run(board=board,dir_path=dir_path,board_names=board_names, profile_path=os.getcwd() + "/CAMs/profiles/", number_of_chips=number_of_chips,tname=tname) #tname = "AMPA_timeconstant"#"Neuron_timeconstant"#"Refactory_period"#
    # dir_path,data_path,imgdir =create_result_dir(parent_dir, tname,board_names) 
    # Input_Freq.experiment_run(board=board,dir_path=dir_path,board_names=board_names, profile_path=os.getcwd() + "/CAMs/profiles/", number_of_chips=number_of_chips,tname=tname)
    
    repeat_count = 1
    for _ in range(repeat_count):
        values = param_grid[random.randint(0,combinations_count-1)]
        weight_config = []
        param_dict = dict(zip(coarse_fine_grid.keys(), values))

        # random_index = random.randint(0,combinations_count-1) # for initial weights from file
        
        for key,weight in param_dict.items():
        # for key,weights in coarse_fine_list.items(): # for initial weights from file
            target,source = key.split('_')
            coarse_weight,fine_weight = weight[0],weight[1] # for random init
            # coarse_weight,fine_weight = weights[random_index][0],weights[random_index][1] # for initial weights from file
            weight_config.append(dict(source = source, 
                                target = target,
                                fine_weight = fine_weight,
                                coarse_weight = coarse_weight))
        print(weight_config)
        dir_path,data_path,imgdir =create_result_dir(parent_dir, tname,board_names)        
        sWTA_post_training.experiment_run(board=board,dir_path=dir_path,board_names=board_names, 
                                    profile_path=os.getcwd() + "/CAMs/profiles/",
                                    number_of_chips=number_of_chips,tname=tname,weight_config=weight_config)

if __name__ == '__main__':
    main()