
import itertools
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import os
import re

sns.set_theme("paper", font_scale=1.2, rc={"lines.linewidth": 1.5}, palette="bright",style='ticks',color_codes = True )


def plot_events(df_network_activity,img_path="",imgfname="raster"):
    ''' raster plot '''
    colors = ['#000000',"#FF0B04", "#4374B3"]# Set your custom color palette
    sns.set_palette(sns.color_palette(colors))

    for iteration in df_network_activity.Iteration.unique():
        # f = plt.figure(figsize=(20,20),dpi=100)        
        kwrg = dict(marker = '.', s = 10)        
        g = sns.FacetGrid(df_network_activity[df_network_activity.Iteration==iteration ], col="Repeat",col_wrap=2,height=3,aspect=1.5, margin_titles=True,sharex= False,sharey= True)

        g.map_dataframe(sns.scatterplot, x="Timestamp", y="Id",hue = "Type",**kwrg)
        g.fig.subplots_adjust(top=.96) 
        g.fig.suptitle(f'ITERATION :: {iteration}')
        g.add_legend()
        g.set_xlabels('Time (sec)')
        g.set_ylabels('Neuron Id')
        # print(g.axes_dict.items())
        sns.move_legend(g, "upper right")#, bbox_to_anchor=(.02, .01))
        g.tight_layout()
        for col_val, ax in g.axes_dict.items():
            ax.axvspan(xmin=0, xmax=0.03, facecolor='yellow', alpha=0.3)
        plt.savefig(f'{img_path}/raster_plots/{imgfname}_{iteration}.svg',facecolor='white', transparent=False)
        plt.close()


def GetFiringFrequency(ti,N,bins,t_window=20.):
    #IMPORTANT: CHECK UNITS
    hists,_ =np.histogram(ti,bins=bins)
    fireRate= np.zeros(len(bins))
    idx_count=1
    for n in hists:
        fireRate[idx_count] =1e3 * n / (t_window * N)
        idx_count+=1
    return fireRate,bins

def read_files(tname,dir_path):
    fnames = glob.glob(f"{dir_path}/**/output/**/*.h5",recursive=True)
    df_network = pd.read_hdf(f'{dir_path}/Network_{tname}.h5')
    return df_network,fnames

def plot_ISI(dir_path,tname):
    #TODO: TEST function again 
    data_path = f"{dir_path}/output"
    df_network,fnames = read_files(tname,dir_path) #TODO: REMOVE
    NTypes = df_network.Type.unique()
    print(NTypes)
    for datafname in fnames: 
    #-------------TODO: MOVE TO DIFFERENT FILE FOR OFFLINE ANALYSIS
        # os.makedirs(f"{img_path}",exist_ok=True)
        coarse = datafname.split('/')[-2]
        df_firingRate = pd.DataFrame()

        img_path = f"{dir_path}/img/{coarse}"
        
        os.makedirs(f"{img_path}",exist_ok=True)
        print(datafname)
        fine = datafname.split('/')[-1].split('_')[1].split('.')[0]
        print(f'--------------{coarse}-------{fine}---------------------')

        with pd.HDFStore(datafname) as store:
            griddimension = len(store.keys())//2
    
            fig_hist = plt.figure(figsize=(20, 20), constrained_layout=True)
            gs_hist = fig_hist.add_gridspec(griddimension, griddimension, hspace=0.2, wspace=0.2)
                    
            fig_freq = plt.figure(figsize=(20, 20), constrained_layout=True)
            gs_freq = fig_freq.add_gridspec(griddimension, griddimension, hspace=0.2, wspace=0.2)
        
            imgFname=f"{fine}"
            for keys,outer_gs_h,outer_gs_f in zip(store.keys(),gs_hist,gs_freq):
            # for keys in store.keys():
                print(f'--------------{coarse}-----------{keys}-------------------------')
                
                inner_gs_f = outer_gs_f.subgridspec(2,2,hspace=0, wspace=0)
                trial_axs_freq = inner_gs_f.subplots(sharex=True, sharey=True)

                inner_gs_h = outer_gs_h.subgridspec(2,2,hspace=0, wspace=0)
                trial_axs_hist = inner_gs_h.subplots(sharex=True, sharey=True)
                trial = keys.replace('/','')

                df_events = store.get(keys)
                df_events = df_events.set_index('Id').join(df_network.set_index('Id'))
                print('Plotting raster plot.....please wait')
                
                plot_events(df_events,img_path,fname=f'{imgFname}_{trial}')           
          
                active_neurons = df_events.reset_index().Id.unique()
                # print("Silent neurons",np.setdiff1d(active_neurons,all_neurons)) 
                t_window = 15 #(in miliseconds)
                bins =np.arange(0,max(df_events["Timestamp"].values),t_window)
                
                for ntype,ax_hist,ax_freq in zip(NTypes,trial_axs_hist.flatten(),trial_axs_freq.flatten()):                
                    if ntype != "Input":
                        # print(f'-------------------{ntype}-----------------------------------')
                        isi_DF = pd.DataFrame()
                        spike_data = df_events[(df_events.Type == ntype)].reset_index()[['Timestamp','Id']]
                        if not spike_data.empty:
                            N = df_network[df_network.Type==ntype].Id.shape[0]
                            group_fireRate,_ = GetFiringFrequency(spike_data.Timestamp.values,N=N,bins=bins,t_window=t_window)
                            rate_dict = dict(Rate = group_fireRate,Timebin =bins)
                            df_neuron_fr = pd.DataFrame(rate_dict)
                            df_neuron_fr['NeuronId'] = 999999
                            df_neuron_fr['Core'] = ntype
                            df_neuron_fr['Coarse'] = coarse
                            df_neuron_fr['Fine'] = fine
                            df_neuron_fr['Trial'] = trial
                        
                            df_firingRate = pd.concat((df_firingRate,df_neuron_fr),ignore_index=True)

                            # plot_frequencies(n_fireRate,bins,ntype,img_path,fname=fname)
                            ax_freq.plot(bins,group_fireRate)
                            ax_freq.set_title(f'{keys} |{ntype}')
                            ax_freq.set_xlabel('Time (msec)', fontsize=12)                      
                            ax_freq.set_ylabel('Frequency (Hz)', fontsize=12)
                            # ax_freq.set_xlim([isi_DF.inter_spike.min()-0.05,isi_DF.inter_spike.max()+0.05])
                            ax_freq.set_ylim([0,1000])  

                        active_neurons = spike_data.Id.unique()
                        for neuronId in active_neurons:
                            spikes = spike_data[spike_data.Id == neuronId].sort_values('Timestamp').Timestamp.values
                            neuron_fireRate,_ = GetFiringFrequency(spikes,N=1,bins=bins,t_window=t_window)
                            rate_dict = dict(Rate = neuron_fireRate,Timebin =bins)
                            df_neuron_fr = pd.DataFrame(rate_dict)
                            df_neuron_fr['NeuronId'] = neuronId
                            df_neuron_fr['Core'] = ntype
                            df_neuron_fr['Coarse'] = coarse
                            df_neuron_fr['Fine'] = fine
                            df_neuron_fr['Trial'] = trial
                            # print(df_neuron_fr.head())                 

                            df_firingRate = pd.concat((df_firingRate,df_neuron_fr),ignore_index=True)
                            isi =np.diff(spikes)
                            
                            if len(isi) > 1:
                                isi_dict = dict(Id = len(isi)*[neuronId],inter_spike = isi,Type = len(isi)*[ntype])
                                isi_DF = pd.concat((isi_DF,pd.DataFrame(isi_dict,columns=['Id','inter_spike','Type'])))
                            
                        if not isi_DF.empty:
                            
                            sns.histplot(isi_DF.reset_index(drop=True), x="inter_spike", binwidth=0.05,stat='percent',ax=ax_hist).set(xlim=(0,30),ylim=(0,30))
                            ax_hist.set_title(f'{keys} |{ntype}')

                            ax2 = ax_hist.inset_axes([0.3, .5, 0.2, .2], facecolor='lightgrey')
                            sns.histplot(isi_DF.reset_index(drop=True), x="inter_spike", binwidth=.05,stat='percent',ax=ax2)
                            ax2.set_title('Zoom in')
                            ax2.set_xlabel('Time (msec)', fontsize=10)                      
                            ax2.set_ylabel('Percent (%)', fontsize=10)
                            ax2.set_xlim([isi_DF.inter_spike.min()-0.1,isi_DF.inter_spike.max()+0.1])
                            # ax2.set_ylim([0,15])
                            for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                                label.set_fontsize(10)
                        else:
                            print('Not enought spikes')
                        
            fig_hist.suptitle('Inter-spike Interval distribution for all for cores')
            fig_hist.savefig(f'{img_path}/{imgFname}_inter_spike_interval.svg')
            plt.close(fig_hist)
            fig_freq.suptitle('Firing Frequency')
            fig_freq.savefig(f'{img_path}/{imgFname}_Frequency.svg')
            plt.close(fig_freq)


        print(df_firingRate.Fine.unique())
        hdf_key1,hdf_key2 = coarse.replace('-','_'), fine.replace('-','_')
        print(df_firingRate.Trial.unique())
        df_firingRate.to_hdf(f'{data_path}/FiringRates.h5',key=f'{hdf_key1}',append=True)

def load_plot_CV(data_path,img_path,Type='Pyr'):
    
    df_CV = pd.read_hdf(f'{data_path}/CV.h5',key=Type).reset_index(drop=True)
    
    figCV = plt.figure(figsize=(15,15)) 
    g = sns.FacetGrid(df_CV, col="Repeat",col_wrap=2,height=7,aspect=1,palette="Set3",margin_titles=True,sharex= False,sharey= True)

    g.map_dataframe(sns.scatterplot, x="Iteration", y="Id",hue='CV',s=20,marker='s',edgecolor=None)
    g.add_legend()
    g.tight_layout()
    plt.savefig(f'{img_path}/raster_plots/{Type}_CV.png',facecolor='white', transparent=False)
    plt.close(figCV)


def get_events_n_plot(data_path,img_path):
    fnames = glob.glob(f"{data_path}/NetworkActivity_*.h5",recursive=True)
    df_network_activity = pd.DataFrame()
    with pd.HDFStore(fnames[0]) as store:
        for key in store.keys():
            Iteration = key.replace('/','').split('_')[1]
            repeat = key.replace('/','').split('_')[-1]
            df_events = store.get(key).reset_index()
            df_events['Iteration'] = Iteration
            df_events['Repeat'] = repeat
            df_network_activity = pd.concat([df_network_activity,df_events],ignore_index=True)
    plot_events(df_network_activity,img_path)
    return df_network_activity

def main():
   
    dir_path = "/media/mb/Data/DYNAP/EI_homeostasis/data/onchip_experimentaldata/board_orange/2022-10-17/EI_persistancy_SHUNT/14-58"
    data_path = f"{dir_path}/data"
    img_path = f"{dir_path}/img"
    os.makedirs(f"{img_path}/raster_plots",exist_ok=True)   
    load_plot_CV(data_path,img_path)
    # df_network_activity = get_events_n_plot(data_path,img_path)

if __name__ == '__main__':
    main()
    