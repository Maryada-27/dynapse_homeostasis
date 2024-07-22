# %%
import pandas as pd
import glob 
import os
import tables

green_path = '/media/mb/Data/DYNAP/EI_homeostasis/results/onchip_experimentaldata/board_green'
orange_path = '/media/mb/Data/DYNAP/EI_homeostasis/results/onchip_experimentaldata/board_orange'

dir_path = f"{orange_path}/**/EI_homeostasis*/**" 

data_path =f"{dir_path}/data"
img_path =f"{dir_path}/img"
fnames = glob.glob(f"{data_path}/NetworkActivity_*.h5",recursive=True)
print(f'{len(fnames)=}')

for fname in fnames:
    print(f'{fname=}')
    tables.file._open_files.close_all()
    chunk_size = 200000
    min_itemsize = dict(Id= 1000,Timestamp =1000,Type=1000, Iteration = 1000,Repeat = 1000)
    print(f'Open file... and read...{fname}')
    try:
        with pd.HDFStore(fname) as store:
            keys = store.keys()
            if len(keys) == 1 and 'homeostasis' in keys[0]:
                for chunk in pd.read_hdf(fname, chunksize=chunk_size,mode='r+'):
                    if 'Id' not in chunk.columns: 
                        chunk = chunk.reset_index()
                    chunk.to_hdf(f'{os.path.dirname(fname)}/compressed_NetworkActivity.h5',complevel=9,append=True,min_itemsize=min_itemsize,key=f'compressed')
                print(f'File Deleted...')
                os.remove(fname)
            else:
                print(f'File {fname} is not a homeostasis file')      
        print('done')
    except Exception as e:
        print(f'Error: {e}')
        continue