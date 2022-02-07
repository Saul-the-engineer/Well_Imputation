# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:06:16 2020

@author: saulg
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils_data_augmentation
from time import time
from PyEMD import EEMD #pip install EMD-signal

# The purpose of this script is to load data and calculate intrinsic mode functions
# (imfs) and coressponding residual using Ensemble Empirical Mode Decomposition 
# (EEMD). This analysis decomposes a signal into compents that create time 
# aspect for original signal. This accounts for much of the groundwater accuracy.

# Data Locations
data_root ="./Datasets/"
figures_root = './Figures EEMD'
plot = False

# Importing well object class
DA = utils_data_augmentation.Data_Augmentation(data_root, figures_root)

# Load pickle Data
Data = DA.read_pickle('GLDAS_Data', data_root)
cell_names = list(Data.keys())
cell_names.remove('Location')


# Code is flexible enough to handle multiple cells with multiple variables
# such as GLDAS. That is why we use nested loop even though PDSI is a single
# variable. Load cell.
for i, cell in enumerate(cell_names):
    data_temp = Data[cell]
    startTime = time()
    for j, var in enumerate(data_temp.columns):
        try:
            if __name__ == "__main__":
                eemd = EEMD(trials= 100, 
                            noise_width = 0.05, 
                            ext_EMD=None, 
                            separate_trends= True,
                            DTYPE = np.float16,
                            spline_kind='akima',
                            parallel = True,
                            processes = 14)
                eemd.noise_seed(42)
                # Convert pdsi EEMD numpy array to Pandas Dataframe to keep indecies.
                eIMF = eemd(data_temp[var].values).T
                out = pd.DataFrame(eIMF, index=data_temp.index)
                label = [var + '_imf_' + str(k+1) for k in range(len(out.columns))]
                out.columns = label
                # Replace cell data with recalculated values        
                Data[cell] = data_temp.join(out, how='outer')
                if plot:
                    # Plot EEMD Results
                    eIMF = pd.concat([data_temp[var], out], join='outer', axis=1)
                    fig, axs = plt.subplots(nrows= eIMF.shape[1], ncols=1, figsize=(12,18))
                    fig.suptitle(str('Ensemble Empirical Mode Decomposition: ' + cell +' ' + var))
                    plt.subplots_adjust(top=.95, hspace=0.25)
                    plot_labels = ['IMF '+str(k) if k>0 else var for k in range(len(eIMF.columns))]
                    for k, _ in enumerate(eIMF):
                        axs[k].plot(data_temp.index, eIMF[eIMF.columns[k]])
                        axs[k].set(ylabel = plot_labels[k])
                    plt.show()
                    # Save Figure
                    fig_namepng = figures_root + '/' + str(cell) + '_EEMD' + '.png'
                    fig.savefig(fig_namepng, format="png", dpi=600 )
        except Exception as e:
            print(e)

    executionTime = (time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    print('Placeholder')
# Save pickle file
DA.Save_Pickle(Data, 'GLDAS_EEMD')



