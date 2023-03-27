
#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from biosppy import storage
from biosppy.signals import ecg, ppg


def init_data(file_name):
    data = pd.read_csv(file_name, index_col=False)
    # Rename columns containing a hyphen in the column title
    # This improves the ease of accessing the columns in code
    columns_to_rename = [d for d in data.columns if "-" in d]
    new_column_names = [d.replace("-","_") for d in columns_to_rename]
    dict_column_renaming = dict(zip(columns_to_rename, new_column_names))
    data.rename(columns=dict_column_renaming, inplace = True)
    return data

def make_ppg_file(data, file_name):
    ppg_data = -1*data["TIA1_3_Value"]
    ppg_data = ppg_data.dropna()

    with open(file_name, 'w') as f:
        f.write("# Simple Text Format\n")
        f.write("# Sampling Rate (Hz):= 50.00\n")
        f.write("# Labels:= PPG\n")
        df_string = ppg_data.to_string(header=False, index=False)
        f.write(df_string)

def get_ppg_info(file_name):
    signal, mdata = storage.load_txt(file_name)
    Fs = mdata['sampling_rate']
    out = ppg.ppg(signal=signal, sampling_rate=Fs, show=True)
    # Get indices of PPG pulse onsets
    onsets = out["onsets"]
    return onsets, out

ppg_file_name = "ppg.txt"
data = init_data("AFE4950_CAPTURED_DATA_Philjae_PPG_ECG_0327_3.csv") #all the data
make_ppg_file(data, ppg_file_name)
onsets, out = get_ppg_info('ppg.txt') 

#%% See where the onset time is
ind_s = 700
ind_e = 750
ppg_data = data[["TIA1_3_Time", "TIA1_3_Value"]]
ppg_data = ppg_data.dropna()

time = ppg_data["TIA1_3_Time"]
original_ppg = ppg_data["TIA1_3_Value"]
filtered_ppg = out['filtered']

plt.figure()
plt.plot(time[ind_s:ind_e], filtered_ppg[ind_s:ind_e])
onsets_in_time_range = list(filter(lambda x: ind_s < x < ind_e, onsets))
for onset_i in onsets_in_time_range:
    onset_time = ppg_data["TIA1_3_Time"][onset_i]
    onset_val = out['filtered'][onset_i]
    plt.plot(onset_time, onset_val, "r*")

#%%
def get_derivative(time, data):
    first_order_ppg_derivative = list()
    for i in range(1, len(time)):
        dv = data[i] - data[i-1]
        dt = time[i] - time[i-1]
        dvdt = dv/dt
        first_order_ppg_derivative.append([time[i], dvdt])
    first_order_ppg_derivative = np.array(first_order_ppg_derivative)
    return first_order_ppg_derivative

ppg_d1 = get_derivative(time, filtered_ppg)
ppg_d2 = get_derivative(ppg_d1[:,0], ppg_d1[:,1])
plt.figure()
plt.plot(ppg_d1[:,0][ind_s:ind_e], ppg_d1[:,1][ind_s:ind_e])
plt.plot(time[ind_s:ind_e], filtered_ppg[ind_s:ind_e])

plt.figure()
plt.plot(ppg_d2[:,0][ind_s:ind_e], ppg_d2[:,1][ind_s:ind_e],'k.')
plt.plot(time[ind_s:ind_e], filtered_ppg[ind_s:ind_e])
interp_ppg_d2 = np.interp(np.arange(14, 15, 0.001), ppg_d2[:,0][ind_s:ind_e], ppg_d2[:,1][ind_s:ind_e])
plt.plot(np.arange(14, 15, 0.001), interp_ppg_d2, "c.")

iszerocrossing = np.isclose(interp_ppg_d2, np.zeros(len(interp_ppg_d2)), atol=.05)
zerocross_inds = np.where(iszerocrossing == True)[0]
zerocross_inds_pos_d1 = list()
for ind in zerocross_inds:
    if ppg_d1[:,1][ind] > 0:
        zerocross_inds_pos_d1.append(ind)
plt.plot(np.arange(14, 15, 0.001)[zerocross_inds_pos_d1], interp_ppg_d2[zerocross_inds_pos_d1], "r*")

# %%
