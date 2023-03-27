#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from biosppy import storage, ecg

file_name = "Feb_27_2022_Reed_Data_1.csv"
data = pd.read_csv(file_name)

# Rename columns containing a hyphen in the column title
# This improves the ease of accessing the columns in code
columns_to_rename = [d for d in data.columns if "-" in d]
new_column_names = [d.replace("-","_") for d in columns_to_rename]
dict_column_renaming = dict(zip(columns_to_rename, new_column_names))
data.rename(columns=dict_column_renaming, inplace = True)

# %%
fig, axs = plt.subplots(2, 1, figsize=(10,4))

# ECG was sampled at 500Hz: every 500 data points is 1s
ind_s = 7000 #index to start on
ind_e = 9000 #index to end at
axs[0].plot(data.ECG_Time[ind_s:ind_e], data.ECG_Value[ind_s:ind_e], "k-")

# PPG was sampled at 50Hz; every 50 points is 1s
ind_s = 700
ind_e = 900
axs[1].plot(data.TIA1_3_Time[ind_s:ind_e], data.TIA1_3_Value[ind_s:ind_e], "r-")

# %%

ecg_data = data[["ECG_Time", "ECG_Value"]]
with open("ecg_data.txt", 'w') as f:
    df_string = ecg_data.to_string(header=False, index=False)
    f.write(df_string)

with open("ecg.txt", 'w') as f:
    f.write("# Simple Text Format\n")
    f.write("# Sampling Rate (Hz):= 500.00\n")
    f.write("# Labels:= ECG\n")
    df_string = data["ECG_Value"].to_string(header=False, index=False)
    f.write(df_string)

ppg_data = data[["TIA1_3_Time", "TIA1_3_Value"]]
with open("ppg_data.txt", 'w') as f:
    df_string = ppg_data.to_string(header=False, index=False)
    f.write(df_string)

ppg_data = data["TIA1_3_Value"]
ppg_data = ppg_data.dropna()

with open("ppg.txt", 'w') as f:
    f.write("# Simple Text Format\n")
    f.write("# Sampling Rate (Hz):= 50.00\n")
    f.write("# Labels:= PPG\n")
    df_string = ppg_data.to_string(header=False, index=False)
    f.write(df_string)
# %%
from biosppy.signals import ecg, ppg

# Load and process ECG signal
signal, mdata = storage.load_txt('ecg.txt')
Fs = mdata['sampling_rate']
out = ecg.ecg(signal=signal, sampling_rate=Fs, show=True)
# Get indices of r-peaks (uses Hamilton-Tompkins r-peak detection algorithm)
rpeaks = out["rpeaks"]

# Load and process PPG signal
signal, mdata = storage.load_txt('ppg.txt')
Fs = mdata['sampling_rate']
out = ppg.ppg(signal=signal, sampling_rate=Fs, show=True)
# Get indices of PPG pulse onsets
onsets = out["onsets"]

#%% See where the onset time is
ind_s = 700
ind_e = 900
ppg_data = data[["TIA1_3_Time", "TIA1_3_Value"]]
ppg_data = ppg_data.dropna()
plt.figure()
plt.plot(ppg_data["TIA1_3_Time"][ind_s:ind_e], out['filtered'][ind_s:ind_e])
onsets_in_time_range = list(filter(lambda x: ind_s < x < ind_e, onsets))
for onset_i in onsets_in_time_range:
    onset_time = ppg_data["TIA1_3_Time"][onset_i]
    onset_val = out['filtered'][onset_i]
    plt.plot(onset_time, onset_val, "r*")
# %%

# Calculate list of PTTs
rpeak_f = 500
onset_f = 50
ptts = list()
for i in range(min(len(rpeaks), len(onsets))):
    rpeak_index = rpeaks[i] 
    # check this. added + 1 because was getting negative PTT values
    rpeak_time = rpeak_f / rpeak_index
    onset_index = onsets[i]
    onset_time = onset_f / onset_index
    ptt = onset_time - rpeak_time
    ptts.append(ptt)
print(ptts)

# %%

# plot filtered ppg signal
ppg_data = data[["TIA1_3_Time", "TIA1_3_Value"]].dropna()
plt.plot(ppg_data.TIA1_3_Time, out["filtered"])