import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

filedestination ='C:\Users\Dhiyanesh Srinivasan\OneDrive\Desktop\ALL EEG DATA\Dhiyanesh-Asynch-4.13.30msec.edf'
raw_data = mne.io.read_raw_edf(filedestination, preload=True)

selected_channels = ['EEG T5-O1', 'EEG T6-O2']
raw_data.pick_channels(selected_channels)

event_duration = 10
generated_events = mne.make_fixed_length_events(raw_data, id=1, duration=event_duration)

epoch_start = 0
epoch_end = 5
event_identifier = {'stimulus': 1}
epoch_data = mne.Epochs(raw_data, generated_events, event_identifier, epoch_start, epoch_end, baseline=(0, 0), preload=True)

data_T5_O1 = epoch_data.get_data(picks=['EEG T5-O1'])
data_T6_O2 = epoch_data.get_data(picks=['EEG T6-O2'])

hilbert_T5_O1 = hilbert(data_T5_O1, axis=2)
hilbert_T6_O2 = hilbert(data_T6_O2, axis=2)

phase_data_T5_O1 = np.angle(hilbert_T5_O1)
phase_data_T6_O2 = np.angle(hilbert_T6_O2)
phase_diff = phase_data_T5_O1 - phase_data_T6_O2

plv_value = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))

time_points = epoch_data.times
plt.figure(figsize=(10, 6))
plt.plot(time_points, plv_value[0])
plt.xlabel('Time (s)')
plt.ylabel('PLV')
plt.title('PLV between EEG T5-O1 and EEG T6-O2 at 13 Hz')
plt.show()

plv_mean_time = np.mean(plv_value, axis=1)
overall_mean_plv = np.mean(plv_mean_time)
print(f'Overall Mean PLV: {overall_mean_plv}')

plv_each_time_point = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
print(f'PLV for each time point: {plv_each_time_point[0]}')

mean_plv_all_points = np.mean(plv_each_time_point)
print(f'Mean PLV (all points): {mean_plv_all_points}')


