# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy import signal
import pandas as pd

import sys
sys.path.append("..")
sys.path.append("C:\\Users\\Jonas\\OneDrive\\Uni\\SoSe23\\RDS\\git-repo\\vg-ecg-detectors\\src")

from vg_beat_detectors.vg_metric import VisGraphMetric
from vg_beat_detectors.fast_nvg import FastNVG

path = "C:\\Users\\Jonas\\OneDrive\\Uni\\SoSe23\\RDS\\git-repo\\vg-ecg-detectors\\correlation.csv"

# %%
def pysionet_db(data='105', db='mitdb', filtered=False, left=0, right=None, normalize=False):
    sig = wfdb.rdsamp(data, sampfrom=left, sampto=right, pn_dir=db)[0][:, 0]

    annotations = wfdb.rdann(data, 'atr', sampfrom=left, sampto=right, pn_dir=db, summarize_labels=True)
    a_sym = np.array(annotations.symbol)
    annotation_pos = annotations.sample[np.in1d(a_sym, ['·','N','L','R', 'A', 'a', 'J', 'S', 'V','F', 'e', 'j', 'E', '/', 'f','Q', '!'])] - left

    artifacts = annotations.sample[np.in1d(a_sym, ['|'])] - left
    fs = annotations.fs

    t = np.arange(left, len(sig)+left,1)/fs


    if filtered:
        nyq = 0.5 * fs
        lowcut = 4      # cutoff frequency 4Hz
        order = 2
        high = lowcut / nyq
        b, a = signal.butter(order, high, btype='highpass')
        sig = signal.filtfilt(b, a, sig)

        nyq = fs / 2
        lowCut = 5 / nyq  #cut off frequencies are normalized from 0 to 1, where 1 is the Nyquist frequency
        highCut = 15 / nyq
        order = 2
        b,a = signal.butter(order, [lowCut, highCut], btype = 'bandpass')
        sig = signal.filtfilt(b, a, sig)

    if normalize:
        sig = sig - sig.min()
        sig = sig / sig.max()

    return t, sig, fs, annotation_pos


# %%
def compute_ACDC(rr, fs, PRSA=False):
   rr = rr/fs*1000
   segment_length = 4
   l = segment_length//2
   r = segment_length - l
   N = len(rr)

   # do the full computation if needed
   sum_acceleration = np.zeros(segment_length)
   num_acceleration = 0
   sum_decceleration = np.zeros(segment_length)
   num_decceleration = 0

   for i in range(l,N-r):
      # acceleration
      if rr[i] < rr[i-1] and rr[i] > 0.95 * rr[i-1]:
         sum_acceleration += rr[i-l:i+r]
         num_acceleration += 1
         # deceleration
      elif rr[i] > rr[i-1] and rr[i] < 1.05 * rr[i-1]:
         sum_decceleration += rr[i-l:i+r]
         num_decceleration += 1
        
   PRSA_AC = sum_acceleration/num_acceleration
   PRSA_DC = sum_decceleration/num_decceleration

   AC = (PRSA_AC[l]+PRSA_AC[l+1]-PRSA_AC[l-1]-PRSA_AC[l-2])/4
   DC = (PRSA_DC[l]+PRSA_DC[l+1]-PRSA_DC[l-1]-PRSA_DC[l-2])/4

   return AC, DC, PRSA_AC, PRSA_DC

def segmentwise_ACDC(rr, fs):
   rr = rr/fs*1000
   N = len(rr)

   values = []
   labels = []
   indices = []

   for i in range(2,N-1):
      # compute AC/DC formula if not outlier, i.e., in 5% bounds
      if not 1.05 * rr[i-1] > rr[i] > 0.95 * rr[i-1]:
         values.append(np.nan)
         labels.append("outlier")
         indices.append(i)
         continue

      values.append((rr[i]+rr[i+1]-rr[i-1]-rr[i-2])/4)
      indices.append(i)

      # label as AC or DC
      if rr[i] <= rr[i-1]:
         labels.append("AC")
      elif rr[i] > rr[i-1]:
         labels.append("DC")

   return np.array(values), np.array(labels), np.array(indices)

# %%
experiments = [100, 	104, 	108, 	113, 	117, 	122, 	201, 	207, 	212, 	217, 	222, 	231,
            101, 	105, 	109, 	114, 	118, 	123, 	202, 	208, 	213, 	219, 	223, 	232,
            102, 	106, 	111, 	115, 	119, 	124, 	203, 	209, 	214, 	220, 	228, 	233,
            103, 	107, 	112, 	116, 	121, 	200, 	205, 	210, 	215, 	221, 	230, 	234]

def calc_correlations(VGM, beats_per_window, metrics):
    for ex in experiments[2:4]:
        data = []
        t, x, fs, true_peaks = pysionet_db(data=str(ex), db='mitdb', filtered=False,normalize=False)#, right=int(5*60*360)
        if true_peaks.size == 0:
            continue
        rr = np.diff(true_peaks)

        # calculate AC/DC
        values, labels, indices = segmentwise_ACDC(rr, fs)
        values = values[labels != "outlier"]
        indices = indices[labels != "outlier"]
        labels = labels[labels != "outlier"]

        #print(metrics)
        m = VGM.calc_metric(rr, metrics=metrics)
        for name, (m, ix_m) in m.items():
            ix_m += beats_per_window//2 # shift to middle of window
            # get common indices
            idx = np.intersect1d(indices, ix_m)

            values = values[np.in1d(indices, idx)]
            m = m[np.in1d(ix_m, idx)]
            labels = labels[np.in1d(indices, idx)]
            
            AC_true = values[labels == "AC"]
            AC_pred = m[labels == "AC"]
            DC_true = values[labels == "DC"]
            DC_pred = m[labels == "DC"]

            AC = np.corrcoef(AC_true, AC_pred, rowvar=False)[0,1]
            DC = np.corrcoef(DC_true, DC_pred, rowvar=False)[0,1]
            ALL = np.corrcoef(values, m, rowvar=False)[0,1]

            if np.any(np.array([ALL, AC, DC]) > 0.5):
                print(f"\n {ex}: {name}_{str(VGM.directed)}_{str(VGM.edge_weight)}_{VGM.beats_per_window} \t AC: {AC} \t DC: {DC} \t ALL: {ALL}")

            data.append([ex, name, ALL, AC, DC, beats_per_window, edge_weight, direction, freq_domain])  

        df = pd.DataFrame(data, columns=['ex', 'name', 'ALL', 'AC', 'DC', 'beats_per_window', 'edge_weight', 'direction', 'freq_domain'])
        df.to_csv(path, index=False, mode='a', header=False)   
        break
# %%
beats_per_window = 4
freq_domain = False
directions = [None, 'left_to_right']
edge_weights = [None, 'slope', 'abs_slope', 'angle', 'abs_angle', 'distance', 'sq_distance', 'v_distance', 'abs_v_distance', 'h_distance', 'abs_h_distance']

#metrics=['minimum_cut_value_left_to_right', 'maximum_flow_value_left_to_right', 'shortest_path_length_left_to_right']
# VGM = VisGraphMetric(edge_weight="slope", direction=None, freq_domain=freq_domain, beats_per_window=beats_per_window, beats_per_step=1)
for edge_weight in edge_weights:
    for direction in directions:
        print(edge_weight, direction)
        VGM = VisGraphMetric(edge_weight=edge_weight, direction=direction, freq_domain=freq_domain, beats_per_window=beats_per_window, beats_per_step=1)
        metrics = VGM.get_available_metrics()
        calc_correlations(VGM, beats_per_window, metrics)




