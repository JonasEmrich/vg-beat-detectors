import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.append("..")
sys.path.append("C:\\Users\\Jonas\\OneDrive\\Uni\\SoSe23\\RDS\\git-repo\\vg-ecg-detectors\\src")
from vg_beat_detectors.vg_metric import VisGraphMetric

import pandas as pd
import numpy as np
import wfdb
from ecg_gudb_database import GUDb
import time



def gudb_data(subject=1, experiment='walking', left=0, right=None, normalize=False, info=False):
    '''
    Returns Data from the "High precision ECG Database with annotated R peaks, recorded and filmed under realistic conditions" of the GUDB
    :param subject:     number of the subject (integer between 0 and 24)
    :param experiment:  ['sitting', 'maths', 'walking', 'hand_bike', 'jogging']
    :param filtered:    True/False if Filtered Data should be returned
    :param left:        left border of data interval that should be returned (max 120)
    :param right:       right border of data interval that should be returned (max 120)
    :param normalize:   True/False if the returned data should lie between a range of 0 to 1
    :param info:        True/False if additional information should be printed on console
    :return:            t: time vector, x: data vector, fs: sampling frequency, anno: the annotated R-Peaks if available
    '''
    fs = 250
    ecg_class = GUDb(subject, experiment)

    t = ecg_class.t
    x = ecg_class.cs_V2_V1

    if info:
        print("Available Experiments:", GUDb.experiments)
        print("Total Data length:", len(t))

    # fetching annotations
    if ecg_class.anno_cs_exists:
        anno = ecg_class.anno_cs
    else:
        print('No chest strap annotations')
        anno = np.array([])

    # truncate
    if right is None:
        right = len(t)

    left = int(left*fs)
    right = int(right*fs)
    t = t[left:right]
    x = x[left:right]
    if ecg_class.anno_cs_exists:
        anno = anno[(left <= anno) & (anno < right)] - left

    if normalize:
        x = x - x.min()
        x = x / x.max()

    return t, x, fs, anno

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

def tol_f1(validate, pred, tol, return_details=False):
    val = np.array(validate)
    pred = np.unique(np.array(pred))
    TP = 0
    FP = 0
    FN = 0
    FP_val = []
    for p in pred:
        diff = np.abs(val - p)
        i = (diff <= tol)
        if np.any(i):
            val_index = np.argmax(i)
            val = np.delete(val, val_index)
            TP += 1
        else:
            FP += 1
            FP_val.append(p)
    FP = len(pred)-TP
    FN = len(validate)-TP

    if return_details:
        return 2*TP / (2*TP + FP + FN), val, np.array(FP_val), TP, FP, FN
    else:
        return 2*TP / (2*TP + FP + FN)
    
def calc_rmssd(x):
    rr = np.diff(x)
    return np.sqrt(np.mean(np.square(np.diff(rr))))

def calcMedianPeakOffset(detected_peaks, true_peaks):
    r_peaks = []
    for i in true_peaks:
        r_peaks.append((i-detected_peaks)[np.argmin(np.absolute(detected_peaks-i))])

    m = int(np.median(r_peaks))
    return m
    
def detector(x, fs, method):
    porr_detector = Detectors(fs)
    taulant_detector = taulantdetector.VisGraphDetector(sampling_frequency=fs)
    vg_detector = jonasdetector.FastVGDetector(sampling_frequency=fs)
    
    if method == "FastNVG detector":
        return vg_detector.visgraphdetect(x, beta=.55, weight=None, use_hvg=False, accelerated=True)
    elif method == "NVG detector":
        return vg_detector.visgraphdetect(x, beta=.55, weight=None, use_hvg=False, accelerated=False)
    elif method == "HVG detector":
        return vg_detector.visgraphdetect(x, beta=.85, weight=None, use_hvg=True, accelerated=False)
    elif method == "FastWHVG detector":
        return vg_detector.visgraphdetect(x, beta=.85, weight='abs_slope', use_hvg=True, accelerated=True)
    elif method == "NVG detector (Koka & Muma)":
        return taulant_detector.visgraphdetect(x)
    elif method == "Elgendi et al (Two average)":
        return porr_detector.two_average_detector(x)
    elif method == "Matched filter":
        return porr_detector.matched_filter_detector(x)
    elif method == "Kalidas & Tamil (Wavelet transform)":
        return porr_detector.swt_detector(x)
    elif method == "Engzee":
        return porr_detector.engzee_detector(x)
    elif method == "Christov":
        return porr_detector.christov_detector(x)
    elif method == "Hamilton":
        return porr_detector.hamilton_detector(x)
    elif method == "Pan Tompkins":
        return porr_detector.pan_tompkins_detector(x)
    elif method == "WQRS":
        return porr_detector.wqrs_detector(x)
    
def compute_ACDC(rr, fs, PRSA=False):
    rr = rr/fs*1000
    segment_length = 4
    l = segment_length//2
    r = segment_length - l
    N = len(rr)

    # do the full computation if needed
    if PRSA:
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
    else:
        sum_acceleration = 0
        num_acceleration = 0
        sum_decceleration = 0
        num_decceleration = 0

        for i in range(l,N-r):
            # acceleration
            if rr[i] < rr[i-1] and rr[i] > 0.95 * rr[i-1]:
                sum_acceleration += rr[i]+rr[i+1]-rr[i-1]-rr[i-2]
                num_acceleration += 1
            # deceleration
            elif rr[i] > rr[i-1] and rr[i] < 1.05 * rr[i-1]:
                sum_decceleration += rr[i]+rr[i+1]-rr[i-1]-rr[i-2]
                num_decceleration += 1
        
        AC = sum_acceleration/num_acceleration/4
        DC = sum_decceleration/num_decceleration/4

        return AC, DC

path = "C:\\Users\\Jonas\\OneDrive\\Uni\\SoSe23\\RDS\\git-repo\\vg-ecg-detectors\\capacity_full.csv"
    
experiments = [100, 	104, 	108, 	113, 	117, 	122, 	201, 	207, 	212, 	217, 	222, 	231,
            101, 	105, 	109, 	114, 	118, 	123, 	202, 	208, 	213, 	219, 	223, 	232,
            102, 	106, 	111, 	115, 	119, 	124, 	203, 	209, 	214, 	220, 	228, 	233,
            103, 	107, 	112, 	116, 	121, 	200, 	205, 	210, 	215, 	221, 	230, 	234]

data = []
for ex in experiments:
    t, x, fs, true_peaks = pysionet_db(data=str(ex), db='mitdb', filtered=False,normalize=False, right=int(5*60*360))
    if true_peaks.size == 0:
        continue
    rr = np.diff(true_peaks)

    # original AC DC
    AC, DC = compute_ACDC(rr, fs)
    data.append([ex, AC, DC])

df = pd.DataFrame(data, columns=['ex', 'AC', 'DC'])
df = df.set_index('ex')
print(df.describe())
df.to_csv(path,index=True, mode='w', header=True)   

metrics=['shortest_path_length_left_to_right']
data = []
for ex in experiments:
    t, x, fs, true_peaks = pysionet_db(data=str(ex), db='mitdb', filtered=False,normalize=False, right=int(5*60*360))
    if true_peaks.size == 0:
        continue
    rr = np.diff(true_peaks)

    # metrics
    m = VisGraphMetric(edge_weight='slope', direction='left_to_right', freq_domain=False, beats_per_window=4, beats_per_step=1).calc_metric(rr, metrics=metrics)
    row = [ex]
    row.extend([np.median(metric) for metric, _ in m.values()])
    data.append(row)
    
    print(data[-1])

cols = ['ex']
cols.extend(["median_"+name for name in m.keys()])
df = pd.DataFrame(data, columns=cols)
df = df.set_index('ex')

df0 = pd.read_csv(path, index_col='ex')
df0 = df0.join(df)
print(df0.describe())
df0.to_csv(path, index=True, mode='w', header=True)            
exit()  






'''
records = [100, 	104, 	108, 	113, 	117, 	122, 	201, 	207, 	212, 	217, 	222, 	231,
            101, 	105, 	109, 	114, 	118, 	123, 	202, 	208, 	213, 	219, 	223, 	232,
            102, 	106, 	111, 	115, 	119, 	124, 	203, 	209, 	214, 	220, 	228, 	233,
            103, 	107, 	112, 	116, 	121, 	200, 	205, 	210, 	215, 	221, 	230, 	234]


for r_detector in list_detector:
    for r in records:
        scores = []
        t, x, fs, true_peaks = pysionet_db(data=str(r), db='mitdb', filtered=False,normalize=False)
        
        if true_peaks.size == 0:
            continue
            
        tol = 5
        peaks_r = detector(x, fs, r_detector)

        if len(peaks_r) > 0:
            offset = calcMedianPeakOffset(peaks_r, true_peaks)
            peaks_r = np.array(peaks_r)+offset            

        f1, val, FP_val, TP, FP, FN = tol_f1(true_peaks, peaks_r, tol, return_details=True)
        acc = TP / (FP + FN + TP+np.finfo(np.float32).eps)
        se = TP / (TP + FN+np.finfo(np.float32).eps) # recall
        p = TP / (TP + FP+np.finfo(np.float32).eps)  # precision
        der = (FP + FN) / (TP+np.finfo(np.float32).eps)# if TP != 0 else 999999
        rmssd = calc_rmssd(peaks_r)

        scores.append([r_detector, r, tol, len(true_peaks), TP, FP, FN, f1, acc, se, p, der, rmssd])
        print(r_detector, r, f1, acc, se, p, der)
        df = pd.DataFrame(scores, columns=['detector', 'ex', 'tol','true_peaks', 'TP', 'FP', 'FN', 'F1', 'Acc', 'Se', 'P', 'DER', 'RMSSD'])
        df.to_csv('mitdb_'+str(tol)+'tol_FIX.csv',index=False, mode='a', header=False)   
exit()


### GUDB ###        
experiments = ['sitting', 'maths', 'walking', 'hand_bike', 'jogging']
for r_detector in list_detector:
    for b in range(0,1,1):
        b = b/10
        scores = []
        for ex in experiments:
            for i in range(0, 25, 1):
                t, x, fs, true_peaks = gudb_data(subject=i, experiment=ex, info=False, normalize=False, left=0, right=120)

                if true_peaks.size == 0:
                        continue

                tol = 0
                peaks_r = detector(x, fs, r_detector)

                if len(peaks_r) > 0:
                    offset = calcMedianPeakOffset(peaks_r, true_peaks)
                    peaks_r = np.array(peaks_r)+offset

                f1, val, FP_val, TP, FP, FN = tol_f1(true_peaks, peaks_r, tol, return_details=True)
                acc = TP / (FP + FN + TP+np.finfo(np.float32).eps)
                se = TP / (TP + FN+np.finfo(np.float32).eps) # recall
                p = TP / (TP + FP+np.finfo(np.float32).eps)  # precision
                der = (FP + FN) / (TP+np.finfo(np.float32).eps)# if TP != 0 else 999999
                rmssd = calc_rmssd(peaks_r)

                scores.append([i, r_detector, ex, tol, len(true_peaks), TP, FP, FN, f1, acc, se, p, der, rmssd])
                print(r_detector, i, ex, f1, acc, se, p, der)
        df = pd.DataFrame(scores, columns=['sub', 'detector', 'ex', 'tol','true_peaks', 'TP', 'FP', 'FN', 'F1', 'Acc', 'Se', 'P', 'DER', 'RMSSD'])
        df.to_csv('gudb_'+str(tol)+'tol_FIX.csv',index=False, mode='a', header=False)  

exit()




experiments = ['118e24', '118e12', '118e06', '118e00', '118e_6', '119e24', '119e12', '119e06', '119e00', '119e_6']

for r_detector in list_detector:
    scores = []
    for s in [0]:
        for ex in experiments:
            for i in range(0, 1, 1):
                t, x, fs, true_peaks = pysionet_db(data=ex, db='nstdb/1.0.0', filtered=False, normalize=False)

                if true_peaks.size == 0:
                        continue

                tol = 5

                peaks_r = detector(x, fs, r_detector)
                
                if len(peaks_r) > 0:
                    offset = calcMedianPeakOffset(peaks_r, true_peaks)
                    peaks_r = np.array(peaks_r)+offset

                f1, val, FP_val, TP, FP, FN = tol_f1(true_peaks, peaks_r, tol, return_details=True)
                acc = TP / (FP + FN + TP+np.finfo(np.float32).eps)
                se = TP / (TP + FN+np.finfo(np.float32).eps) # recall
                p = TP / (TP + FP+np.finfo(np.float32).eps)  # precision
                der = (FP + FN) / (TP+np.finfo(np.float32).eps)# if TP != 0 else 999999
                rmssd = calc_rmssd(peaks_r)

                scores.append([i, r_detector, ex, s,len(true_peaks), TP, FP, FN, f1, acc, se, p, der, rmssd])
                print(s, r_detector, i, ex, f1, acc, se, p, der)
    df = pd.DataFrame(scores, columns=['sub', 'detector', 'ex', 'tol','true_peaks', 'TP', 'FP', 'FN', 'F1', 'Acc', 'Se', 'P', 'DER', 'RMSSD'])
    #df.to_csv('porr_5tol_nstdb.csv',index=False, mode='a', header=False)            
exit()  



### MITDB ###
records = [100, 	104, 	108, 	113, 	117, 	122, 	201, 	207, 	212, 	217, 	222, 	231,
            101, 	105, 	109, 	114, 	118, 	123, 	202, 	208, 	213, 	219, 	223, 	232,
            102, 	106, 	111, 	115, 	119, 	124, 	203, 	209, 	214, 	220, 	228, 	233,
            103, 	107, 	112, 	116, 	121, 	200, 	205, 	210, 	215, 	221, 	230, 	234]


for r_detector in list_detector:
    for r in records:
        scores = []
        t, x, fs, true_peaks = pysionet_db(data=str(r), db='mitdb', filtered=False,normalize=False)
        
        if true_peaks.size == 0:
            continue
            
        tol = 5
        peaks_r = detector(x, fs, r_detector)

        if len(peaks_r) > 0:
            offset = calcMedianPeakOffset(peaks_r, true_peaks)
            peaks_r = np.array(peaks_r)+offset            

        f1, val, FP_val, TP, FP, FN = tol_f1(true_peaks, peaks_r, tol, return_details=True)
        acc = TP / (FP + FN + TP+np.finfo(np.float32).eps)
        se = TP / (TP + FN+np.finfo(np.float32).eps) # recall
        p = TP / (TP + FP+np.finfo(np.float32).eps)  # precision
        der = (FP + FN) / (TP+np.finfo(np.float32).eps)# if TP != 0 else 999999
        rmssd = calc_rmssd(peaks_r)

        scores.append([r_detector, r, tol, len(true_peaks), TP, FP, FN, f1, acc, se, p, der, rmssd])
        print(r_detector, r, f1, acc, se, p, der)
        df = pd.DataFrame(scores, columns=['detector', 'ex', 'tol','true_peaks', 'TP', 'FP', 'FN', 'F1', 'Acc', 'Se', 'P', 'DER', 'RMSSD'])
        df.to_csv('mitdb_'+str(tol)+'tol_taulantoff_lfilt.csv',index=False, mode='a', header=False)   


          

    
    experiments = ['sitting', 'maths', 'walking', 'hand_bike', 'jogging']    
for ex in experiments:
    for i in range(0, 25, 1):
        runtime = []
        for r_detector in list_detector:
            t, x, fs, true_peaks = gudb_data(subject=i, experiment=ex, info=False, normalize=False, left=0, right=120)
        
            if true_peaks.size == 0:
                    continue

            
            start_time = time.time()
            
            peaks_r = detector(x, fs, r_detector)
            
            stop_time = time.time() - start_time
            runtime.append([r_detector, stop_time])
            print(r_detector, stop_time)


    
        rt = pd.DataFrame(runtime, columns=['detector', 'runtime'])
        rt.to_csv('porr_runtime_new.csv',index=False, mode='a', header=False)  
exit() 
'''