import numpy as np
from matplotlib import pyplot as plt
from visibilitygraphdetector import VisGraphDetector
from scipy.misc import electrocardiogram

# # # load some data
x = electrocardiogram()
fs = 360
t = np.arange(0,len(x)/360,1/360)

# # detect R-peaks with the accelerated weighted HVG
detector = VisGraphDetector(sampling_frequency=fs)
peaks_r, weights = detector.visgraphdetect(x, beta=0.85, weighted='abs_slope', hvg=True, remove_FP=False, accelerated=True, peak_correction=True)

# # # plot some results # # #
fig, ax = plt.subplots()
ax.plot(t,x)
ax.plot(t[peaks_r],x[peaks_r],'X')
ax.set_title('ECG-signal')

# # # plot RR-intervals # # #
fig, ax = plt.subplots()
ax.plot(np.diff(peaks_r))
ax.set_title('RR-intervals')
