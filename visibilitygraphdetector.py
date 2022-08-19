import numpy as np
from scipy import signal
from ts2vg import HorizontalVG, NaturalVG

class VisGraphDetector:
    """
    Modified, Improved and Accelerated: Fast and R-Peak Detection Algorithm for Noisy ECG Signals Using Visibility Graphs
    Authors: Jonas Emrich
    Based on: "Fast and Sample Accurate R-Peak Detection for Noisy ECG Using Visibility Graphs" by Taulant Koka and Michael Muma
    """

    def __init__(self, sampling_frequency=250):
        """
        Takes in the sampling rate used for acquisition of the ECG-signal.
        """
        self.fs = sampling_frequency

    def highpass(self, lowcut=3, order=2):
        """
        Implements a highpass Butterworth filter. Takes in cutoff frequency 'lowcut', sampling frequency 'fs' and the order.
        Returns numerator (b) and denominator (a) polynomials of the IIR filter.
        """
        nyq = 0.5 * self.fs
        high = lowcut / nyq
        b, a = signal.butter(order, high, btype='highpass')
        return b, a

    def calc_weight(self, s, beta, max_inds=None, weighted=None, hvg=False):
        """
        This function computes the weights for the input signal s using its visibility graph transformation.
        The weights are calculated by iterative multiplication of the initial weights (1,1,..,1) with the adjacency matrix of the visibility graph
        and normalization at each iteration. Terminates when a predetermined amount of weights is equal to zero and returns the weights.

        Args:
            s (array): Signal for which to compute the weights.
            beta (float): Sparsity parameter between 0 and 1. Defines the termination criterion. Describes the portion of non-zero elements in weights.
            max_inds (ndarray): Indices of signal at which there are maxima
            weighted (str, optional): if 'None' create an undirected graph, otherwise use one of the following weighting options ['slope', 'abs_slope', 'angle', 'abs_angle', 'distance', 'sq_distance', 'v_distance', 'abs_v_distance', 'h_distance', 'abs_h_distance']. Defaults to None.
            hvg (bool, optional): if 'True' using horizontal visibility graph transformation instead of the natural visibility graph. Defaults to False.
        Returns:
            w (array):     Weights corresponding to Signal s.
        """
        size = len(s)

        # calculate the visibility graph
        direction = 'top_to_bottom'
        vg = HorizontalVG(directed=direction, weighted=weighted) if hvg else NaturalVG(directed=direction, weighted=weighted)
        if max_inds is not None:
            vg.build(s, max_inds)
        else:
            vg.build(s)

        edgelist = vg.edges

        # calculate adjacency matrix
        adjacency = np.zeros((size, size))
        for edge in edgelist:
            adjacency[edge[0]][edge[1]] = edge[2] if weighted is not None else 1

        # calculate weights through iterative k-Hops
        w = np.ones(size)
        while np.count_nonzero(w) > beta * size:
            Av = adjacency @ w
            w_new = abs(Av / np.linalg.norm(Av))

            if np.any(np.isnan(w_new)):
                break
            w = w_new
        return w

    def visgraphdetect(self, sig, beta=0.55, gamma=0.5, lowcut=4.0, window_seconds=2, filter=True, weighted=None,
                       hvg=False, remove_FP=False, accelerated=False, peak_correction=False):
        """
        This function implements a R-peak detector using the directed natural visibility graph.
            Takes in an ECG-Signal and returns the R-peak indices, the weights and the weighted signal.

        Args:
            sig <float>(array): The ECG-signal as a numpy array of length N.
            beta (float, optional): Sparsity parameter for the computation of the weights. Defaults to 0.55.
            gamma (float, optional): Overlap between consecutive segments in the interval (0,1). Defaults to 0.5.
            lowcut (float, optional): Cutoff frequency of the highpass filter in Hz. Defaults to 4.
            window_seconds (int, optional): Length of segment window in seconds. Determine the segment size. Defaults to 2
            filter (bool, optional): Enables the pre highpass filtering of the signal. Defaults to True.
            weighted (str, optional): if 'None' create an undirected graph, otherwise use one of the following weighting options ['slope', 'abs_slope', 'angle', 'abs_angle', 'distance', 'sq_distance', 'v_distance', 'abs_v_distance', 'h_distance', 'abs_h_distance']. Defaults to None.
            hvg (bool, optional): if 'True' using horizontal visibility graph transformation instead of the natural visibility graph. Defaults to False.
            accelerated (bool, optional): if the accelerated method should be used for computation. Defaults False.
        Returns:
            R_peaks <int>(array): Array of the R-peak indices.
            weights <float>(array): Array of length N containing the weights for the full signal.
        """

        # # # Initialize some variables # # #
        N = len(sig)  # total signal length
        M = int(window_seconds * self.fs)  # length of window segment
        l = 0  # Left segment boundary
        r = M  # Right segment boundary
        dM = int(np.ceil(gamma * M))  # Size of segment overlap
        L = int(np.ceil(((N - r) / (M - dM) + 1)))  # Number of segments
        weights = np.zeros(N)  # Empty array to store the weights

        # # # filter the signal with a highpass butterworth filter of order 2 # # #
        if filter:
            b, a = self.highpass(lowcut)
            sig = signal.filtfilt(b, a, sig)

        if accelerated:
            # # # iterate over signal and perform calculation on segments
            for jj in range(L):  # while right <= N and left<=right:
                s = sig[l:r]
                # # # reduce data length, extract only needed information # # #

                # Sebastan Wirth proposed a pre-selection by peaks
                # further reduction: remove all smaller than signal median since we assume that the R-peaks are greater
                thresh = np.quantile(s, 0.5)
                max_inds = signal.find_peaks(s)[0]
                max_heights = s[max_inds]
                lim = np.argwhere(max_heights>thresh).flatten()
                max_inds = max_inds[lim]
                max_heights = max_heights[lim]

                ### Compute the weights for the filtered signal ###
                w = self.calc_weight(max_heights, beta, max_inds=max_inds, weighted=weighted, hvg=hvg)

                ### Update full weight vector ###
                if N - dM + 1 <= l and l + 1 <= N:
                    weights[l + max_inds] = (weights[l + max_inds] + w)*0.5
                else:
                    weights[l + max_inds] = w

                if r - l < M:
                    break

                ### Update segment boundaries ###
                l = l + (M - dM)
                if r + (M - dM) <= N:
                    r = r + (M - dM)
                else:
                    r = N
        else:
            # # # Compute the weights for every segment of the filtered signal # # #
            for jj in range(L):
                s = sig[l:r]
                w = self.calc_weight(s, beta, weighted=weighted, hvg=hvg)

                # # # Update weight vector # # #
                if l == 0:
                    weights[l:r] = w
                elif N - dM + 1 <= l and l + 1 <= N:
                    weights[l:] = 0.5 * (w + weights[l:])
                else:
                    weights[l:l + dM] = 0.5 * (w[:dM] + weights[l:l + dM])
                    weights[l + dM:r] = w[dM:]

                # # # break loop, if end of signal is reached # # #
                if r - l < M:
                    break
                # # # Update segment boundaries # # #
                l += M - dM
                if r + (M - dM) <= N:
                    r += M - dM
                else:
                    r = N

        # # # weight the signal and use thresholding algorithm for the peak detection # # #
        weighted_sig = sig * weights
        R_peaks, peaks = self.panTompkinsPeakDetect(weighted_sig, sig)

        # # # optional postprocessing: correct R-peak positions to local maxima in signal
        if peak_correction:
            R_peaks = correct_peaks(R_peaks, sig, 0.04*self.fs)

        # # # optional postprocessing: remove false positive detection
        # # # DISCLAIMER: Further work may develop generalized decision rules. Until then, it probably gives better results to disable this step
        if remove_FP:
            # optional filtering to remove noise, according to Pan Tompkins
            nyq = self.fs / 2
            lowCut = 5 / nyq  #cut off frequencies are normalized from 0 to 1, where 1 is the Nyquist frequency
            highCut = 15 / nyq
            order = 2
            b,a = signal.butter(order, [lowCut, highCut], btype = 'bandpass')
            sig_band = signal.filtfilt(b, a, sig)

            sigma = 2 if hvg else 3
            R_peaks, removed = self.remove_by_comparision(R_peaks, sig_band, sigma)

        return np.array(R_peaks), weights

    def remove_by_comparision(self, peaks, sig, sigma):
        """
        Post Processing that removed False-Positive Detected R-Peaks based on comparison with distribution of a Metric.
        Peaks are then removed if they are outlier.

        Args:
            sig_unfilt <float>(array): unfilted version of the signal
            peaks <int>(array): Indices of the peaks found

        Returns:
            R_peaks <int>(array): Array of the accepted R-peak indices.

        """
        # # # calculate metric # # #
        metric = []
        for p in peaks:
            metric.append(self.slope_metric(sig, p))

        # # # determine estimators of distribution # # #
        loc = np.mean(metric)
        scale = np.std(metric)

        # # # return only peak indices that lie in the confidence interval # # #
        signal_peaks = []
        removed = []
        for p, m in zip(peaks, metric):
            if loc - sigma * scale <= m <= loc + sigma * scale:
                signal_peaks.append(p)
            else:
                removed.append(p)

        return signal_peaks, removed

    def slope_metric(self, x, k):
        """
        Evaluates the Slopes around a given sample (k). Calculates a symmetry metric of the sum of squared slopes around the peaks.
        Args:
            x (array): the data / signal that is used to calculate the metric
            k (int): index of the sample that should be analysed

        Returns:
            s (float): the calculated metric
        """
        m = int(0.04 * self.fs)
        l = max(k - m, 0)
        r = min(k + m + 1, len(x))

        # # other metric
        xi = x[l:r]
        max_y = (max(xi)+min(xi))/2  # Find the maximum y value
        xs = [i for i in range(r-l) if xi[i] > max_y]
        s = max(xs)-min(xs) # Print the points at half-maximum
        s /=(max(xi)-min(xi))

        return s

    def panTompkinsPeakDetect(self, weight, signal):
        """
        This function implements the thresholding introduced by Pan and Tompkins in [1]. Based on the implementation of [2]. Customized and modified further according to [1] and for useage with KHop Path.
        [1] J. Pan and W. J. Tompkins, “A Real-Time QRS Detection Algorithm,” IEEE Transactions on Biomedical Engineering, vol. BME-32, Mar. 1985. pp. 230-236.
        [2] https://github.com/berndporr/py-ecg-detectors
        """
        # # # initialise variables
        N = len(weight)
        max_heart_rate = 300  # in bpm
        min_distance = int(60 / max_heart_rate * self.fs)
        signal_peaks = [-0.36*self.fs]
        noise_peaks = []
        index = 0
        indexes = [0]
        regular_rr = []
        peaks = [0]
        RR_missed = 0

        # # # Learning Phase, 2sec
        SPKI = np.max(weight[0:2 * self.fs]) * 0.25  # running estimate of signal level
        NPKI = np.mean(weight[0:2 * self.fs]) * 0.5  # running estimate of noise level
        threshold_I1 = SPKI
        threshold_I2 = NPKI

        # # # iterate over the whole array / series
        for i in range(N):
            # skip first and last elements
            if 1 < i < N - 2:
                # peak candidates should lie above noise threshold
                if weight[i] < threshold_I2:
                    continue

                # detect peak candidates based on a rising + falling slope
                if weight[i - 1] <= weight[i] >= weight[i + 1] or signal_peaks[-1] == i - 1:  # or last point was peak
                    peak = i
                    peaks.append(i)

                    # peak candidates should be greater than signal threshold
                    if weight[peak] > threshold_I1:
                        # distance to last peak is greater than minimum detection distance
                        if (peak - signal_peaks[-1]) > 0.36*self.fs:
                            # for better sample accurate detection: test if neighbour point is a better candidate
                            if peaks[-2] != peak-1 and signal[(peak - 1)] >= signal[peak]:
                                signal_peaks.append(peak-1)
                                indexes.append(index-1)
                            else:
                                # label current candidate as real peak and update signal level
                                signal_peaks.append(peak)
                                indexes.append(index)
                            SPKI = 0.125 * weight[signal_peaks[-1]] + 0.875 * SPKI
                        # candidate is close to last detected peak -> check if current candidate is better choice
                        elif 0.2*self.fs > (peak - signal_peaks[-1]):
                            # for better sample accurate detection: last sample was peak, test if neighbour has higher signal level
                            if signal_peaks[-1] == i - 1 and signal[peak] > signal[peak-1]:
                                SPKI = (SPKI - 0.125 * weight[signal_peaks[-1]]) / 0.875  # reset threshold
                                signal_peaks[-1] = peak
                                indexes[-1] = index
                                SPKI = 0.125 * weight[signal_peaks[-1]] + 0.875 * SPKI
                            # compare slope of last peak with current candidate
                            elif signal_peaks[-1] != i - 1 and 2*weight[peak]-weight[peak-1]-weight[peak+1] > 2*weight[signal_peaks[-1]]-weight[signal_peaks[-1]-1]-weight[signal_peaks[-1]+1]: # test greater slope -> qrs
                                SPKI = (SPKI - 0.125 * weight[signal_peaks[-1]]) / 0.875  # reset threshold
                                signal_peaks[-1] = peak
                                indexes[-1] = index
                                SPKI = 0.125 * weight[signal_peaks[-1]] + 0.875 * SPKI
                            else:
                                noise_peaks.append(peak)
                                NPKI = 0.125 * weight[noise_peaks[-1]] + 0.875 * NPKI
                        else:
                            # not a peak -> label as noise and update noise level
                            noise_peaks.append(peak)
                            NPKI = 0.125 * weight[noise_peaks[-1]] + 0.875 * NPKI

                        # # # back search for missed peaks
                        if RR_missed != 0:
                            # if time difference of the last two signal peaks found is too large
                            if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                                missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                                missed_section_peaks2 = []
                                # search trough last interval
                                for missed_peak in missed_section_peaks:
                                    # if distance large enough to the peaks before and after AND candidate greater than second threshold
                                    if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                        -1] - missed_peak > min_distance and weight[missed_peak] > threshold_I2:
                                        missed_section_peaks2.append(missed_peak)

                                # if (missed_section_peaks == [] or missed_section_peaks[0] == signal_peaks[-2]) and (signal_peaks[-1] - signal_peaks[-2]) > min_distance:
                                #     missed_section_peaks2 = range(signal_peaks[-2]+int(min_distance/2), signal_peaks[-1]-int(min_distance/2))

                                # add largest peak in missed interval to peaks
                                if len(missed_section_peaks2) > 0:
                                    missed_peak = missed_section_peaks2[np.argmax(weight[missed_section_peaks2])]
                                    signal_peaks.append(signal_peaks[-1])
                                    signal_peaks[-2] = missed_peak

                    else:
                        # not a peak -> label as noise and update noise level
                        noise_peaks.append(peak)
                        NPKI = 0.125 * weight[noise_peaks[-1]] + 0.875 * NPKI

                    threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                    threshold_I2 = 0.5 * threshold_I1

                    # # # calculate back search parameters
                    # initialize RR averages
                    if 2 == len(signal_peaks):
                        RR_ave = signal_peaks[-1]-signal_peaks[-2]
                        RR_ave2 = RR_ave

                    if 2 < len(signal_peaks) <= 8:
                        RR = signal_peaks[-1]-signal_peaks[-2]
                        if 0.92*RR_ave2 <= RR <= 1.16*RR_ave2:
                            regular_rr.append(RR)
                            RR_ave2 = np.mean(regular_rr)

                    if len(signal_peaks) > 8:
                        RR = np.diff(signal_peaks[-9:])
                        RR_ave = int(np.mean(RR))

                        # test if last RR is in regular boundary's
                        if 0.92*RR_ave2 <= RR[-1] <= 1.16*RR_ave2:
                            regular_rr.append(RR[-1])
                            # if min 8 regular RR -> calculate RR_avg2
                            if len(regular_rr) >= 8:
                                RR_ave2 = np.mean(regular_rr[-8])
                        RR_missed = int(1.66 * RR_ave)

                        if RR_ave != RR_ave2:
                            #not regular
                            threshold_I1 *= 0.5
                            threshold_I2 *= 0.5
                        #else -> regular

                    index += 1

        # remove first dummy elements
        if signal_peaks[0] == -0.36*self.fs:
            signal_peaks.pop(0)
        indexes.pop(0)
        peaks.pop(0)

        return signal_peaks, peaks

def correct_peaks(detected_peaks, unfiltered_ecg, search_samples):
    '''
    corrects the detected peak locations to the maximum values in the time series for a sample accurate detection
    :param detected_peaks: the detected R-peaks
    :param unfiltered_ecg: original unfiltered ECG-signal
    :param search_samples: amount of samples that should be searched around the detected peak
    :return: corrected peaks
    '''
    r_peaks = []
    window = int(search_samples)

    for i in detected_peaks:
        i = int(i)
        l = max(0, i-window)
        r = min(len(unfiltered_ecg), i+window+1)

        section = unfiltered_ecg[l:r]
        r_peaks.append(l + np.argmax(section))

    return r_peaks