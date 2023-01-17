import numpy as np
from scipy import signal
from ts2vg import HorizontalVG, NaturalVG

class VisGraphDetector:
    """
    Further developed Visibility Graph Algorithm for fast and sample accurate R-peak detection in ECG-signals based on [1].
    Utilizing multiple Visibility Graph transformations (Natural Visibility Graph & Horizontal Visibility Graph) with several edge weights, an acceleration technique (based on the idea of [2]) as well as a local peak correction procedure.
    Authors: Jonas Emrich, Taulant Koka
    
    [1] T. Koka and M. Muma, "Fast and Sample Accurate R-Peak Detection for Noisy ECG Using Visibility Graphs," 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 2022, pp. 121-126
    [2] S. Wirth, "Radar-based Blood Pressure Estimation", Master Thesis, 2022
    """

    def __init__(self, sampling_frequency=250):
        """
        Takes in the sampling rate used for acquisition of the ECG-signal.
        """
        self.fs = sampling_frequency

    def highpass(self, lowcut=3, order=2):
        """
        Implements a highpass Butterworth filter. 
        Args:
            lowcut (float, optional): cutoff frequency. Defaults to 3.
            order (int, optional): the order of the filter. Defaults to 2.
        Returns:
            b: numerator polynomial of the IIR filter
            a: denominator polynomial of the IIR filter
        """
        nyq = 0.5 * self.fs
        high = lowcut / nyq
        b, a = signal.butter(order, high, btype='highpass')
        return b, a

    def calc_weight(self, s, beta, max_inds=None, weight=None, use_hvg=False):
        """
        This function computes the weights for the input signal s using its visibility graph transformation.
        The weights are calculated by iterative multiplication of the initial weights (1,1,..,1) with the adjacency matrix of the visibility graph
        and normalization at each iteration. Terminates when a predetermined amount of weights is equal to zero and returns the weights.
        Args:
            s (array): Signal for which to compute the weights.
            beta (float): Sparsity parameter between 0 and 1. Defines the termination criterion. Describes the portion of non-zero elements in weights.
            max_inds (ndarray, optional): Indices of signal at which the maximas are / samples that are considered. Only needed for accelerated computation. Defaults to None.
            weight (str, optional): if 'None' create an undirected graph, otherwise use one of the following weighting options ['slope', 'abs_slope', 'angle', 'abs_angle', 'distance', 'sq_distance', 'v_distance', 'abs_v_distance', 'h_distance', 'abs_h_distance']. Defaults to None.
            use_hvg (bool, optional): if 'True' using horizontal visibility graph transformation instead of the natural visibility graph. Defaults to False.
        Returns:
            w (array): Weights corresponding to Signal s.
        """
        size = len(s)

        # calculate the visibility graph
        direction = 'top_to_bottom'
        vg = HorizontalVG(directed=direction, weighted=weight) if use_hvg else NaturalVG(directed=direction, weighted=weight)
        if max_inds is not None:
            vg.build(s, max_inds)
        else:
            vg.build(s)

        edgelist = vg.edges

        # calculate adjacency matrix
        adjacency = np.zeros((size, size))
        for edge in edgelist:
            adjacency[edge[0]][edge[1]] = edge[2] if weight is not None else 1

        # calculate weights through iterative k-Hops
        w = np.ones(size)
        while np.count_nonzero(w) > beta * size:
            Av = adjacency @ w
            w_new = abs(Av / np.linalg.norm(Av))

            if np.any(np.isnan(w_new)):
                break
            w = w_new
        return w

    def visgraphdetect(self, sig, beta=0.55, gamma=0.5, lowcut=4.0, window_seconds=2, filtered=True, weight=None,
                       use_hvg=False, accelerated=True, peak_correction=True):
        """
        This function implements an R-peak detector using the directed natural visibility graph.
            Takes in an ECG-Signal and returns the R-peak indices.
        Args:
            sig <float>(array): The ECG-signal as a numpy array of length N.
            beta (float, optional): Sparsity parameter for the computation of the weights. Defaults to 0.55.
            gamma (float, optional): Overlap between consecutive segments in the interval (0,1). Defaults to 0.5.
            lowcut (float, optional): Cutoff frequency of the highpass filter in Hz. Defaults to 4.
            window_seconds (int, optional): Length of segment window in seconds. Determine the segment size. Defaults to 2
            filtered (bool, optional): Enables the pre highpass filtering of the signal. Defaults to True.
            weight (str, optional): if 'None' create an undirected graph, otherwise uses one of the following weighting options ['slope', 'abs_slope', 'angle', 'abs_angle', 'distance', 'sq_distance', 'v_distance', 'abs_v_distance', 'h_distance', 'abs_h_distance']. Defaults to None.
            use_hvg (bool, optional): if 'True' using horizontal visibility graph transformation instead of the natural visibility graph. Defaults to False.
            accelerated (bool, optional): if the accelerated method should be used for computation. Defaults False.
            peak_correction (bool, optional): Enables correction of R-peaks towards local maximum in sig
        Returns:
            R_peaks <int>(array): Array of the R-peak indices.
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
                w = self.calc_weight(max_heights, beta, max_inds=max_inds, weight=weight, use_hvg=use_hvg)

                ### Update full weight vector ###
                if l == 0:
                    weights[l + max_inds] = w
                elif N - dM + 1 <= l and l + 1 <= N:
                    weights[l + max_inds] = 0.5 * (weights[l + max_inds] + w)
                else:
                    weights[l + max_inds[max_inds<=dM]] = 0.5 * (w[max_inds<=dM] + weights[l + max_inds[max_inds<=dM]])
                    weights[l + max_inds[max_inds>dM]] = w[max_inds>dM]

                if r - l < M:
                    break

                ### Update segment boundaries ###
                l += M - dM
                if r + (M - dM) <= N:
                    r += M - dM
                else:
                    r = N
        else:
            # # # Compute the weights for every segment of the filtered signal # # #
            for jj in range(L):
                s = sig[l:r]
                w = self.calc_weight(s, beta, weight=weight, use_hvg=use_hvg)

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
        R_peaks, peaks = self.panTompkinsPeakDetect(weighted_sig)

        # # # optional postprocessing: correct R-peak positions to local maxima in signal
        if peak_correction:
            R_peaks = self.correct_peaks(R_peaks, sig, 0.04*self.fs)

        return np.array(R_peaks)#, weighted_sig

    def panTompkinsPeakDetect(self, weight):
        """
        This function implements the thresholding introduced by Pan and Tompkins in [3]. Based on the implementation of [2]. Customized and modified further according to [3] and for application with KHop Path.
        [3] J. Pan and W. J. Tompkins, “A Real-Time QRS Detection Algorithm,” IEEE Transactions on Biomedical Engineering, vol. BME-32, Mar. 1985. pp. 230-236.
        [4] https://github.com/berndporr/py-ecg-detectors
        Args:
            weight (array): the weighted signal that is thresholded
        Returns:
            signal_peaks (array): indices of detected R-peaks
            peaks (array): indices of peak candidates
        """
        # # # initialise variables
        N = len(weight)
        max_heart_rate = 240  # in bpm
        min_distance = int(60 / max_heart_rate * self.fs)
        signal_peaks = [-min_distance]
        noise_peaks = []
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
            if 0 < i < N - 1:

                # detect peak candidates based on a rising + falling slope
                if weight[i - 1] <= weight[i] >= weight[i + 1]:
                    # peak candidates should be greater than signal threshold
                    if weight[i] > threshold_I1:
                        # distance to last peak is greater than minimum detection distance
                        if (i - signal_peaks[-1]) > 0.3*self.fs:
                            signal_peaks.append(i)
                            SPKI = 0.125 * weight[signal_peaks[-1]] + 0.875 * SPKI
                        # candidate is close to last detected peak -> check if current candidate is better choice
                        elif 0.3*self.fs >= (i - signal_peaks[-1]):
                            # compare slope of last peak with current candidate
                            if weight[i] > weight[signal_peaks[-1]]: # test greater slope -> qrs
                                SPKI = (SPKI - 0.125 * weight[signal_peaks[-1]]) / 0.875  # reset threshold
                                signal_peaks[-1] = i
                                SPKI = 0.125 * weight[signal_peaks[-1]] + 0.875 * SPKI
                            else:
                                noise_peaks.append(i)
                                NPKI = 0.125 * weight[noise_peaks[-1]] + 0.875 * NPKI
                        else:
                            # not a peak -> label as noise and update noise level
                            NPKI = 0.125 * weight[i] + 0.875 * NPKI

                        # # # back search for missed peaks
                        if len(signal_peaks) > 8:
                            RR = np.diff(signal_peaks[-9:])
                            RR_ave = int(np.mean(RR))
                            RR_missed = int(1.66 * RR_ave)
                            
                            # if time difference of the last two signal peaks found is too large
                            if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                                threshold_I2 = 0.5 * threshold_I1
                                # get range of candidates and apply noise threshold
                                missed_section_peaks = range(signal_peaks[-2]+min_distance, signal_peaks[-1]-min_distance)
                                missed_section_peaks = [p for p in missed_section_peaks if weight[p] > threshold_I2 ]
                                        
                                # add largest sample in missed interval to peaks
                                if len(missed_section_peaks) > 0:
                                    missed_peak = missed_section_peaks[np.argmax(weight[missed_section_peaks])]
                                    signal_peaks.append(signal_peaks[-1])
                                    signal_peaks[-2] = missed_peak

                    else:
                        # not a peak -> label as noise and update noise level
                        NPKI = 0.125 * weight[i] + 0.875 * NPKI

                    threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)

        # remove first dummy elements
        if signal_peaks[0] == -min_distance:
            signal_peaks.pop(0)
        peaks.pop(0)

        return signal_peaks, peaks
    
    def correct_peaks(self, detected_peaks, unfiltered_ecg, search_samples):
        '''
        corrects the detected peak locations to the local maximum in the time series for a sample accurate detection
        Args:
            detected_peaks (array): indices of the detected R-peaks
            unfiltered_ecg (array): original unfiltered ECG-signal
            search_samples (int): amount of samples that should be searched around the detected peak
        Returns:
            r_peaks (array): corrected R-peaks
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
    
    
    