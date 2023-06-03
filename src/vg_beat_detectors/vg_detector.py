from scipy import signal
from ts2vg import HorizontalVG, NaturalVG
import numpy as np

_graph_options = ['nvg', 'hvg', 'NVG', 'HVG']
_edge_weight_options = [None,
                        'slope',
                        'abs_slope',
                        'angle',
                        'abs_angle',
                        'distance',
                        'sq_distance',
                        'v_distance',
                        'abs_v_distance',
                        'h_distance',
                        'abs_h_distance']


class VisGraphDetector:
    """
    Base algorithm for detecting R-peaks in ECG signals using visibility graphs [1,2].
    This class is intended for advanced and experimental usage while the classes of the proposed detectors``FastNVG``
    and ``FastWHVG`` are implemented for a ready-to-use experience.

    Please consult the papers [1,2] for a further detailed explanation and understanding of the options and parameters
    available in this class.

    Parameters
    ----------
    sampling_frequency : int
        The sampling frequency of the ECG signal in which R-peaks will be detected (in Hz, samples/second).
        Defaults to 250
    graph_type : str
        Specifies the visibility graph transformation used for computation. Has to be one of ["nvg", "hvg"].
        Defaults to ``"nvg"``
    edge_weight : str
        Specifies the metric used to weight the edges in the visibility graph. For unweighted edges choose ``None``.
        Has to be one of [None, 'slope', 'abs_slope', 'angle', 'abs_angle', 'distance', 'sq_distance', 'v_distance',
        'abs_v_distance', 'h_distance', 'abs_h_distance']. For further details consult the ts2vg package.
        Defaults to ``None``
    beta : float
        Sparsity parameter (between 0 and 1) that determines the termination of the k-Hop paths algorithm. Must be
        determined with respect to the used visibility graph and edge weight. Proposed values are 0.55 for the NVG and
        0.85 for the HVG. Defaults to 0.55
    accelerated : bool
        Enables the data pre-processing in which the input signal is reduced only to local maxima which reduces the
        computation time by one order of magnitude while th performance remains comparable. Defaults to False.
    window_length : float
        Length of on data segment (in seconds!) used in the segment-wise processing of the ECG signal. Defaults to 2
    window_overlap : float
        Overlap percentage (between 0 and 1) of the data segments used in the segment-wise computation. Defaults to 0.5
    lowcut : float
        Cutoff frequency of the input highpass filter (in Hz). Defaults to 4.0

    References
    ----------
    * [1] J. Emrich, T. Koka, S. Wirth, M. Muma, "Accelerated Sample-Accurate R-peak Detectors Based on Visibility Graphs",
      31st European Signal Processing Conference (EUSIPCO), 2023
    * [2] T. Koka and M. Muma, "Fast and Sample Accurate R-Peak Detection for Noisy ECG Using Visibility Graphs",
      44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 2022, pp. 121-126
    """

    def __init__(self,
                 sampling_frequency=250,
                 graph_type="nvg",
                 edge_weight=None,
                 beta=0.55,
                 accelerated=False,
                 window_overlap=0.5,
                 window_length=2,
                 lowcut=4.0):

        if sampling_frequency <= 0:
            raise ValueError(f"'sampling_frequency' has to be a positive non-zero value (got {sampling_frequency}).")
        self.fs = sampling_frequency

        self.lowcut = lowcut

        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"'beta' mus be a value in the interval [0.0;1.0] (got {beta}).")
        self.beta = beta

        if graph_type not in _graph_options:
            raise ValueError(f"Invalid 'graph_type' parameter: {graph_type}. Must be one of {_graph_options}.")
        self.graph_type = graph_type

        if edge_weight not in _edge_weight_options:
            raise ValueError(f"Invalid 'edge_weight' parameter: {edge_weight}. Must be one of {_edge_weight_options}.")
        self.edge_weight = edge_weight

        if not 0.0 <= window_overlap < 1.0:
            raise ValueError(f"'window_overlap' mus be a value in the interval [0.0;1.0) (got {window_overlap}).")
        self.window_overlap = window_overlap

        if window_length <= 0:
            raise ValueError(f"'window_seconds' has to be a positive non-zero value (got {window_length}).")
        self.window_seconds = window_length

        self.accelerated = accelerated

    def find_peaks(self, sig):
        """
        Detects R-peaks in ECG signals using visibility graphs.

        Parameters
        ----------
        sig : np.array
            The ECG signal which will be processed.

        Returns
        -------
        R_peaks : np.array
            Array containing the sample locations (indices) of the determined R-peaks.

        """

        if sig is None:
            raise ValueError(f"The input signal 'sig' is None.")

        # initialize some variables
        N = len(sig)  # total signal length
        M = int(self.window_seconds * self.fs)  # length of window segment
        l = 0  # Left segment boundary
        r = M  # Right segment boundary
        dM = int(np.ceil(self.window_overlap * M))  # Size of segment overlap
        L = int(np.ceil(((N - r) / (M - dM) + 1)))  # Number of segments
        weights = np.zeros(N)  # Empty array to store the weights

        # if input length is smaller than window, compute only one segment of this length without any overlap
        if N < M:
            M, r = N, N
            L = 1

        # filter the signal with a highpass butterworth filter
        sig = self._filter_highpass(sig)

        # do computation in small segments
        for jj in range(L):
            s = sig[l:r]

            # compute weight vector using KHop paths and visibility graphs
            w, indices = self._calculate_weights(s)

            # update weight vector
            if l == 0:
                weights[l + indices] = w
            elif N - dM + 1 <= l and l + 1 <= N:
                weights[l + indices] = 0.5 * (weights[l + indices] + w)
            else:
                weights[l + indices[indices <= dM]] = 0.5 * (w[indices <= dM] + weights[l + indices[indices <= dM]])
                weights[l + indices[indices > dM]] = w[indices > dM]

            if r - l < M:
                break

            # update segment boundaries
            l += M - dM
            if r + (M - dM) <= N:
                r += M - dM
            else:
                r = N

        # weight signal with obtained weight and use thresholding algorithm for peak location extraction
        weighted_sig = sig * weights
        R_peaks = self._pantompkins_threshold(weighted_sig)
        return R_peaks

    def _filter_highpass(self, sig, order=2):
        """implements a butterworth highpass filter"""
        nyq = 0.5 * self.fs
        high = self.lowcut / nyq
        b, a = signal.butter(order, high, btype='highpass')
        return signal.filtfilt(b, a, sig)

    def _calculate_weights(self, sig):
        """calculates k-Hop weights from adjacency matrix"""
        if self.accelerated:
            ts, xs = self._reduce(sig)
        else:
            ts, xs = np.arange(len(sig)), sig

        adjacency = self._ts2adjacency(ts, xs)
        w = self._khop_paths(adjacency, self.beta)
        return w, ts

    def _reduce(self, sig):
        """reduces signal by only considering sufficiently large local peaks"""
        threshold = np.quantile(sig, 0.5)
        indices = signal.find_peaks(sig)[0]
        values = sig[indices]
        greater = np.argwhere(values > threshold).flatten()
        return indices[greater], values[greater]

    def _ts2adjacency(self, ts, xs):
        """translates time series (ts, xs) into visibility graph representation and returns adjacency matrix"""
        size = len(xs)
        # calculate the visibility graph
        if self.graph_type in ['nvg', 'NVG']:
            vg = NaturalVG(directed='top_to_bottom', weighted=self.edge_weight)
        else:
            vg = HorizontalVG(directed='top_to_bottom', weighted=self.edge_weight)
        vg.build(xs, ts)

        # convert list of edges into adjacency matrix
        adjacency = np.zeros((size, size))
        for edge in vg.edges:
            adjacency[edge[0]][edge[1]] = 1 if self.edge_weight is None else edge[2]
        return adjacency

    def _khop_paths(self, adjacency, beta):
        """calculates weights through iterative k-Hop paths metric"""
        size = len(adjacency)
        w = np.ones(size) / size
        while np.count_nonzero(w) > beta * size:
            Av = adjacency @ w
            w_new = abs(Av / np.linalg.norm(Av))
            if np.any(np.isnan(w_new)):
                break
            w = w_new
        return w

    def _pantompkins_threshold(self, weight):
        """
        Based on the python implementation [3] of the thresholding proposed by Pan and Tompkins in [4].
        Modifications were made with respect to the application of R-peak detection using visibility graphs
        and the k-Hop paths metric.

        References
        ----------
        [3] https://github.com/berndporr/py-ecg-detectors
        [4] J. Pan and W. J. Tompkins, “A Real-Time QRS Detection Algorithm”, IEEE Transactions on Biomedical
         Engineering, vol. BME-32, Mar. 1985. pp. 230-236.

        """
        # initialise variables
        N = len(weight)
        min_distance = int(0.25 * self.fs)
        signal_peaks = [-min_distance]
        noise_peaks = []
        peaks = [0]

        # Learning Phase, 2sec
        spki = np.max(weight[0:2 * self.fs]) * 0.25  # running estimate of signal level
        npki = np.mean(weight[0:2 * self.fs]) * 0.5  # running estimate of noise level
        threshold_I1 = spki

        # iterate over the whole array / series
        for i in range(N):
            # skip first and last elements
            if 0 < i < N - 1:
                # detect peak candidates based on a rising + falling slope
                if weight[i - 1] <= weight[i] >= weight[i + 1]:
                    # peak candidates should be greater than signal threshold
                    if weight[i] > threshold_I1:
                        # distance to last peak is greater than minimum detection distance
                        if (i - signal_peaks[-1]) > 0.3 * self.fs:
                            signal_peaks.append(i)
                            spki = 0.125 * weight[signal_peaks[-1]] + 0.875 * spki
                        # candidate is close to last detected peak -> check if current candidate is better choice
                        elif 0.3 * self.fs >= (i - signal_peaks[-1]):
                            # compare slope of last peak with current candidate
                            if weight[i] > weight[signal_peaks[-1]]:  # test greater slope -> qrs
                                spki = (spki - 0.125 * weight[signal_peaks[-1]]) / 0.875  # reset threshold
                                signal_peaks[-1] = i
                                spki = 0.125 * weight[signal_peaks[-1]] + 0.875 * spki
                            else:
                                noise_peaks.append(i)
                                npki = 0.125 * weight[noise_peaks[-1]] + 0.875 * npki
                        else:
                            # not a peak -> label as noise and update noise level
                            npki = 0.125 * weight[i] + 0.875 * npki

                        # back search for missed peaks
                        if len(signal_peaks) > 8:
                            RR = np.diff(signal_peaks[-9:])
                            RR_ave = int(np.mean(RR))
                            RR_missed = int(1.66 * RR_ave)

                            # if time difference of the last two signal peaks found is too large
                            if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                                threshold_I2 = 0.5 * threshold_I1
                                # get range of candidates and apply noise threshold
                                missed_section_peaks = range(signal_peaks[-2] + min_distance,
                                                             signal_peaks[-1] - min_distance)
                                missed_section_peaks = [p for p in missed_section_peaks if weight[p] > threshold_I2]

                                # add the largest sample in missed interval to peaks
                                if len(missed_section_peaks) > 0:
                                    missed_peak = missed_section_peaks[np.argmax(weight[missed_section_peaks])]
                                    signal_peaks.append(signal_peaks[-1])
                                    signal_peaks[-2] = missed_peak

                    else:
                        # not a peak -> label as noise and update noise level
                        npki = 0.125 * weight[i] + 0.875 * npki

                    threshold_I1 = npki + 0.25 * (spki - npki)

        # remove first dummy elements
        if signal_peaks[0] == -min_distance:
            signal_peaks.pop(0)
        peaks.pop(0)

        return np.array(signal_peaks)
