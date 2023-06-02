from scipy import signal
from ts2vg import HorizontalVG, NaturalVG
import numpy as np
import networkx as nx

_directions = [None, 'top_to_bottom', 'left_to_right']
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

_metrics_list_directed = {'node_connectivity': (lambda x: nx.node_connectivity(x)),
                'average_node_connectivity': (lambda x: nx.average_node_connectivity(x)),
                'average_clustering': (lambda x: nx.average_clustering(x)),
                'edge_connectivity': (lambda x: nx.edge_connectivity(x)),
                'degree_assortativity_coefficient': (lambda x: nx.degree_assortativity_coefficient(x)),
                'degree_pearson_correlation_coefficient': (lambda x: nx.degree_pearson_correlation_coefficient(x)),
                'trophic_incoherence_parameter': (lambda x: nx.trophic_incoherence_parameter(x)),
                'global_reaching_centrality': (lambda x: nx.global_reaching_centrality(x)),
                'number_strongly_connected_components': (lambda x: nx.number_strongly_connected_components(x)),
                'number_weakly_connected_components': (lambda x: nx.number_weakly_connected_components(x)),
                'number_attracting_components': (lambda x: nx.number_attracting_components(x)),
                'dag_longest_path_length': (lambda x: nx.dag_longest_path_length(x)),
                'average_shortest_path_length': (lambda x: nx.average_shortest_path_length(x)),
                'flow_hierarchy': (lambda x: nx.flow_hierarchy(x)),
                'number_of_isolates': (lambda x: nx.number_of_isolates(x)),
                'wiener_index': (lambda x: nx.wiener_index(x)),
                'density': (lambda x: nx.density(x)),
}

_metrics_list_undirected = {'diameter': (lambda x: nx.diameter(x)),
                            'radius': (lambda x: nx.radius(x)),
                            'estrada_index': (lambda x: nx.estrada_index(x)),
                            'graph_clique_number': (lambda x: max(len(c) for c in nx.find_cliques(x))),
                            'graph_number_of_cliques': (lambda x: sum(1 for _ in nx.find_cliques(x))),
                            'number_connected_components': (lambda x: nx.number_connected_components(x)),
                            'stoer_wagner': (lambda x: nx.stoer_wagner(x)[0]),
                            'local_efficiency': (lambda x: nx.local_efficiency(x)),
                            'global_efficiency': (lambda x: nx.global_efficiency(x)),
                            'non_randomness': (lambda x: nx.non_randomness(x)[0]),
                            'small_world_sigma': (lambda x: nx.sigma(x)),
                            'small_world_omega': (lambda x: nx.omega(x)),
}
_metrics_list = [_metrics_list_undirected, _metrics_list_directed]

_centrality_metrics_list = {'degree': (lambda x: nx.degree_centrality(x)),
                        'in_degree': (lambda x: nx.in_degree_centrality(x)),
                        'out_degree': (lambda x: nx.out_degree_centrality(x)),
                        #'eigenvector': (lambda x: nx.eigenvector_centrality(G, max_iter=600)), # does not converge everytime
                        'katz': (lambda x: nx.katz_centrality(x)),
                        'closeness': (lambda x: nx.closeness_centrality(x)),
                        'betweenness': (lambda x: nx.betweenness_centrality(x)),
                        'load': (lambda x: nx.load_centrality(x)),
                        'harmonic': (lambda x: nx.harmonic_centrality(x)),
                        'trophic_levels': (lambda x: nx.trophic_levels(x)),
                        'pagerank': (lambda x: nx.pagerank(x)),
                        #'laplacian': (lambda x: nx.laplacian_centrality(x, normalized=False)) # needs a lot of time
                        }

class VisGraphMetric:
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
    direction : str
        Defines according to which direction edges are established. If 'None' an undirected graoph is used. 
        Otherwise ["left_to_right", "top_to_bottom"] produce directed graphs.
        Defaults to ``"top_to_bottom"``
    edge_weight : str
        Specifies the metric used to weight the edges in the visibility graph. For unweighted edges choose ``None``.
        Has to be one of [None, 'slope', 'abs_slope', 'angle', 'abs_angle', 'distance', 'sq_distance', 'v_distance',
        'abs_v_distance', 'h_distance', 'abs_h_distance']. For further details consult the ts2vg package.
        Defaults to ``None``
    window_length : float
        Length of on data segment (in seconds!) used in the segment-wise processing of the ECG signal. Defaults to 2
    window_overlap : float
        Overlap percentage (between 0 and 1) of the data segments used in the segment-wise computation. Defaults to 0.5
    lowcut : float
        Cutoff frequency of the input highpass filter (in Hz). Defaults to 4.0

    """

    def __init__(self,
                 sampling_frequency=250,
                 graph_type="nvg",
                 direction='top_to_bottom',
                 edge_weight=None,
                 window_overlap=0.5,
                 window_length=2,
                 lowcut=4.0,
                 ):

        if sampling_frequency <= 0:
            raise ValueError(f"'sampling_frequency' has to be a positive non-zero value (got {sampling_frequency}).")
        self.fs = sampling_frequency

        self.lowcut = lowcut

        if graph_type not in _graph_options:
            raise ValueError(f"Invalid 'graph_type' parameter: {graph_type}. Must be one of {_graph_options}.")
        self.graph_type = graph_type

        if direction not in _directions:
            raise ValueError(f"Invalid 'direction' parameter: {direction}. Must be one of {_directions}.")
        self.directed = direction

        if direction is None:
            self._metrics_list =  _metrics_list[0]
        else:
            self._metrics_list = _metrics_list[1]

        if edge_weight not in _edge_weight_options:
            raise ValueError(f"Invalid 'edge_weight' parameter: {edge_weight}. Must be one of {_edge_weight_options}.")
        self.edge_weight = edge_weight

        if not 0.0 <= window_overlap < 1.0:
            raise ValueError(f"'window_overlap' mus be a value in the interval [0.0;1.0) (got {window_overlap}).")
        self.window_overlap = window_overlap

        if window_length <= 0:
            raise ValueError(f"'window_seconds' has to be a positive non-zero value (got {window_length}).")
        self.window_seconds = window_length


    def calc_centrality(self, sig, metrics="all"):
        """ Calcuate several node metrics for the given signal.

        Parameters
        ----------
        sig: np.array
            The ECG signal which will be processed.
        metrics: 
            defaults to `"all"` which results in the calculation of all available metrics, while passing a list of the preferred metrics 
            result in the calculation of those (e.g. `['in_degree','harmonic']`)

        Returns
        -------
        R_peaks : dict
            Dictionary of the computed metrics. The key corresponds to the name of the metric, while the value is an array containing the computed metric value for each node.

        """
        def _dict2array(dict):
            """converts dictionary into list ordered by indicies"""
            max_key = max(dict.keys())
            return np.array([dict.get(i, 0) for i in range(max_key + 1)])

        if sig is None:
            raise ValueError(f"The input signal 'sig' is None.")

        if self.directed is None:
            raise ValueError(f"The graph must be directed for this computation.")

        if metrics != "all" and not set(metrics).issubset(set(_centrality_metrics_list.keys())):
            raise ValueError(f"At least one of the provided metrics is not known.")

        # initialize some variables
        N = len(sig)  # total signal length
        M = int(self.window_seconds * self.fs)  # length of window segment
        l = 0  # Left segment boundary
        r = M  # Right segment boundary
        dM = int(np.ceil(self.window_overlap * M))  # Size of segment overlap
        L = int(np.ceil(((N - r) / (M - dM) + 1)))  # Number of segments
        output = {}

        # if input length is smaller than window, compute only one segment of this length without any overlap
        if N < M:
            M, r = N, N
            L = 1

        # filter the signal with a highpass butterworth filter
        sig = self._filter_highpass(sig)

        # do computation in small segments
        for jj in range(L):
            s = sig[l:r]
            print(len(s))

            # compute vg graph
            vg = self._ts2vg(s)
            G = vg.as_networkx()

            for name, function in _centrality_metrics_list.items():
                if metrics != "all" and name not in metrics:
                    continue
                segment = _dict2array(function(G))

                # # # Update weight vector # # #
                output = self._merge_segments(name, output, segment, l, r, dM, N)

            # # # break loop, if end of signal is reached # # #
            if r - l < M:
                break
            # # # Update segment boundaries # # #
            l += M - dM
            if r + (M - dM) <= N:
                r += M - dM
            else:
                r = N

        return output

    def calc_metric(self, sig, metrics="all"):
        """ Calcuate several metrics for the given signal.

        Parameters
        ----------
        sig: np.array
            The signal which will be processed.
        metrics: 
            defaults to `"all"` which results in the calculation of all available metrics, while passing a list of the preferred metrics 
            result in the calculation of those (e.g. `['in_degree','harmonic']`).
            NOTE: The choice of wether using a directed or undirected graph results in a different set of available metrics!

        Returns
        -------
        R_peaks : dict{tupel(metrics, window_indicies)}
            Dictionary of the computed metrics. The key corresponds to the name of the metric, while the value is a tupel containing an array with the computed metric value in the first element, while the second element provides the starting-indicies of the corresponding windowed segments.

        """
        if sig is None:
            raise ValueError(f"The input signal 'sig' is None.")

        if metrics != "all" and not set(metrics).issubset(set(self._metrics_list.keys())):
            raise ValueError(f"At least one of the provided metrics is not known. Or the metric is only available for directed/undirected graphs.")

        # initialize some variables
        N = len(sig)  # total signal length
        M = int(self.window_seconds * self.fs)  # length of window segment
        l = 0  # Left segment boundary
        r = M  # Right segment boundary
        dM = int(np.ceil(self.window_overlap * M))  # Size of segment overlap
        L = int(np.ceil(((N - r) / (M - dM) + 1)))  # Number of segments
        output = {}

        # if input length is smaller than window, compute only one segment of this length without any overlap
        if N < M:
            M, r = N, N
            L = 1

        # filter the signal with a highpass butterworth filter
        sig = self._filter_highpass(sig)

        # do computation in small segments
        for jj in range(L):
            s = sig[l:r]
            print(len(s))

            # compute vg graph
            vg = self._ts2vg(s)
            G = vg.as_networkx()

            for name, function in self._metrics_list.items():
                if metrics != "all" and name not in metrics:
                    continue

                # # # Update weight vector # # #
                output = self._append_segments(name, output, function(G), l)

            # # # break loop, if end of signal is reached # # #
            if r - l < M:
                break
            # # # Update segment boundaries # # #
            l += M - dM
            if r + (M - dM) <= N:
                r += M - dM
            else:
                r = N

        return output

    def _merge_segments(self, name, data, segment, l, r, dM, N):
        """merge a segment with the full data vector by averaging overlapping parts. Similar to Welchs method for spectral estimation."""
        if l == 0:
            data[name] = np.pad(segment, (0, N - (r-l)), 'constant')
        elif N - dM + 1 <= l and l + 1 <= N:
            data[name][l:] = 0.5 * (segment + data[name][l:])
        else:
            data[name][l:l + dM] = 0.5 * (segment[:dM] + data[name][l:l + dM])
            data[name][l + dM:r] = segment[dM:]
        return data

    def _append_segments(self, name, data, segment, l):
        if l == 0:
            data[name] = (segment, np.array(l))
        else:
            data[name] = (np.append(data[name][0], segment), np.append(data[name][1], l))
        return data

    def _filter_highpass(self, sig, order=2):
        """implements a butterworth highpass filter"""
        nyq = 0.5 * self.fs
        high = self.lowcut / nyq
        b, a = signal.butter(order, high, btype='highpass')
        return signal.filtfilt(b, a, sig)

    def _ts2vg(self, xs, ts=None):
        """translates time series (ts, xs) into visibility graph representation and returns adjacency matrix"""
        # calculate the visibility graph
        if self.graph_type in ['nvg', 'NVG']:
            vg = NaturalVG(directed=self.directed, weighted=self.edge_weight)
        else:
            vg = HorizontalVG(directed=self.directed, weighted=self.edge_weight)
        vg.build(xs, ts)
        return vg
    
    def _vg2adjacency(self, vg):
        """creates adjacency matrix from VG graph object"""
        edges = vg.edges
        size = vg.n_vertices
        adjacency = np.zeros((size, size))
        for edge in edges:
            adjacency[edge[0]][edge[1]] = 1 if self.edge_weight is None else edge[2]    
        return adjacency

