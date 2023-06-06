from scipy import signal
from scipy.fft import fft as scipy_fft
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

# dictionary of available graph metrics
# format: Dict{Key(name_of_metric): Value(Tupel(Functor_of_metric, List(directed_or_undirected)))}
_metrics_list = {'node_connectivity':                       ((lambda x: nx.node_connectivity(x)),                       ['directed', 'undirected']),
                'average_node_connectivity':                ((lambda x: nx.average_node_connectivity(x)),               ['directed', 'undirected']),
                'average_clustering':                       ((lambda x: nx.average_clustering(x)),                      ['directed', 'undirected']),
                'edge_connectivity':                        ((lambda x: nx.edge_connectivity(x)),                       ['directed', 'undirected']),
                'degree_assortativity_coefficient':         ((lambda x: nx.degree_assortativity_coefficient(x)),        ['directed', 'undirected']),
                'degree_pearson_correlation_coefficient':   ((lambda x: nx.degree_pearson_correlation_coefficient(x)),  ['directed', 'undirected']),
                'trophic_incoherence_parameter':            ((lambda x: nx.trophic_incoherence_parameter(x)),           ['directed']),
                'global_reaching_centrality':               ((lambda x: nx.global_reaching_centrality(x)),              ['directed', 'undirected']),
                'number_strongly_connected_components':     ((lambda x: nx.number_strongly_connected_components(x)),    ['directed']),
                'number_weakly_connected_components':       ((lambda x: nx.number_weakly_connected_components(x)),      ['directed']),
                'number_attracting_components':             ((lambda x: nx.number_attracting_components(x)),            ['directed']),
                'dag_longest_path_length':                  ((lambda x: nx.dag_longest_path_length(x)),                 ['directed']),
                'average_shortest_path_length':             ((lambda x: nx.average_shortest_path_length(x)),            ['directed', 'undirected']),
                'flow_hierarchy':                           ((lambda x: nx.flow_hierarchy(x)),                          ['directed']),
                'number_of_isolates':                       ((lambda x: nx.number_of_isolates(x)),                      ['directed', 'undirected']),
                'wiener_index':                             ((lambda x: nx.wiener_index(x)),                            ['directed', 'undirected']),
                'density':                                  ((lambda x: nx.density(x)),                                 ['directed', 'undirected']),
                'diameter':                                 ((lambda x: nx.diameter(x)),                                ['undirected']),
                'radius':                                   ((lambda x: nx.radius(x)),                                  ['undirected']),
                'estrada_index':                            ((lambda x: nx.estrada_index(x)),                           ['undirected']),
                'graph_clique_number':                      ((lambda x: max(len(c) for c in nx.find_cliques(x))),       ['undirected']),
                'graph_number_of_cliques':                  ((lambda x: sum(1 for _ in nx.find_cliques(x))),            ['undirected']),
                'number_connected_components':              ((lambda x: nx.number_connected_components(x)),             ['undirected']),
                'stoer_wagner':                             ((lambda x: nx.stoer_wagner(x)[0]),                         ['undirected']),
                'local_efficiency':                         ((lambda x: nx.local_efficiency(x)),                        ['undirected']),
                'global_efficiency':                        ((lambda x: nx.global_efficiency(x)),                       ['undirected']),
                'non_randomness':                           ((lambda x: nx.non_randomness(x)[0]),                       ['undirected']),
                'small_world_sigma':                        ((lambda x: nx.sigma(x)),                                   ['undirected']),
                'small_world_omega':                        ((lambda x: nx.omega(x)),                                   ['undirected']),
}


class VisGraphMetric:
    """
    This class provides several graph metrics computed on the visibility graph of a given input sequence (signal).

    Parameters
    ----------
    sampling_frequency : int
        The sampling frequency of the signal (in Hz, samples/second).
        Defaults to 250
    graph_type : str
        Specifies the visibility graph transformation used for computation. Has to be one of ["nvg", "hvg"].
        Defaults to ``"nvg"``
    direction : str
        Defines according to which direction edges are established. If 'None' an undirected graph is used. 
        Otherwise ["left_to_right", "top_to_bottom"] produce directed graphs.
        Defaults to ``"top_to_bottom"``
    edge_weight : str
        Specifies the metric used to weight the edges in the visibility graph. For unweighted edges choose ``None``.
        Has to be one of [None, 'slope', 'abs_slope', 'angle', 'abs_angle', 'distance', 'sq_distance', 'v_distance',
        'abs_v_distance', 'h_distance', 'abs_h_distance']. For further details consult the ts2vg package.
        Defaults to ``None``
    window_length : float
        Length of on data segment (in seconds!) used in the segment-wise processing of the signal. Defaults to 2
    window_overlap : float
        Overlap percentage (between 0 and 1) of the data segments used in the segment-wise computation. Defaults to 0.5
    freq_domain : bool
        Specifies if the visibility graph from which the metrics are generated is generated from the input signal in 
        the time domain (False) or frequency domain (True). Defaults to False

    """

    def __init__(self,
                 sampling_frequency=250,
                 graph_type="nvg",
                 direction='top_to_bottom',
                 edge_weight=None,
                 window_overlap=0.5,
                 window_length=2,
                 freq_domain=False
                 ):

        if sampling_frequency <= 0:
            raise ValueError(f"'sampling_frequency' has to be a positive non-zero value (got {sampling_frequency}).")
        self.fs = sampling_frequency

        if graph_type not in _graph_options:
            raise ValueError(f"Invalid 'graph_type' parameter: {graph_type}. Must be one of {_graph_options}.")
        self.graph_type = graph_type

        if direction not in _directions:
            raise ValueError(f"Invalid 'direction' parameter: {direction}. Must be one of {_directions}.")
        self.directed = direction

        if edge_weight not in _edge_weight_options:
            raise ValueError(f"Invalid 'edge_weight' parameter: {edge_weight}. Must be one of {_edge_weight_options}.")
        self.edge_weight = edge_weight

        if not 0.0 <= window_overlap < 1.0:
            raise ValueError(f"'window_overlap' mus be a value in the interval [0.0;1.0) (got {window_overlap}).")
        self.window_overlap = window_overlap

        if window_length <= 0:
            raise ValueError(f"'window_seconds' has to be a positive non-zero value (got {window_length}).")
        self.window_samples = window_length

        self._metrics_list = _metrics_list

        if not isinstance(freq_domain, bool):
            raise ValueError(f"'freq_domain' has to be of type bool (got {freq_domain}).")
        self.freq_domain = freq_domain


    def calc_metric(self, sig, idx=None, metrics="all"):
        """ Calculate several graph metrics on the visibility graph of the given signal. 
        For this the signal is partitioned into overlapping segments for which the visibility graph and the graph metric is computed. 

        Parameters
        ----------
        sig : np.array
            The signal which will be processed.
        idx : np.array
            Integer array indices of the given signal samples, defining the sample location in time when the given signal is sparse. 
            If None, then it is assumes that all samples are contiguous. Defaults to None
        metrics : 
            defaults to `"all"` which results in the calculation of all available metrics, while passing a list of the preferred metrics 
            result in the calculation of those (e.g. `['average_clustering','node_connectivity']`).
            NOTE: The choice of wether using a directed or undirected graph results in a different set of available metrics!

        Returns
        -------
        output : dict{tupel(metrics, window_indices)}
            Dictionary of the computed metrics. The key corresponds to the name of the metric, while the value is a tupel containing an array with the computed metric value in the first element, while the second element provides the starting-indicies of the corresponding windowed segments.

        """
        if sig is None:
            raise ValueError(f"The input signal 'sig' is None.")

        if metrics != "all" and not set(metrics).issubset(set(self._metrics_list.keys())):
            raise ValueError(f"At least one of the provided metrics is not known. Or the metric is only available for directed/undirected graphs.")

        # initialize some variables
        N = len(sig) if idx is None else idx[-1] # total signal length
        M = int(self.window_samples*self.fs)  # length of window segment
        l = 0  # Left segment boundary
        r = M  # Right segment boundary
        dM = int(np.ceil(self.window_overlap * M))  # Size of segment overlap
        L = int(np.ceil(((N - r) / (M - dM) + 1)))  # Number of segments

        # if input length is smaller than window, compute only one segment of this length without any overlap
        if N < M:
            M, r = N, N
            L = 1

        # initialize output 
        if metrics == "all":
            metrics = self._metrics_list.keys()
        
        output = {}
        for name in metrics:
            (function, attr) = self._metrics_list[name]
            if self.directed is None and 'undirected' not in attr:
                print(f'The metric {name} is not defined for undirected graphs. Set the parameter "direction" of the main class to one of {_directions[1:]}.')
                continue

            if self.directed is not None and 'directed' not in attr:
                print(f'The metric {name} is not defined for directed graphs. Set the parameter "direction" of the main class to `None`.')
                continue
            output[name] = (np.zeros(L), np.zeros(L))

        # do computation in small segments
        for jj in range(L):
            if idx is None:
                s = sig[l:r]
                t = None
            else: 
                idx_l = np.absolute(idx-l).argmin()
                idx_r = np.absolute(idx-r).argmin()
                s = sig[idx_l:idx_r]
                t = idx[idx_l:idx_r]
                #print(idx_l, idx_r, l, r, M,s, t)

            if self.freq_domain:
                s = self._ts2fft(s)
                t = None #TODO: fft for unevenly spaced samples
            else:
                s = s

            # compute vg graph
            vg = self._ts2vg(s,t)
            G = vg.as_networkx()

            for name in output.keys():
                (function, attr) = self._metrics_list[name]

                # # # Update weight vector # # #
                output[name][0][jj] = function(G)
                output[name][1][jj] = l

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

    def get_available_metrics(self):
        '''
        This function returns the names of all available graph metrics under the current configuration of the class, since several metrics are only defined for directed or undirected graphs.

        Returns
        -------
        metrics: list
            List of names (strings) of the available metrics.

        '''
        metrics = []
        for name, (function, attr) in self._metrics_list.items():
            if self.directed is None and 'undirected' not in attr:
                continue

            if self.directed is not None and 'directed' not in attr:
                continue

            metrics.append(name)
        return metrics

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

    def _ts2fft(self, ts):
        """computes the magnitude within the frequency domain of a time series input"""
        N = len(ts)
        yf = scipy_fft(ts)
        return 2.0/N * np.abs(yf[0:N//2])

