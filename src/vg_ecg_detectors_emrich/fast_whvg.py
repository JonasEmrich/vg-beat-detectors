from vg_detector import VisGraphDetector


class FastWHVG(VisGraphDetector):
    """
    Algorithms for fast and sample accurate R-peak detection in ECG signals based on visibility graphs.
    Implementing the FastWHVG and FastNVG R-peak detectors [1] based on a weighted version of the horizontal visibility graph as well as the natural visibility graph.
    Both detectors integrate an (optional) acceleration technique that reduces the computation time by an order of magnitude by processing only peak candidates of the input time series.

    Authors: Jonas Emrich, Taulant Koka [2]

    [1] J. Emrich, T. Koka, S. Wirth, M. Muma, "Accelerated Sample-Accurate R-peak Detectors Based on Visibility Graphs", 31st European Signal Processing Conference (EUSIPCO), 2023
    [2] T. Koka and M. Muma, "Fast and Sample Accurate R-Peak Detection for Noisy ECG Using Visibility Graphs", 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 2022, pp. 121-126
    """

    def __init__(self, sampling_frequency=250):
        VisGraphDetector.__init__(self,
                                  sampling_frequency=sampling_frequency,
                                  graph_type='hvg',
                                  edge_weight='abs_slope',
                                  beta=0.85,
                                  accelerated=True)
