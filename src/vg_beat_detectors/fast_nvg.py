from vg_beat_detectors.vg_detector import VisGraphDetector


class FastNVG(VisGraphDetector):
    """
    Implementation of the ``FastNVG`` detector [1] providing fast and sample-accurate R-peak detection in ECG signals.

    Parameters
    ----------
    sampling_frequency : int
        The sampling frequency of the ECG signal in which R-peaks will be detected (in Hz, samples/second).
        Defaults to 250

    References
    ----------
    * [1] J. Emrich, T. Koka, S. Wirth, M. Muma, "Accelerated Sample-Accurate R-peak Detectors Based on Visibility Graphs",
      31st European Signal Processing Conference (EUSIPCO), 2023

    """

    def __init__(self, sampling_frequency=250):
        VisGraphDetector.__init__(self,
                                  sampling_frequency=sampling_frequency,
                                  graph_type='nvg',
                                  edge_weight=None,
                                  beta=0.55,
                                  accelerated=True)
