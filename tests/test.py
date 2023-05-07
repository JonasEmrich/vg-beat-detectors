import numpy as np
import pytest
import vg_beat_detectors

# test signals
triag = -abs(np.arange(1000)-501)
zeros = np.zeros(1000)

def test_basic():
    '''Tests basic usage with a triangular shaped signal. Should detect the middle point.'''
    detected = vg_beat_detectors.VisGraphDetector().find_peaks(triag)
    assert detected == [501]

def test_basic2():
    '''Tests basic usage with a const zero signal. Should detect nothing.'''
    detected = vg_beat_detectors.VisGraphDetector().find_peaks(zeros)
    assert detected.size == 0

def test_overlap_0():
    '''Tests peak detection with 0% overlap. Should work but might inaccurate.'''
    detected = vg_beat_detectors.VisGraphDetector(window_overlap=0.0).find_peaks(triag)
    assert detected == [501]

def test_overlap_1():
    '''Tests peak detection with 100% overlap. Should raise value error.'''
    with pytest.raises(ValueError):
        vg_beat_detectors.VisGraphDetector(window_overlap=1.0).find_peaks(triag)

def test_small_input():
    '''Tests peak detection with too small input.'''
    with pytest.raises(ValueError):
        x = np.ones(10)
        vg_beat_detectors.VisGraphDetector().find_peaks(x)

def test_None_input():
    '''Tests peak detection with None input.'''
    with pytest.raises(ValueError):
        vg_beat_detectors.VisGraphDetector().find_peaks(None)

def test_samplingrate():
    '''Tests negative sampling rate. Should raise value error.'''
    with pytest.raises(ValueError):
        vg_beat_detectors.VisGraphDetector(sampling_frequency=-5).find_peaks(triag)

def test_beta():
    '''Tests too large beta. Should raise value error.'''
    with pytest.raises(ValueError):
        vg_beat_detectors.VisGraphDetector(beta=55).find_peaks(triag)

def test_type():
    '''Tests wrong graph type. Should raise value error.'''
    with pytest.raises(ValueError):
        vg_beat_detectors.VisGraphDetector(graph_type="WHVG").find_peaks(triag)

def test_edge_weight():
    '''Tests wrong edge weight. Should raise value error.'''
    with pytest.raises(ValueError):
        vg_beat_detectors.VisGraphDetector(edge_weight="unknown_metric_goes_here").find_peaks(triag)

def test_window_length():
    '''Tests negative window length. Should raise value error.'''
    with pytest.raises(ValueError):
        vg_beat_detectors.VisGraphDetector(window_length=-3.0).find_peaks(triag)


