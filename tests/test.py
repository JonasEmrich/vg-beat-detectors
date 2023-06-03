import numpy as np
import pytest
import vg_beat_detectors
from scipy.datasets import electrocardiogram

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
    '''Tests peak detection with input smaller than window size. Should produce same output as if the windowsize equals the signal length.'''
    a = vg_beat_detectors.VisGraphDetector(window_length=10, sampling_frequency=250).find_peaks(triag)
    b = vg_beat_detectors.VisGraphDetector(window_length=4, sampling_frequency=250).find_peaks(triag) # signal length is 1000, meaning 4sec by fs=250Hz
    assert a == b

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

def test_regression():
    '''This regression test evaluates the detector on a data snippet and compares its performace to a pre-determined
    level using the F1-score. This test ensures that the detection performance of new releases does not collapse due to
    errors. '''

    def tol_f1(validate, pred, tol):
        '''Calculate the F1-score with a tolerance window.'''
        val = np.array(validate)
        pred = np.unique(np.array(pred))
        TP = 0
        FP = 0
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
        return 2*TP / (2*TP + FP + FN)

    ecg = electrocardiogram()
    detected = vg_beat_detectors.VisGraphDetector(sampling_frequency=360).find_peaks(ecg)
    expected = np.array([125, 343, 552, 748, 944, 1130, 1317, 1501, 1691, 1880, 2065, 2251, 2431, 2608, 2779, 2956,
                         3125, 3292, 3456, 3614, 3776, 3948, 4129, 4310, 4482, 4652, 4812, 4984, 5157, 5323, 5496,
                         5674, 5857, 6048, 6250, 6454, 6663, 6865, 7055, 7245, 7423, 7608, 7797, 7975, 8174, 8376,
                         8582, 8789, 8987, 9178, 9372, 9567, 9751, 9936, 10112, 10306, 10512, 10708, 10907, 11103,
                         11287, 11472, 11658, 11845, 12037, 12221, 12404, 12601, 12804, 13005, 13207, 13403, 13606,
                         13810, 14012, 14223, 14409, 14611, 14825, 15038, 15258, 15877, 16074, 16274, 16476, 16662,
                         16865, 17302, 17515, 17909, 18099, 18290, 18479, 18666, 18856, 19065, 19281, 19505, 19723,
                         19943, 20154, 20356, 20560, 20759, 20955, 21170, 21379, 21592, 21803, 22001, 22203, 22379,
                         22598, 22798, 22987, 23173, 23367, 23567, 23770, 23964, 24164, 24366, 24566, 24757, 24947,
                         25142, 25347, 25536, 25835, 26030, 26265, 26477, 26685, 26917, 27142, 27344, 27566, 27791,
                         28009, 28267, 28485, 28683, 28940, 29168, 29357, 29609, 29830, 30033, 30280, 30506, 30722,
                         30936, 31146, 31307, 31569, 31764, 31963, 32179, 32395, 32615, 32819, 33051, 33269, 33437,
                         33881, 34077, 34277, 34480, 34676, 35611, 35829, 35978, 36167, 36359, 36566, 36779, 37251,
                         37470, 37905, 38102, 38294, 38508, 38732, 38954, 39181, 39396, 39606, 39820, 40022, 40222,
                         40429, 40651, 40864, 41149, 41394, 41591, 41862, 42066, 42268, 42465, 42666, 42876, 43089,
                         43288, 43485, 43671, 43869, 44072, 44271, 44464, 44650, 44860, 45090, 45312, 45533, 45756,
                         45973, 46193, 46405, 46616, 46806, 47006, 47205, 47410, 47620, 47822, 48022, 48219, 48417,
                         48620, 48813, 49064, 49308, 49693, 50030, 50243, 50442, 50643, 50845, 51041, 51323, 51530,
                         51833, 52076, 52235, 52522, 52714, 52938, 53137, 53328, 53591, 53806, 54026, 54241, 54468,
                         54655, 54812, 55110, 55494, 55689, 55886, 56083, 56281, 56470, 56662, 56874, 57109, 57305,
                         57583, 57773, 58006, 58216, 58429, 58655, 58885, 59077, 59364, 59557, 59783, 59993, 60207,
                         60433, 60662, 60856, 61126, 61340, 61516, 61710, 61977, 62164, 62409, 62641, 62837, 63093,
                         63293, 63485, 63715, 63926, 64105, 64389, 64602, 64840, 65054, 65256, 65451, 65643, 65822,
                         66004, 66180, 66351, 66536, 66726, 66923, 67121, 67327, 67525, 67739, 67936, 68130, 68322,
                         68499, 68697, 68905, 69116, 69326, 69538, 69710, 69975, 70185, 70375, 70615, 70829, 71027,
                         71316, 71533, 71792, 72009, 72432, 72638, 72829, 73042, 73282, 73490, 73810, 74011, 74285,
                         74509, 74707, 74891, 75188, 76892, 77093, 77580, 77811, 78015, 78250, 78471, 78903, 79105,
                         79546, 79776, 79996, 80224, 80445, 80943, 81169, 81605, 81832, 82297, 82521, 83012, 83238,
                         83597, 83899, 84341, 84549, 84758, 85168, 85368, 85565, 85764, 85967, 86174, 86367, 86599,
                         86807, 87006, 87204, 87406, 87606, 87785, 88029, 88250, 88456, 88738, 88972, 89198, 89380,
                         89638, 89838, 90048, 90263, 90475, 90731, 90953, 91154, 91373, 91587, 91805, 92006, 92200,
                         92409, 92594, 92894, 93101, 93384, 93608, 93809, 94038, 94251, 94464, 94678, 94886, 95099,
                         95310, 95524, 95740, 95932, 96217, 96411, 96621, 96834, 97056, 97274, 97499, 97701, 97926,
                         98143, 98341, 98594, 98803, 99008, 99256, 99493, 99697, 99969, 100186, 100389, 100796, 100993,
                         101208, 101441, 101657, 101905, 102111, 102322, 102531, 102732, 102922, 103113, 103323, 103530,
                         103726, 103949, 104166, 104361, 104643, 104863, 105303, 105741, 105950, 106152, 106355, 106566,
                         106782, 107219, 107423, 107607, 107871])
    assert tol_f1(detected, expected, 10) > 0.95

