# Accelerated Sample-Accurate R-Peak Detectors Based on Visibility Graphs
This repository provides an algorithm for R-peak detection in ECG-signals based on a visibility graph approach. As part of my bachelor thesis, I modified an existing algorithm [1] and applied an acceleration technique [2] which was modified further. 
Those changes led to a significant improvement in terms of accuracy for the algorithm using the horizontal visibility graph, towards a level of the natural visibility graph or even greater. This was accomplished while the runtime was significantly reduced (for both visibility graph transformations).
- [1] T. Koka and M. Muma, “Fast and Sample Accurate R-Peak Detection for Noisy ECG Using Visibility
Graphs,” in 44th Annual International Conference of the IEEE Engineering in Medicine Biology Society
(EMBC), July 2022.
- [2] S. Wirth, “Radar-based blood pressure estimation,” M.S. Thesis, 2022.

## Used Python & Packages
- Python 3.10
- ts2vg 1.0.0
- scipy 1.8.0
- numpy 1.22.3

