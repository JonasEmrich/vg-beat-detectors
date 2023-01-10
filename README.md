# Fast and Accurate R-Peak Detectors Based on Visibility Graph Transformations
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

## Performance
The plot below shows the detection performance ($F_1$-score) of the proposed algorithms (visibility graph algorithms) against common detectors of the NeuroKit2 package on an ECG database of the University of Glasgow [3]. A tolerance window of +/-0 samples for a sample accurate evaluation was used.
- [3] Howell, L. and Porr, B. (2018) High precision ECG Database with annotated R peaks, recorded and filmed under realistic conditions. [Data Collection], Datacite DOI: 10.5525/gla.researchdata.716 

![Evaluation on GUDB](plots/gudb_full_0sample.png?raw=true "Evaluation")

