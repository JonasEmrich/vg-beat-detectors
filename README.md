# Robust-Visibility-Graph-Methods-for-Biomedical-Signal-Processing
This repository provides the algorithm that I developed and modified as part of my bachelor thesis.

## Used Python & Packages
- Python 3.10
- ts2vg 1.0.0
- scipy 1.8.0
- numpy 1.22.3

## Abstract of my Thesis
The Electrocardiogram (ECG), which is a recording of the heart’s electrical activity, is one of the most commonly
used biomedical signals. Whether for heartbeat analysis and diagnosis of heart diseases in clinical trials or
for health and fitness monitoring with wearable devices, a fast and accurate R-peak detection, i.e., the
determination of heartbeat positions, has become increasingly important. Recently, a new approach for fast
and accurate R-peak detection has been developed that uses visibility graphs to map ECG-signals to a graph
representation and then detect R-peaks in the graph domain. However, due to the complexity of medical
phenomena, the occurrence of artifacts and outliers, and the need to process very large amounts of data
with limited computing and storage capacities, the demand for further improvements is still great. In this
Bachelors Project, it is shown that using an alternative visibility graph, i.e., the horizontal visibility graph
(HVG), which can be computed more efficiently, results in significantly lower accuracy. Therefore, during this
Bachelors Project, several techniques to improve the R-peak detection accuracy are proposed and evaluated on
both the HVG and the natural visibility graph (NVG) transformations. In addition, an acceleration procedure
that has been recently developed is introduced and further improved. It is shown, using the high precision
ECG Database from the Glasgow University, the MIT-BIH Arrhythmia Database, the MIT-BIH Noise Stress
Test Database and ECG data from the MyoVasc Study of the Johannes Gutenberg University Mainz that the
resulting detector based on the weighted HVG outperforms the original detector in terms of runtime, while
accuracy remains comparable or even increases. Furthermore, it is demonstrated that the proposed algorithm
based on the HVG requires significantly less memory for computation than the original detector. Generally
speaking, the runtime could be reduced by 91%, and the memory could be reduced by 58%, respectively. The
results of this Bachelors Project are highly useful for a wide range of applications and, in future, can be further
adapted to related biomedical signals.
