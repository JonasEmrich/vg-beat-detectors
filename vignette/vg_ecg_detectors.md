---
title: "Fast and Sample-Accurate R-Peak Detectors Based on Visibility Graphs"
author: |
  | Jonas Emrich, Taulant Koka, Sebastian Wirth, Michael Muma
  |
  | Technische Universität Darmstadt
date: "2023-04-14"
output:
  html_document: 
    theme: flatly
    highlight: pygments
    toc: yes
    toc_depth: 1
    toc_float: yes
    css: vignette_styles.css
    keep_md: yes
  prettydoc::html_pretty:
    theme: tactile
    highlight: vignette
    toc: yes
    toc_depth: 2
  md_document:
    variant: markdown_github
link-citations: true
toc-title: "Table of Contents"    
csl: the-annals-of-statistics.csl # citation style https://www.zotero.org/styles
bibliography: refs.bib
nocite: |
  @emrich_vg_2023, @koka_vg_2022
vignette: |
  %\VignetteKeyword{ECG, R-peak detection, Visibility graph}
  %\VignetteEncoding{UTF-8}
  %\VignetteIndexEntry{Fast and Sample-Accurate R-Peak Detectors Based on Visibility Graphs}
  %\VignetteEngine{knitr::rmarkdown}
---



------------------------------------------------------------------------

# Motivation

This Python package provides an implementation of visibility graph (VG) based approaches for detecting R-peaks in ECG signals. The utilized visibility graph transformation maps a signal into a graph representation by expressing sampling locations as nodes and establishing edges between mutually visible samples.

In @emrich_vg_2023, @koka_vg_2022 benchmarking on several popular databases showed that the visibility graph based methods provide significantly superior performance compared to popular R-peak detectors and allow for sample-accurate R-peak detection.

# Installation

You can install the latest version of the 'vg-ecg-detectors' package from the [Python Package Index (PyPI)](https://pypi.org/project/vg-ecg-detectors/) by running:

```         
pip install vg-ecg-detectors
```

Additionally, the source code is available on [GitHub](https://github.com/JonasEmrich/vg-ecg-detectors).

# Quick Start

In the following, the basic usage of the 'FastNVG' and 'FastWHVG' detectors [@emrich_vg_2023] is illustrated, which utilize the natural visibility graph and a weighted horizontal visibility graph, respectively.

The 'vg-ecg-detetcors' package provides 'FastNVG' and 'FastWHVG' detector classes that are initialized with the *sampling frequency* 'fs' of the given ECG signal. R-peaks can then be determined by calling the detectors `find_peaks(ecg)` method and passing the ECG signal.

-   FastNVG


```python
from vg_ecg_detectors import FastNVG

detector = FastNVG(sampling_frequency=fs)
rpeaks = detector.find_peaks(ecg)
```

-   FastWHVG


```python
from vg_ecg_detectors import FastWHVG

detector = FastWHVG(sampling_frequency=fs)
rpeaks = detector.find_peaks(ecg)
```

## Complete Working Example

The following example demonstrates the usage of the 'FastNVG' detector. The use of the 'FastWHVG' works analogously.


```python
# import the FastNVG detector
from vg_ecg_detectors_emrich import FastNVG

# import packages used in this example
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram

# generate some ECG signal
fs = 360
ecg = electrocardiogram() # sampling frequency of this excerpt is 360Hz
time = np.arange(ecg.size) / fs

# visibility graph R-peak detector
detector = FastNVG(sampling_frequency=fs)
R_peaks = detector.find_peaks(ecg)

# plot signal and detected peaks
plt.plot(time,ecg)
plt.plot(time[R_peaks],ecg[R_peaks],'X')
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.xlim(9, 10.2)
#> (9.0, 10.2)
plt.ylim(-1, 1.5)
#> (-1.0, 1.5)
plt.show()
```

<img src="vg_ecg_detectors_files/figure-html/unnamed-chunk-4-1.png" width="85%" style="display: block; margin: auto;" />

------------------------------------------------------------------------

# Advanced Usage

For advanced and experimental usage the package provides a 'VisGraphDetector' base class in which a number of parameters can be set. For further explanation and understanding of the listed options and their influence consult the papers [@emrich_vg_2023, @koka_vg_2022].

The usage follows the same structure as above:


```python
from vg_ecg_detectors import VisGraphDetector

detector = VisGraphDetector(sampling_frequency=250,
                            graph_type='nvg',
                            edge_weight=None,
                            beta=0.55,
                            accelerated=False,
                            window_overlap=0.5,
                            window_seconds=2,
                            lowcut=4.0)
                            
rpeaks = detector.find_peaks(ecg)
```

## Visibility graph types

The underlying visibility graph transformation can be set with `graph_type`. The option 'nvg' results in the natural visibility graph and 'hvg' in the horizontal visibility graph.

## Weighted edges

The edges in the visibility graph representation can be constructed with a weighting factor. Therefore, the option `edge_weight` determines the metric for calculating the edge weight between two nodes (or samples).

Available weights are 'distance', 'sq_distance', 'v_distance', 'abs_v\_distance', 'h_distance', 'abs_h\_distance', 'slope', 'abs_slope', 'angle' and 'abs_angle' as well as None for no weighting.

For further explanation of each available weight see the [documentation of the 'ts2vg' package](https://cbergillos.com/ts2vg/api/graph_options.html#weighted-graphs)

## Accelerated and non-accelerated processing

The acceleration technique proposed in [@emrich_vg_2023] which reduces the input signal to only local maxima samples can be enabled or disabled by setting `accelerated` to `False` or `True` respectively. The acceleration results in a reduced run-time by one magnitude while the detection accuracy remains comparable with the non-accelerated detector.

## Sparsity parameter $\beta$

As described in [@emrich_vg_2023, @koka_vg_2022] the choice of the sparsity parameter $\beta \in [0, 1]$ depends on the used visibility graph transformation and edge weights and is a crucial setting for a well-functioning detector. Sparsity parameter values for the NVG and WHVG were determined by numerical experiments in [@emrich_vg_2023, @koka_vg_2022]. We highly recommend redetermining `beta` as described in [@emrich_vg_2023, @koka_vg_2022], when changes have been made in the `graph_type` and `edge_weight` options.

## Adjusting segments

The processing of the ECG signal is made in segments with a default length of $2 \sec$ and an overlap of $50\%$, i.e, `window_seconds=2` and `window_seconds=0.5`.

## Setting highpass cutoff frequency

To change the lower cutoff frequency of the highpass filter that pre-processes the input ECG signal, the parameter `lowcut` can be adjusted. The default value is $4 \mathrm{Hz}$.

# Outlook

Adapting the methods for related biomedical signals, such as PPG signals, is of great interest and will be the subject to future work.

...

# References {.unnumbered}

<!-- <div class="tocify-extend-page" data-unique="tocify-extend-page" style="height: 0px;"></div> -->