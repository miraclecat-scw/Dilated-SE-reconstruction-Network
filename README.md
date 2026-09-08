# Dilated-SE-reconstruction-Network
A Precipitation (TP) Reconstruction Network Designed to Explore Predictable Mechanisms
# Interpretable Deep Learning Framework for Lead-Time-Dependent Mechanism Exploration of the South China Sea Summer Monsoon

<p align="center">
A deep learning framework for investigating nonlinear relationships between atmospheric-oceanic states and subseasonal evolution of the South China Sea summer monsoon.
</p>

---

## Overview

This repository provides the implementation of an **interpretable deep learning framework for exploring the physical mechanisms governing subseasonal variability of the South China Sea (SCS) summer monsoon**.

The objective of this framework is not to develop an operational forecasting system, but to use deep learning as a nonlinear relationship extractor to investigate how antecedent atmospheric and oceanic states influence subsequent monsoon evolution.

By combining a deep learning model with an interpretation module, this framework aims to identify:

- The nonlinear relationships between climate background states and future monsoon evolution;
- The dominant physical drivers at different lead times;
- The transition of controlling mechanisms during subseasonal evolution.

---

# Scientific Motivation

The South China Sea summer monsoon exhibits pronounced subseasonal variability, which is controlled by interactions among:

- Atmospheric circulation;
- Oceanic thermal conditions;
- Moisture transport;
- Large-scale climate variability.

Traditional statistical approaches usually rely on predefined physical relationships and may struggle to capture complex nonlinear interactions among multiple climate variables.

This framework addresses two scientific questions:

1. **Can deep learning extract nonlinear relationships between antecedent climate states and subsequent SCS summer monsoon evolution?**

2. **How do dominant physical drivers vary with increasing lead time?**

The goal is to improve understanding of monsoon predictability sources rather than to construct an operational prediction system.

---

# Framework Overview

The framework consists of two major components:

```
Historical Atmospheric-Oceanic Climate States

                ↓

Deep Learning Nonlinear Mapping Network

                ↓

Future SCS Summer Monsoon Evolution

                ↓

Interpretability Analysis Module

                ↓

Lead-Time-Dependent Physical Mechanism Diagnosis
```

The deep learning model serves as a nonlinear mapping function that captures hidden relationships between climate states and subsequent monsoon variations.

The interpretation module further investigates which variables contribute most strongly at different lead times.

---

# Mechanism Exploration Task

Given historical climate conditions:

```
X(t-k:t)
```

where `X` represents averaged atmospheric-oceanic states during a historical period,

the model learns the nonlinear relationship with future monsoon evolution:

```
Y(t+Δ)
```

where:

- `t` represents the reference time;
- `k` represents the length of historical climate information;
- `Δ` represents the lead time used for mechanism diagnosis.

Different lead times are investigated to reveal how controlling factors evolve during subseasonal monsoon development.

---

# Input Dataset

The framework uses multi-variable atmospheric and oceanic fields, including:

- Sea surface temperature (SST);
- Atmospheric temperature;
- Zonal wind components;
- Meridional wind components;
- Geopotential height;
- Specific humidity;
- Divergence;
- Vorticity;
- Vertical velocity;
- Precipitation;
- Outgoing thermal radiation.

All variables are stored as daily two-dimensional spatial fields.

---

# Data Organization

The dataset is organized according to physical variables:

```
data/

├── div_200/

├── div_850/

├── q_700/

├── sst/

├── t_200/

├── t_850/

├── tp/

├── ttr/

├── u_200/

├── u_850/

├── v_200/

├── v_850/

├── vor_850/

├── w_500/

├── z_200/

└── z_850/
```

Each variable directory contains daily `.npy` files:

```
yyyymmdd.npy
```

Example:

```
sst/

├── 20200101.npy

├── 20200102.npy

└── ...
```

Files with identical dates represent the same climate state.

Example:

```
sst/20200101.npy

u_850/20200101.npy

v_850/20200101.npy
```

represent different physical variables at the same time.

---

# Data Format

Each `.npy` file contains a two-dimensional spatial field:

```
(height, width)
```

Example:

```python
import numpy as np

sst = np.load(
    "data/sst/20200101.npy"
)
```

All variables are interpolated onto the same spatial grid and can be combined into multi-variable inputs.

---

# Model Input and Output

## Input

The model receives historical climate states:

```
[Variables, Time, Height, Width]
```

where:

- `Variables` represent atmospheric and oceanic predictors;
- `Time` represents historical sequence information;
- `Height` and `Width` represent spatial dimensions.

---

## Output

The model generates representations of future SCS summer monsoon evolution at different lead times.

These outputs are used for:

- Relationship evaluation;
- Lead-time-dependent analysis;
- Physical mechanism interpretation.

---

# Interpretability Analysis

The key component of this framework is the **lead-time-dependent physical forcing analysis**.

The interpretation module evaluates the contribution of different climate variables during different stages of monsoon evolution.

This analysis helps answer:

- Which variables dominate early-stage monsoon evolution?
- When does oceanic memory become important?
- How do atmospheric circulation anomalies affect future monsoon states?
- How do controlling mechanisms transition with increasing lead time?

The framework connects:

```
Deep Learning Representation

            +

Physical Mechanism Understanding
```

---

# Repository Structure

```
.

├── README.md

├── data/
|    ├── train
|          └── README.md
|    └── test
|          └── README.md

├── models/
│    ├── model.py
│    ├── main.py
│    ├── valid.py
│    ├── earlystopping.py
│    ├── gradientshap.py
│    └── dataset.py

├── configs/
|       ├── leading0
|       ├── leading2
|       ├── leading4
|       ├── ········

```
---

# Summary

This repository provides a deep learning-based framework for investigating the physical mechanisms underlying subseasonal variability of the South China Sea summer monsoon.

Rather than focusing on operational forecasting, the framework aims to uncover how different atmospheric and oceanic factors contribute to monsoon evolution across different lead times.
