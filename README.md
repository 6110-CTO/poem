# Protected Online Entropy Matching (POEM) 🎼🎵🎶

Protected Online Entropy Matching (`POEM`) is a framework designed to enhance test-time domain adaptation via online self-training. The method dynamically updates model parameters based on distribution shifts in the test data, improving robustness and maintaining accuracy.

## Introduction

POEM introduces a statistically principled approach to detect and adapt to distribution shifts in real-time. The framework consists of two main components:

1. A statistical framework that monitors and detects distribution shifts in classifier entropy values on a stream of unlabeled samples.
2. An online adaptation mechanism that dynamically updates the classifier’s parameters based on detected distribution shifts.

The method leverages betting martingales and online learning to quickly react to distribution shifts, driving the distribution of test entropy values to match those from the source domain. This approach ensures improved test-time accuracy under distribution shifts while maintaining accuracy and calibration when no shifts are present.

For a detailed explanation of the methodology and experimental results, please refer to the relevant sections of the paper.

## Files Overview

### `cdf.py`
This script contains the implementation of our empirical Cumulative Distribution Function (CDF). The CDF is used to estimate the distribution of entropy values and is essential for detecting distribution shifts.

### `protector.py`
This file implements the protection algorithm using the test martingale and the SF-OGD (Scale-Free Online Gradient Descent) method, corresponding to Algorithm 1 in the paper. The protection mechanism helps in dynamically adjusting the model parameters to counteract detected shifts.

### `poem.py`
This is the core test-time adaptation model that utilizes the components from `cdf.py` and `protector.py` to perform online entropy matching and adapt the classifier during testing.

## Installation

To install the POEM repository, follow these steps:

```bash
git clone https://github.com/yarinbar/poem.git
cd poem
conda create --name poem python=3.10
conda activate poem
pip install -r requirements.txt
```

### Note on ImageNet V2

Due to an [issue with ImageNet V2](https://github.com/modestyachts/ImageNetV2/issues/10) not loading labels correctly, we have created a replacement for the `torchvision.datasets.folder` file. To use this replacement, copy our modified `folder.py` into your torchvision installation:

```bash
cp datasets/folder.py /path/to/torchvision/datasets/folder.py
```
Replace `/path/to/torchvision/` with the actual path where your torchvision package is installed. This path can typically be found by running:

```bash
python -c "import torchvision; print(torchvision.__file__)"
```

## Run Example

To run an example using the POEM framework, ensure you are in the `poem` directory and have activated the `poem` environment:

```bash
python run_example.py
```

This will execute a predefined script that demonstrates the capabilities of the POEM model on a selected dataset, showcasing its robustness and performance enhancements.

## Relevant Sections in the Paper

For detailed information on the methodologies and algorithms used in POEM, please refer to the following sections in the paper:

- **Section 3.3: Online Drift Detection** - This section explains the statistical framework for detecting distribution shifts using entropy values.
- **Section 3.4: Online Model Adaptation** - This section describes the online adaptation mechanism using the SF-OGD method.
- **Algorithm 1** - Detailed description of the protection algorithm implemented in `protector.py`.

We hope you find POEM useful for your test-time domain adaptation tasks. For any questions or issues, please feel free to open an issue on GitHub or contact the authors.
