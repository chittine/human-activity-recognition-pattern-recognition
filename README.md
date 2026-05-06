# Human Activity Recognition using Pattern Recognition and Machine Learning

## Overview

This project evaluates human activity recognition using high-dimensional sensor data from smartphone and wearable devices. The goal is to compare multiple pattern recognition and machine learning techniques for classifying physical activities from sensor-based features.

The project uses two datasets: the HAR Smartphone dataset and the WISDM dataset. It includes preprocessing, dimensionality reduction, feature selection, Bayesian classification, standard machine learning classifiers, confusion matrix analysis, and stability analysis across repeated train/test partitions.

## Project Goals

- Classify human activities using sensor-based features
- Compare multiple supervised classification methods
- Analyze the impact of normalization, PCA, and feature selection
- Evaluate Bayesian and non-Bayesian classifiers
- Study model stability using repeated random partitions
- Compare performance across HAR Smartphone and WISDM datasets

## Datasets

### HAR Smartphone Dataset
The HAR Smartphone dataset contains sensor-based activity recognition features collected from smartphone sensors.

### WISDM Dataset
The WISDM dataset contains raw accelerometer and gyroscope sensor data from phone and wearable devices. Window-based statistical features are extracted from raw sensor readings.

## Methodology

### Preprocessing
- Train/validation/test split
- Z-score normalization
- Min-max normalization
- Missing value handling
- Feature extraction for WISDM raw sensor windows

### Feature Engineering
- Principal Component Analysis
- Sequential Forward Feature Selection
- Window-based statistical features for WISDM:
  - mean
  - standard deviation
  - min/max
  - median
  - energy
  - interquartile range
  - magnitude-based features

### Classifiers

#### Standard Machine Learning Models
- k-Nearest Neighbors
- Support Vector Machine
- Random Forest

#### Bayesian Classifiers
- MAP classifier with multivariate Gaussian class-conditional densities
- Gaussian Naive Bayes
- KDE / Parzen-window MAP classifier

## Evaluation Metrics

The models are evaluated using:

- Validation accuracy
- Test accuracy
- Confusion matrices
- Classification reports
- Most frequent class confusions
- Mean error rate across repeated runs
- Variance of error rate across repeated runs

## Results

The project compares classifier performance across:

- HAR Smartphone vs. WISDM datasets
- Z-score vs. min-max normalization
- Original features vs. PCA features vs. selected features
- Standard classifiers vs. Bayesian classifiers
