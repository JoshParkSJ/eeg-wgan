# Improving EEG Signal Availability using Deep Learning and Generative Adversarial Networks

<p align="center">
  <img src='data/cover.png'  align="center" width='100%'>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

## Project Description
This repository contains the implementation of the Wasserstein GAN (WGAN) model with gradient penalty developed in our [research paper](). The focus of our research was to generate synthetic 1-D time-series Electroencephalogram (EEG) signals that mimic real EEG data.

The implemented WGAN-GP model takes as input a tensor of shape (batch size, 64, 3152). Here, '64' corresponds to the 64 channels of the EEG, and '3152' is the sequence length of the time-series for each channel. After generating synthetic EEG data, we assess the quality of these generated signals by using the classification accuracy of three classifiers: Fully Connected Neural Network (FNN), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN). 

Model architecture:
<p align="center">
  <img src='data/architecture.png'  align="center" width='100%'>
</p>


## Requirements
Install the necessary libraries by running:

```
pip install -r requirements.txt
```

## Quick Start
To train the model and reproduce the results from our paper, simply run wgan-gp.ipynb file in a notebook environment such as Jupyter notebook

## Results
Detailed results and comparisons are available in our paper.

<p align="center">
  <img src='data/accuracy_fid.png'  align="center" width='100%'>
</p>

<p align="center">
  <img src='data/topomap1.png'  align="center" width='100%'>
  <img src='data/topomap2.png'  align="center" width='100%'>
</p>

<p align="center">
  <img src='data/psd.png'  align="center" width='100%'>
</p>

## Citation
If you find our work useful, please cite our paper: [Improving EEG Signal Availability using Deep Learning and Generative Adversarial Networks]():

## Contact
Feel free to contact joshparksj(at)gmail(dot)com for any question or suggestion.
