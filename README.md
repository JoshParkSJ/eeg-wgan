# Improving EEG Signal Availability using Deep Learning and Generative Adversarial Networks

<p align="center">
  <img src='data/cover.png'  align="center" width='100%'>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

## Project Description
This repository contains the implementation of the Wasserstein GAN (WGAN) model with gradient penalty developed in our [research paper](). The focus of our research was to generate synthetic 1D time-series Electroencephalogram (EEG) signals that mimic real EEG data.

Electroencephalography (EEG) plays a vital role in recording brain activities and is integral to the development of brain-computer interface (BCI) technologies. However, the limited availability and high variability of EEG signals present substantial challenges in creating reliable BCIs. To address this challenge, we propose a practical solution drawing on the latest developments in deep learning and Wasserstein Generative Adversarial Network (WGAN). The WGAN was trained on the BCI2000 dataset, consisting of around 1500 EEG recordings and 64 channels from 45 individuals. The generated EEG signals were evaluated via three classifiers yielding improved average accuracies. The quality of generated signals measured using Frechet Inception Distance (FID) yielded scores of 1.345 and 11.565 for eyes open and closed respectively. Even without a spectral or spatial loss term, our WGAN model was able to emulate the spectral and spatial properties of the EEG training data. The WGAN generated data mirrored the dominant alpha activity during closed-eye resting and high delta waves in the training data in its topographic map and power spectral density (PSD) plot. Our research testifies to the potential of WGANs in addressing the limited EEG data issue for BCI development by enhancing a small dataset to improve classifier generalizability.

Model architecture:
<p align="center">
  <img src='data/architecture.png'  align="center" width='100%'>
</p>

## Quick Start
To train the model and reproduce the results from our paper, simply run wgan-gp.ipynb file in a notebook environment such as Jupyter notebook

## Results
Detailed results and comparisons are available in our paper.

<p align="center">
  <img src='data/psd.png'  align="center" width='100%'>
</p>

<p align="center">
  <img src='data/topomap1.png'  align="center" width='100%'>
  <img src='data/topomap2.png'  align="center" width='100%'>
</p>

<p align="center">
  <img src='data/accuracy_fid.png'  align="center" width='100%'>
</p>

## Requirements
Install the necessary libraries by running:

```
pip install -r requirements.txt
```

## Citation
If you find our work useful, please cite our paper: [Improving EEG Signal Availability using Deep Learning and Generative Adversarial Networks]():

## Contact
Feel free to contact joshparksj(at)gmail(dot)com for any question or suggestion.
