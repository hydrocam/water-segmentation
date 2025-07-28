# Water Segmentation

This repository provides a complete pipeline for training, evaluating, and testing SAM-based (Segment Anything Model and MobileSAM) and Segformer models for river water segmentation using RGB imagery. It includes code for model fine-tuning, inference, accuracy evaluation, and edge deployment tests.

## Repository Structure

├── data_analysis/ # Notebooks and files for site specific machine learning analysis for predicting Water Stage and Discharge from pixel count values

├── edge_test/ # Scripts to test trained models on edge devices (Jetson Orin, Jetson Orin Nano, LattePanda Sigma and Raspberry Pi 5)

├── train_SAM.ipynb # Fine-tuning and inference notebook for the original SAM model

├── train_mobileSAM.ipynb # Fine-tuning and inference notebook for MobileSAM (optimized for edge)

├── .gitattributes # Git LFS tracking configuration for large models

├── LICENSE # MIT License

└── README.md # Project overview and usage instructions

## Features

- Fine-tunes SAM, MobileSAM and segformer for water segmentation
- Works with arbitrary image sizes
- Calculate Pixel value counts from the segmented images
- Visualizes overlayed predictions and ground truth with color-coded evaluation
- Machine learning analysis for predicting water stage and discharge
- Compatible with Jetson Nano, Orin, Raspberry Pi, and LattePanda devices

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.13
- OpenCV
- Matplotlib
- [Git LFS](https://git-lfs.com/)

> Make sure to install Git LFS and run `git lfs install` before cloning this repo if you want to use the trained models.

## Training and Inference
- To train or fine-tune the original SAM model:
Open and run train_SAM.ipynb

- To train or fine-tune MobileSAM (recommended for edge):
Open and run train_mobileSAM.ipynb

The scripts support:

- Custom image sizes
- Binary mask supervision
- GPU or CPU inference

## Edge Testing
Inside edge_test/, you'll find inference, power and performance logging scripts as well as pretrained models tailored for:

- Jetson Nano
- Jetson Orin Nano
- Raspberry Pi 5
- LattePanda Sigma

## Evaluation and Analysis
data_analysis/ contains scripts for:

- Visualizing predicted masks with error codes (TP, FP, FN)
- Time series comparisons
- Statistical metrics such as IoU, F1-score, etc.
- Temporal allignment between ground truth data and pixel value counts from segmentation results
- Machine Learning analysis for predicting Water Stage and Discharge from pixel values

## Funding and Acknowledgments
This research was supported by the Cooperative Institute for Research to Operations in Hydrology (CIROH) with joint funding under award NA22NWS4320003 from the NOAA Cooperative Institute Program and the U.S. Geological Survey. The statements, findings, conclusions, and recommendations are those of the author(s) and do not necessarily reflect the opinions of NOAA or USGS. Utah State University is a founding member of CIROH and receives funding under subaward from the University of Alabama. Additional funding and support have been provided by the Utah Water Research laboratory at Utah State University.

Training codes of SAM and mobileSAM build on the work of:
- Armin Moghimi, whose modifications made it possible to fine-tune SAM on arbitrary image sizes without resizing to 512×512, greatly improving flexibility for water segmentation and beyond.
- Alexandre Bonnet and Piotr Skalski, for their excellent tutorials and open-source contributions to SAM fine-tuning: https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/
