# hippodeep
Brain Hippocampus Segmentation

This program can quickly segment (<2min) the Hippocampus of raw brain T1 images.


![screenshot](blink.gif?raw=True)

It relies on a Convolutional Neural Network pre-trained on thousands of images from multiple large cohorts, and is therefore quite robust to subject- and MR-contrast variation.
For more details on how it has been created, refer to the corresponding manuscript at http://dx.doi.org/10.1016/j.media.2017.11.004

This version is a PyTorch port of the original Theano model. While the code segmentation model is exactly the same as described in the paper, some pre- and post-processing code may differ.

## Requirement

This program requires Python 3, with the PyTorch library, version > 1.0.0.

No GPU is required

Tested on Linux CentOS 6.x and 7.x, and MacOS X

## Installation

In addition to PyTorch, the code requires scipy and nibabel.

The simplest way to install from scratch is maybe to use a Anaconda environment, then
* install scipy (`conda install scipy` or `pip install scipy`) and  nibabel (`pip install nibabel`)
* get pytorch for python from `https://pytorch.org/get-started/locally/`. CUDA is not necessary.


## Usage:
To use the program, simply run:

`deepseg1.sh example_brain_t1.nii.gz`.

The resulting segmentation should be stored as `example_brain_t1_mask_L.nii.gz` (or R for right) and `example_brain_t1_brain_mask.nii.gz`.  The mask volumes (in mm^3) are stored in a csv file named `example_brain_t1_hippoLR_volumes.csv`.  If more than one input was specified, a summary table named `all_subjects_hippo_report.csv` is created.
