# hippodeep
Brain Hippocampus Segmentation

This program segments the Hippocampus of raw brain T1 images in a few seconds.

![blink_rotated](https://user-images.githubusercontent.com/590921/75311442-1a705a00-589a-11ea-9cb6-d889fb226516.gif)

It relies on a Convolutional Neural Network pre-trained on thousands of images from multiple large cohorts, and is therefore quite robust to subject- and MR-contrast variation.
For more details on its creation, refer the corresponding manuscript at http://dx.doi.org/10.1016/j.media.2017.11.004

This official hippodeep version is a modern PyTorch port of the original Theano version that is technically obsolete. While the hippocampal segmentation model is exactly the same as described in the paper, the pre- and post-processing steps had been improved, and thus, results may differ sligthly. The deprecated theano repo is still available at https://github.com/bthyreau/hippodeep


## Requirement

This program requires Python 3 with the PyTorch library.

No GPU is required.

It should work on most distro and platform that supports pytorch, though mainly tested on Linux CentOS 6.x ~ 8.x, Ubuntu 18.04 ~ 22.04 and MacOS 10.13 ~ 12, using PyTorch versions from 1.0.0 to 2.6.0. 

## Installation and Usage

Just clone or download this repository.

If you have the uv packaging tool ( https://docs.astral.sh/uv/ ), you can do 

`uv run hippodeep.py example_brain_t1.nii.gz`

which should take care of downloading the dependencies in the first run. 

You can optionally call the deepseg1.sh script or symlink it somewhere in your $PATH

Otherwise, you need to configure a python 3 environment on your machine: In addition to PyTorch, the code requires scipy and nibabel. A possible way to install python from scratch is to use Anaconda (anaconda.com) to create an environment, then
 - install scipy (`conda install scipy` or `pip install scipy`) and  nibabel (`pip install nibabel`)
 - get pytorch for python from `https://pytorch.org/get-started/locally/`. CUDA is not necessary.
 - Then, to use the program, call it with `python hippodeep.py example_brain_t1.nii.gz` , possibly changing 'python' to 'python3' depending on your exact setup.

## Results

To process multiple subjects, pass them as multiple arguments. e.g:

`deepseg1.sh subject_*.nii.gz`. (or equivalent with 'uv run')

The resulting segmentations should be stored as `example_brain_t1_mask_L.nii.gz` (or R for right) and `example_brain_t1_brain_mask.nii.gz`.  The mask volumes (in mm^3) are stored in a csv file named `example_brain_t1_hippoLR_volumes.csv`.  If more than one input was specified, a summary table named `all_subjects_hippo_report.csv` is created.

## MRA image
Hippodeep has a specific model to process MR Angiography images. Simply adds the "-mra" option on the command line.

Optionally, add the "-mra-head" option to limit the MRA analysis to the hippocampal heads. This may be useful if the hippocampus tail is known to be outside the field of view.

Keep in mind that the eTIV brain size estimate would likely be inaccurate in case of an incomplete brain coverage



## License
Hippodeep is MIT License

For the MRA model: MIT + Attribution clause license
