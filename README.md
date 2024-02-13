# Quantifying the Resolution of a Template after Image Registration


This is the official repository for

**Quantifying the Resolution of a Template after Image Registration.**   
Matthias Glock, Thomas Hotz.  
(submitted to the IEEE for possible publication)


## Abstract

In many image processing applications (e.g. computational anatomy) a groupwise registration is performed on a sample of images and a template image is simultaneously generated. From the template alone it is in general unclear to which extent the registered images are still misaligned, which means that some regions of the template represent the structural features in the sample images less reliably than others. In a sense, the template exhibits a lower resolution there. Guided by characteristic examples of misaligned image features in one dimension, we develop a visual measure to quantify the resolution at each location of a template which is based on the observation that misalignments between the registered sample images are reduced by smoothing with the strength of the smoothing being related to the magnitude of the misalignment. Finally the resulting resolution measure is applied to example datasets in two and three dimensions.

## Running our code

Project Organization:

    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── requirements.txt
    ├── data                <- created by `make template_*`
    ├── plots               <- created by `make visualization_*`
    └── src
        ├── analysis        <- scripts to compute resolution measure for datasets
        ├── data            <- scripts for data download and normalization
        ├── templates       <- scripts to create templates for datasets
        ├── tools           <- tools for image registration and resolution measure
        ├── visualization   <- scripts to create plots for datasets
        └── config.py       <- set global parameters

--------

usage:

```bash
make all
make all_mnist
make all_nfbs
```