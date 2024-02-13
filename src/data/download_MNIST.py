"""
Download MNIST dataset of 2d images of handwritten digits
"""

from torchvision import datasets

data_path = 'data/raw/' # relative to Makefile

mnist = datasets.MNIST(data_path, download=True)
