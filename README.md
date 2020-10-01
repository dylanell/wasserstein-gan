# wasserstein-gan

Implementation of the [Improved Wasserstein Generative Adversarial Network (GAN)](https://arxiv.org/pdf/1704.00028.pdf) on the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

| ![](docs/generator_train.gif) |
| :-: |
| *Generator outputs during training.* |

### Environment:

Python 3.7.4

### Python Packages:

jupyterlab
torch
torchvision
numpy
imageio

### Image Dataset Format:

This project assumes you have the MNIST dataset pre-configured locally on your machine in the format described below. My [dataset-helpers]() Github project also contains a Python script that performs this local configuration automatically.

The MNIST dataset consists of images of written numbers (0-9) with corresponding labels. The dataset can be accessed a number of ways using Python packages (`mnist`, `torchvision`, `tensorflow_datasets`, etc.), or it can be downloaded directly from the [MNIST homepage](http://yann.lecun.com/exdb/mnist/). In order to develop image-based data pipelines in a standard way, we organize the MNIST dataset into training/testing directories of raw image files (`png` or `jpg`) accompanied by a `csv` file listing one-to-one correspondences between the image file names and their label. This "generic image dataset format" is summarized by the directory tree structure below.

```
dataset_directory/
|__ train_labels.csv
|__ test_labels.csv
|__ train/
|   |__ train_image_01.png
|   |__ train_image_02.png
|   |__ ...
|__ test/
|   |__ test_image_01.png
|   |__ test_image_02.png
|   |__ ...   
```

Each labels `csv` file has the format:

```
Filename, Label
train_image_01.png, 4
train_image_02.png, 7
...
```

If you would like to re-use the code here to work with other image datasets, just format any new image dataset to follow the outline above and be sure to edit corresponding hyperparameters in the `config.yaml` file.

### Training:

Training hyperparameters are pulled from the `config.yaml` file and can be changed by editing the file contents.

Train the Wasserstein GAN by running:

```
$ python train.py
```

By default,

### Jupyter Notebook:

This project is accompanied by a Jupyter notebook that explains the theory behind Wasserstein GANs as well as some details on how to reload model files and create instances of trained critic and generator networks.

Run the following command to start the Jupyter notebook server in your browser:

```
$ jupyter-notebook notebook.ipynb
```

### References:

1. Wasserstein GAN:

 - https://arxiv.org/pdf/1704.00028.pdf
