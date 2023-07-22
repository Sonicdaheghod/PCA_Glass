# Principal Component Analysis - Glass Dataset
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)
* [Technologies](#technologies)
* [Setup](#setup)
* [Using the Program](#Using-the-Program)

## Purpose of Program
The purpose of this program was to practice using Principal Component Analysis (PCA) on a variable in the Glass dataset.

## Technologies
Languages/ Technologies used:

* Jupyter Notebook

* Python3

## Setup

Import the following modules and libraries:

``` 
import pandas as pd
from sklearn.decomposition import PCA
from pydataset import data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
``` 
## Using the Program

1. Import dataset and use fit.transform() to allow PCA to use such data. This turns the dataset into an array.
<img width="455" alt="image" src="https://github.com/Sonicdaheghod/PCA_Glass/assets/68253811/4b56aabf-93a7-46b3-9c7d-bd0b7adebe9a">

2. Determine how much components influence variance in dataset
<img width="610" alt="image" src="https://github.com/Sonicdaheghod/PCA_Glass/assets/68253811/3c3e80e7-d817-4085-93b1-9858d1093def">

3. Plot the PCA graph for a variable of our choice from the dataset
<img width="350" alt="image" src="https://github.com/Sonicdaheghod/PCA_Glass/assets/68253811/fbc95e98-8114-41e7-9042-3d43924e6955">

### Credits
This project was inspired by [Educational Research Techniques](https://youtu.be/yDUCqI4zBlM)

Dataset from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/42/glass+identification)

