<p align="center">
<img width="600" src="https://github.com/jvendrow/Network-Dictionary-Learning/blob/master/NDL_logo.png" alt="logo">
</p>


# Network-Dictionary-Learning
Learning from and reconstructing networks using MCMC motif sampling and Nonnegative Matrix Factorization.

[![PyPI Version](https://img.shields.io/pypi/v/ndlearn.svg)](https://pypi.org/project/ndlearn/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/ndlearn.svg)](https://pypi.org/project/ndlearn/)
[![Downloads](https://pepy.tech/badge/ndlearn)](https://pepy.tech/project/ndlearn)

By Hanbaek Lyu, Joshua Vendrow, and Yacoub Kureh.

## Installation

To install the Network Dictionary Learning package, run this command in your terminal:

```bash
$ pip install ndlearn
```

This is the preferred method to install Network Dictionary Learning. If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## Usage

Our package lies on the backbone of the NNetwork class (see https://github.com/HanbaekLyu/NNetwork). 

```python
>>> from ndl import Wtd_NNetwork
>>> G = Wtd_NNetwork()
>>> G.load_add_edges_wtd("example.txt", use_genfromtxt=True, delimiter=",")
```
#### Learning a Dictionary

```python
>>> from ndl import NetDictLearner
>>> NDL = NetDictLearner(G=G, n_components=25, k=21)
>>> NDL.train_dict()
>>> W = NDL.get_dictionary()
```

Display and save the learned dictionary:
```python
>>> NDL.display_dict(path="example_dict.npy")
```

Replace the dictionary with a pre-trained dictionary and/or replace the network:
```python
>>> NDL.set_dict(W)
>>> NDL.set_network(G)
```
#### Reconstructing a Network

```python
>>> G_recons = NDL.reconstruct(recons_iter=10000)
```


The NetDictLearner class provices the base code to perform network dictionary learning and network reconstruction, but we also provide a series of helper fuctions to use alongside the NetDictLearner class to assist on performing tasks related to Network Dictionary Learning and evaluate performance. 

#### Measure Accuracy of Reconstruction (Jaccard)

```python
>>> from ndl import utils
>>> utils.recons_accuracy(G, G_recons)
0.92475345
```

#### Network Denoising Application

To add positive corruption (overlaying edges) or negative corruption (deleting edges) from a networks:
```python
>>> len(G.edges())
1000
>>> G_add = utils.corrupt(G, p=0.1, noise_type="ER")
>>> G_remove_10 = utils.corrupt(G, p=0.1, noise_type="negative")
>>>len(G_remove_10.edges())
900
```

To measure the AUC of network denoising with positive (or negative) noise:
```python
>>> G_corrupt = utils.corrupt(G, p=0.1, noise_type="ER")
>>> NDL_corrupt = NetDictLearner(G=G_corrupt)
>>> NDL_corrupt.train_dict()
>>> G_corrupt_recons = NDL_corrupt.reconstruct(recons_iter=10000)
>>> utils.auc(original=G, corrupt=G_corrupt, corrupt_recons=G_corrupt_recons, type="positive")
0.864578
```



