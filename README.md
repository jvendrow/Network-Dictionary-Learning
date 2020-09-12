<p align="center">
<img width="600" src="https://github.com/jvendrow/Network-Dictionary-Learning/blob/master/NDL_logo.png" alt="logo">
</p>


# Network-Dictionary-Learning
Learning from and reconstructing networks using Nonnegative Matrix Factorization

## Installation

To install the Network Dictionary Learning package, run this command in your terminal:

```bash
$ pip install ndl
```

This is the preferred method to install Network Dictionary Learning. If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## Usage

Our package lies on the backbone of the NNetwork class (see https://github.com/HanbaekLyu/NNetwork). 

```python
>>> from NNetwork import Wtd_NNetwork
>>> G = Wtd_NNetwork()
>>> G.load_edges("Data/example.txt", delimiter=",")
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
>>> NDL.display_dict(save_file="Dictionaries/example_dict.npy")
```

Replace the dictionary with a pre-trained dictionary and/or replace the network:
```python
>>> NDL.set_dict(W)
>>> NDL.set_network(G)
```
#### Reconstructing a Network

```python
>>> G_recons = NDL.reconstruct(recon_iters=10000)
```


The NetDictLearner class provices the base code to perform network dictionary learning and network reconstruction, but we also provide a series of helper fuctions to use alongside the NetDictLearner class to assist on performing tasks related to Network Dictionary Learning and evaluate performance. 

#### Measure Accuracy of Reconstruction (Jaccard)

```python
>>> from ndl import Utils
>>> Utils.jaccard(G, G_recons)
0.92475345
```

#### Network Denoising Application

To add positive corruption (overlaying edges) or negative corruption (deleting edges) from a networks:
```python
>>> len(G.edges())
1000
>>> G_add_50 = Utils.corrupt(G, p=0.5, noise_type="ER")
>>>len(G_add_50.edges())
1500
>>> G_remove_10 = Utils.corrupt(G, p=-0.1)
>>>len(G_remove_10.edges())
900
```

To measure the AUC of network denoising with positive (or negative) noise:
```python
>>> G_corrupt = Utils.corrupt(G, p=0.5, noise_type="ER")
>>> NDL_corrupt = NetDictLearner(G=G_corrupt)
>>> NDL.train_dict()
>>> G_corrupt_recons = NDL.reconstruct(recon_iters=10000)
>>> Utils.auc(original=G, corrupt=G_corrupt, corrupt_recons=G_corrupt_recons, type="positive")
0.864578
```



