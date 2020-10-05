import sys
sys.path.insert(1, '../')

from ndl import Wtd_NNetwork
from ndl import NetDictLearner
#import ndl.utils as utils
from ndl import utils


def learn_and_reconstruct():

    print("Learning and reconstructing network...")

    G = Wtd_NNetwork()
    G.load_add_wtd_edges("Caltech36.txt", use_genfromtxt=True)

    NDL = NetDictLearner(G=G, n_components=25, MCMC_iterations=5)
    NDL.train_dict(verbose=True)
    G_recons = NDL.reconstruct(recons_iter=10000)

    acc = utils.recons_accuracy(G, G_recons)
    print(acc)

def learn_and_plot():

    print("Learning and reconstructing network...")

    G = Wtd_NNetwork()
    G.load_add_wtd_edges("Caltech36.txt", use_genfromtxt=True)

    NDL = NetDictLearner(G=G, n_components=25, MCMC_iterations=10)
    NDL.train_dict(verbose=True)

    NDL.display_dict(title="Network Dictionary")


def additive_corruption():

    print("Denoising for additive corruption...")

    G = Wtd_NNetwork()
    G.load_add_wtd_edges("Caltech36.txt", use_genfromtxt=True)

    G_corrupt, edges_changed = utils.corrupt(G, parameter=0.01, noise_type="ER")

    NDL_corrupt = NetDictLearner(G=G_corrupt, n_components=25, MCMC_iterations=40)

    NDL_corrupt.train_dict()
    G_corrupt_recons = NDL_corrupt.reconstruct(recons_iter=10000, return_weighted=True, omit_chain_edges=True, omit_folded_edges=False)
    auc, fpr, tpr = utils.auc_roc(G, G_corrupt, G_corrupt_recons, noise_type="positive")
    print(auc)

def subtractive_corruption():

    print("Denoising for subtractive corruption...")
    G = Wtd_NNetwork()
    G.load_add_wtd_edges("Caltech36.txt", use_genfromtxt=True)

    G_corrupt, edges_changed = utils.corrupt(G, parameter=0.1, noise_type="negative")
    print(len(G.get_edges()))
    print(len(G_corrupt.get_edges()))

    NDL_corrupt = NetDictLearner(G=G_corrupt, n_components=25, MCMC_iterations=50)

    NDL_corrupt.train_dict()
    G_corrupt_recons = NDL_corrupt.reconstruct(recons_iter=10000, return_weighted=True, omit_chain_edges=True, omit_folded_edges=False)
    auc, fpr, tpr = utils.auc_roc(G, G_corrupt, G_corrupt_recons, noise_type="negative")
    print(auc)


#learn_and_reconstruct()
learn_and_plot()
#additive_corruption()
#subtractive_corruption()
