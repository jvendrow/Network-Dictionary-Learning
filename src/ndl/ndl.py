# from utils.onmf.onmf import Online_NMF
from ndl.ontf import Online_NTF

from ndl.NNetwork import NNetwork, Wtd_NNetwork
import numpy as np
import itertools
from time import time
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt
import networkx as nx
import os
import psutil
import matplotlib.gridspec as gridspec
from time import sleep
import sys
from sklearn.metrics import roc_curve
from scipy.spatial import ConvexHull
import random
from tqdm import trange

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

DEBUG = False


class NetDictLearner():
    def __init__(self,
                 G,
                 n_components=100,
                 MCMC_iterations=500,
                 sub_iterations=100,
                 loc_avg_depth=1,
                 sample_size=1000,
                 batch_size=10,
                 k1=0,
                 k2=21,
                 patches_file='',
                 alpha=None,
                 is_glauber_dict=True,
                 is_glauber_recons=True,
                 Pivot_exact_MH_rule=False,
                 ONMF_subsample=True,
                 if_wtd_network=False,
                 if_tensor_ntwk=False,
                 omit_folded_edges=False):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.G = G  ### Full netowrk -- could have positive or negagtive edge weights (as a NNetwork or Wtd_NNetwork class)
        if if_tensor_ntwk:
            self.G.set_clrd_edges_signs()
            ### Each edge with weight w is assigned with tensor weight [+(w), -(w)] stored in the field colored_edge_weight

        self.n_components = n_components
        self.MCMC_iterations = MCMC_iterations
        self.sub_iterations = sub_iterations
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.loc_avg_depth = loc_avg_depth
        self.k1 = k1
        self.k2 = k2
        self.patches_file = patches_file
        self.if_tensor_ntwk = if_tensor_ntwk  # if True, input data is a 3d array
        self.omit_folded_edges = omit_folded_edges  # if True, get induced k by k patch without off-chain edges appearing
        ### due to folding of the underlying motif (e.g., completely folded k-chain --> no checkerboard pattern)
        self.W = np.random.rand((k1 + k2 + 1) ** 2, n_components)
        if if_tensor_ntwk:
            self.W = np.random.rand(G.color_dim * (k1 + k2 + 1) ** 2, n_components)

        self.code = np.zeros(shape=(n_components, sample_size))
        self.code_recons = np.zeros(shape=(n_components, sample_size))
        self.alpha = alpha
        self.is_glauber_dict = is_glauber_dict  ### if false, use pivot chain for dictionary learning
        self.is_glauber_recons = is_glauber_recons  ### if false, use pivot chain for reconstruction
        self.Pivot_exact_MH_rule = Pivot_exact_MH_rule
        self.edges_deleted = []
        self.ONMF_subsample = ONMF_subsample
        self.result_dict = {}
        self.if_wtd_network = if_wtd_network

    def list_intersection(self, lst1, lst2):
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    def path_adj(self, k1, k2):
        # generates adjacency matrix for the path motif of k1 left nodes and k2 right nodes
        if k1 == 0 or k2 == 0:
            k3 = max(k1, k2)
            A = np.eye(k3 + 1, k=1, dtype=int)
        else:
            A = np.eye(k1 + k2 + 1, k=1, dtype=int)
            A[k1, k1 + 1] = 0
            A[0, k1 + 1] = 1
        return A

    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def find_parent(self, B, i):
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # Find the index of the unique parent of i in B
        j = self.indices(B[:, i], lambda x: x == 1)  # indices of all neighbors of i in B
        # (!!! Also finds self-loop)
        return min(j)

    def tree_sample(self, B, x):
        # A = N by N matrix giving edge weights on networks
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # samples a tree B from a given pivot x as the first node
        N = self.G
        k = np.shape(B)[0]
        emb = np.array([x], dtype='<U32')  # initialize path embedding

        if sum(sum(B)) == 0:  # B is a set of isolated nodes
            y = np.random.randint(N.num_nodes(), size=(1, k - 1))
            y = y[0]  # juts to make it an array
            emb = np.hstack((emb, y))
        else:
            for i in np.arange(1, k):
                j = self.find_parent(B, i)
                nbs_j = np.asarray(list(N.neighbors(emb[j])))
                if len(nbs_j) > 0:
                    y = np.random.choice(nbs_j)
                else:
                    y = emb[j]
                    print('tree_sample_failed:isolated')
                emb = np.hstack((emb, y))
        # print('emb', emb)
        return emb

    def glauber_gen_update(self, B, emb):
        N = self.G
        k = np.shape(B)[0]

        if k == 1:

            emb[0] = self.walk(emb[0], 1)
            # print('Glauber chain updated via RW')
        else:
            j = np.random.choice(np.arange(0, k))  # choose a random node to update
            nbh_in = self.indices(B[:, j], lambda x: x == 1)  # indices of nbs of j in B
            nbh_out = self.indices(B[j, :], lambda x: x == 1)  # indices of nbs of j in B

            # build distribution for resampling emb[j] and resample emb[j]
            time_a = time()
            cmn_nbs = N.nodes(is_set=True)
            time_1 = time()
            time_neighbor = 0

            if not self.if_wtd_network:
                for r in nbh_in:
                    time_neighb = time()
                    nbs_r = N.neighbors(emb[r])
                    end_neighb = time()
                    time_neighbor += end_neighb - time_neighb
                    if len(cmn_nbs) == 0:
                        cmn_nbs = nbs_r
                    else:
                        cmn_nbs = cmn_nbs & nbs_r

                for r in nbh_out:
                    nbs_r = N.neighbors(emb[r])
                    if len(cmn_nbs) == 0:
                        cmn_nbs = nbs_r
                    else:
                        cmn_nbs = cmn_nbs & nbs_r

                cmn_nbs = list(cmn_nbs)
                if len(cmn_nbs) > 0:
                    y = np.random.choice(np.asarray(cmn_nbs))
                    emb[j] = y
                else:
                    emb[j] = np.random.choice(N.nodes())
                    print('Glauber move rejected')  # Won't happen once a valid embedding is established

            else:  ### Now need to use edge weights for Glauber chain update as well
                # build distribution for resampling emb[j] and resample emb[j]
                cmn_nbs = [i for i in N.nodes()]
                for r in nbh_in:
                    # print('emb[r]:',emb[r])
                    nbs_r = [i for i in N.neighbors(emb[r])]
                    cmn_nbs = list(set(cmn_nbs) & set(nbs_r))
                for r in nbh_out:
                    nbs_r = [i for i in N.neighbors(emb[r])]
                    cmn_nbs = list(set(cmn_nbs) & set(nbs_r))

                if len(cmn_nbs) > 0:

                    ### Compute distribution on cmn_nbs
                    dist = np.ones(len(cmn_nbs))
                    for v in np.arange(len(cmn_nbs)):
                        for r in nbh_in:
                            dist[v] = dist[v] * abs(N.get_edge_weight(emb[r], cmn_nbs[v]))
                        for r in nbh_out:
                            dist[v] = dist[v] * abs(N.get_edge_weight(cmn_nbs[v], emb[r]))
                            ### As of now (05/15/2020) Wtd_NNetwork class has weighted edges without orientation,
                            ### so there is no distinction between in- and out-neighbors
                            ### Use abs since edge weights could be negative
                    dist = dist / np.sum(dist)
                    # idx = np.random.choice(np.arange(len(cmn_nbs)), p=dist)
                    ### 7/25/2020: If just use np.random.choice(cmn_nbs, p=dist), then it somehow only selects first six-digit and causes key erros
                    idx = np.random.choice(np.arange(len(cmn_nbs)), p=dist)

                    emb[j] = cmn_nbs[idx]
                    # if len(emb[j]) == 7:
                    #    print('y len 7')

                else:
                    emb[j] = np.random.choice(np.asarray([i for i in self.G.nodes]))
                    print('Glauber move rejected')  # Won't happen once valid embedding is established

        return emb

    def Pivot_update(self, emb):
        # G = underlying simple graph
        # emb = current embedding of a path in the network
        # k1 = length of left side chain from pivot
        # updates the current embedding using pivot rule

        k1 = self.k1
        k2 = self.k2
        x0 = emb[0]  # current location of pivot
        x0 = self.RW_update(x0, Pivot_exact_MH_rule=self.Pivot_exact_MH_rule)  # new location of the pivot
        B = self.path_adj(k1, k2)
        #  emb_new = self.Path_sample_gen_position(x0, k1, k2)  # new path embedding

        emb_new = self.tree_sample(B, x0)  # new path embedding
        return emb_new

    def RW_update(self, x, Pivot_exact_MH_rule=False):
        # G = simple graph
        # x = RW is currently at site x
        # stationary distribution = uniform
        # Pivot_exact_MH_rule = True --> RW is updated so that the Pivot chain is sampled from the exact conditional distribution
        # otherwise the pivot of the Pivot chain performs random walk with uniform distribution as its stationary distribution

        N = self.G
        length = self.k1 + self.k2  # number of edges in the chain motif
        nbs_x = np.asarray(list(N.neighbors(x)))  # array of neighbors of x in G

        if len(nbs_x) > 0:  # this holds if the current location x of pivot is not isolated
            y = np.random.choice(nbs_x)  # choose a uniform element in nbs_x
            # x ---> y move generated
            # Use MH-rule to accept or reject the move
            # stationary distribution = Uniform(nodes)
            # Use another coin flip (not mess with the kernel) to keep the computation local and fast
            nbs_y = np.asarray(list(N.neighbors(y)))
            prob_accept = min(1, len(nbs_x) / len(nbs_y))

            if Pivot_exact_MH_rule:
                a = N.count_k_step_walks(y, radius=length)
                b = N.count_k_step_walks(x, radius=length)
                print('!!!! MHrule a', a)
                print('!!!! MHrule b', b)

                prob_accept = min(1, a * len(nbs_x) / b * len(nbs_y))

            if np.random.rand() > prob_accept:
                y = x  # move to y rejected

        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.asarray(N.nodes()))
        return y

    def update_hom_get_meso_patch(self,
                                  B,
                                  emb,
                                  iterations=1,
                                  is_glauber=True,
                                  verbose=0,
                                  omit_folded_edges=False):
        # computes a mesoscale patch of the input network G using Glauber chain to evolve embedding of B in to G
        # also update the homomorphism once
        # iterations = number of iteration
        # underlying graph = specified by A
        # B = adjacency matrix of rooted tree motif
        start = time()

        N = self.G
        emb2 = emb
        k = B.shape[0]
        #  x0 = np.random.choice(np.arange(0, N))  # random initial location of RW
        #  emb2 = self.tree_sample(B, x0)  # initial sampling of path embedding

        hom_mx2 = np.zeros([k, k])
        if self.if_tensor_ntwk:
            hom_mx2 = np.zeros([k, k, N.color_dim])

        nofolding_ind_mx = np.zeros([k, k])

        for i in range(iterations):
            start_iter = time()
            if is_glauber:
                emb2 = self.glauber_gen_update(B, emb2)
            else:
                emb2 = self.Pivot_update(emb2)
            end_update = time()
            # start = time.time()

            ### Form induced graph = homomorphic copy of the motif given by emb2 (may have < k nodes)
            ### e.g., a k-chain can be embedded onto a K2, then H = K2.
            H = Wtd_NNetwork()
            for q in np.arange(k):
                for r in np.arange(k):
                    edge = [emb2[q], emb2[r]]  ### "edge" may repeat for distinct pairs of [q,r]
                    if B[q, r] > 0:  ### means [q,r] is an edge in the motif with adj mx B
                        H.add_edge(edge=edge, weight=1)

            if not self.if_tensor_ntwk:
                # full adjacency matrix or induced weight matrix over the path motif
                a2 = np.zeros([k, k])
                start_loop = time()
                for q in np.arange(k):
                    for r in np.arange(k):
                        if not self.if_wtd_network or N.has_edge(emb2[q], emb2[r]) == 0:
                            if not omit_folded_edges:
                                a2[q, r] = int(N.has_edge(emb2[q], emb2[r]))
                            elif not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
                                a2[q, r] = int(N.has_edge(emb2[q], emb2[r]))
                                nofolding_ind_mx[q, r] = 1

                        else:
                            if not omit_folded_edges:
                                a2[q, r] = N.get_edge_weight(emb2[q], emb2[r])
                            elif not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
                                a2[q, r] = N.get_edge_weight(emb2[q], emb2[r])
                                nofolding_ind_mx[q, r] = 1

                hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)

            else:  # full induced weight tensor over the path motif (each slice of colored edge gives a weight matrix)
                a2 = np.zeros([k, k, N.color_dim])
                start_loop = time()
                for q in np.arange(k):
                    for r in np.arange(k):
                        if N.has_edge(emb2[q], emb2[r]) == 0:
                            if not omit_folded_edges:
                                a2[q, r, :] = np.zeros(N.color_dim)
                            elif not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
                                a2[q, r, :] = np.zeros(N.color_dim)
                                nofolding_ind_mx[q, r] = 1
                        else:
                            if not omit_folded_edges:
                                a2[q, r, :] = N.get_colored_edge_weight(emb2[q], emb2[r])
                            elif not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
                                a2[q, r, :] = N.get_colored_edge_weight(emb2[q], emb2[r])
                                nofolding_ind_mx[q, r] = 1
                                # print('np.sum(a2[q, r, :])', np.sum(a2[q, r, :]))
                hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)

            if (verbose):
                print([int(i) for i in emb2])

        if omit_folded_edges:
            return hom_mx2, emb2, nofolding_ind_mx
        else:
            return hom_mx2, emb2

    def get_patches_glauber(self, B, emb):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        X = np.zeros((k ** 2, 1))
        if self.if_tensor_ntwk:
            X = np.zeros((k ** 2, self.G.color_dim, 1))

        for i in np.arange(self.sample_size):
            meso_patch = self.update_hom_get_meso_patch(B, emb,
                                                        iterations=1,
                                                        is_glauber=self.is_glauber_dict,  # k by k matrix
                                                        omit_folded_edges=self.omit_folded_edges)

            Y = meso_patch[0]
            emb = meso_patch[1]

            if not self.if_tensor_ntwk:
                Y = Y.reshape(k ** 2, -1)
            else:
                Y = Y.reshape(k ** 2, self.G.color_dim, -1)

            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=-1)  # x is class ndarray
        #  now X.shape = (k**2, sample_size) or (k**2, color_dim, sample_size)
        # print(X)
        return X, emb

    def get_single_patch_glauber(self, B, emb, omit_folded_edges=False):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        meso_patch = self.update_hom_get_meso_patch(B,
                                                    emb, iterations=1,
                                                    is_glauber=self.is_glauber_recons,
                                                    omit_folded_edges=omit_folded_edges)

        Y = meso_patch[0]
        emb = meso_patch[1]

        if not self.if_tensor_ntwk:
            X = Y.reshape(k ** 2, -1)
        else:
            X = Y.reshape(k ** 2, self.G.get_edge_color_dim(), -1)

        if not omit_folded_edges:
            return X, emb
        else:
            return X, emb, meso_patch[2]  # last output is the nofolding indicator mx

    def glauber_walk(self, x0, length, iters=1, verbose=0):

        N = self.G
        B = self.path_adj(0, length)
        # x0 = 2
        # x0 = np.random.choice(np.asarray([i for i in G]))
        emb = self.tree_sample(B, x0)
        k = B.shape[0]

        emb, _ = self.update_hom_get_meso_patch(B, emb, iterations=iters, verbose=0)

        return [int(i) for i in emb]

    def walk(self, node, iters=10):
        for i in range(iters):
            node = np.random.choice(self.G.neighbors(node))

        return node

    def train_dict(self, jump_every=20, update_dict_save=True, verbose=True):
        # emb = initial embedding of the motif into the network
        if(verbose):
            print('training dictionaries from patches...')
        '''
        Trains dictionary based on patches.
        '''

        G = self.G
        B = self.path_adj(self.k1, self.k2)
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb = self.tree_sample(B, x0)
        W = self.W
        errors = []
        code = self.code

        if(verbose):
            f = trange
        else:
            f = np.arange

        for t in f(self.MCMC_iterations):
            X, emb = self.get_patches_glauber(B, emb)
            # print('X.shape', X.shape)  ## X.shape = (k**2, sample_size)

            ### resample the embedding for faster mixing of the Glauber chain
            if t % jump_every == 0:
                x0 = np.random.choice(np.asarray([i for i in G.vertices]))
                emb = self.tree_sample(B, x0)

            if not self.if_tensor_ntwk:
                X = np.expand_dims(X, axis=1)  ### X.shape = (k**2, 1, sample_size)
            if t == 0:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      alpha=self.alpha,
                                      mode=2,
                                      learn_joint_dict=True,
                                      subsample=self.ONMF_subsample)  # max number of possible patches
                self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                self.H = code
            else:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=self.W,
                                      ini_A=self.At,
                                      ini_B=self.Bt,
                                      ini_C=self.Ct,
                                      alpha=self.alpha,
                                      history=self.ntf.history,
                                      subsample=self.ONMF_subsample,
                                      mode=2,
                                      learn_joint_dict=True)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                code += self.H
                error = np.trace(self.W @ self.At @ self.W.T) - 2 * np.trace(self.W @ self.Bt) + np.trace(self.Ct)
                errors.append(error)
        self.code = code
        if update_dict_save:
            self.result_dict.update({'Dictionary learned': self.W})
            self.result_dict.update({'Motif size': self.k2 + 1})
            self.result_dict.update({'Code learned': self.code})
            self.result_dict.update({'Code COV learned': self.At})
        # print(self.W)
        return self.W

    def display_dict(self,
                     title,
                     save_filename,
                     make_first_atom_2by2=False,
                     save_folder=None,
                     show_importance=False):
        #  display learned dictionary
        W = self.W
        n_components = W.shape[1]
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if rows ** 2 == n_components:
            cols = rows
        else:
            cols = rows + 1

        if save_folder is None:
            save_folder = "Network_dictionary"

        # cols=3
        # rows=6
        k = self.k1 + self.k2 + 1

        ### Use the code covariance matrix At to compute importance
        importance = np.sqrt(self.At.diagonal()) / sum(np.sqrt(self.At.diagonal()))
        # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
        idx = np.argsort(importance)
        idx = np.flip(idx)

        if make_first_atom_2by2:
            ### Make gridspec
            fig = plt.figure(figsize=(3, 6), constrained_layout=False)
            gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.1, hspace=0.1)

            for i in range(rows * cols - 3):
                if i == 0:
                    ax = fig.add_subplot(gs1[:2, :2])
                elif i < 2 * cols - 3:  ### first two rows of the dictionary plot
                    if i < cols - 1:
                        ax = fig.add_subplot(gs1[0, i + 1])
                    else:
                        ax = fig.add_subplot(gs1[1, i - (cols - 1) + 2])
                else:
                    i1 = i + 3
                    a = i1 // cols
                    b = i1 % cols
                    ax = fig.add_subplot(gs1[a, b])

                ax.imshow(self.W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                ax.set_xticks([])
                ax.set_yticks([])

            plt.suptitle(title)
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
            fig.savefig(save_folder + '/' + save_filename)
            # plt.show()

        else:
            if not self.if_tensor_ntwk:
                figsize = (5, 5)
                if show_importance:
                    figsize = (5, 6)

                fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,
                                        subplot_kw={'xticks': [], 'yticks': []})

                k = self.k1 + self.k2 + 1  # number of nodes in the motif F
                for ax, j in zip(axs.flat, range(n_components)):
                    ax.imshow(self.W.T[idx[j]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    if show_importance:
                        ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=13)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    # use gray_r to make black = 1 and white = 0

                plt.suptitle(title)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
                fig.savefig(save_folder + '/' + save_filename)
            else:
                W = W.reshape(k ** 2, self.G.color_dim, self.n_components)
                for c in range(self.G.color_dim):
                    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 5),
                                            subplot_kw={'xticks': [], 'yticks': []})

                    for ax, j in zip(axs.flat, range(n_components)):
                        ax.imshow(W[:, c, :].T[j].reshape(k, k), cmap="gray_r", interpolation='nearest')
                        # use gray_r to make black = 1 and white = 0
                        if show_importance:
                            ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=13)  # get the largest first
                            ax.xaxis.set_label_coords(0.5,
                                                      -0.05)  # adjust location of importance appearing beneath patches

                plt.suptitle(title)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
                fig.savefig(save_folder + '/' + save_filename + '_color_' + str(c))

        # plt.show()

    def display_dict_gridspec(self,
                              title,
                              save_filename,
                              save_folder):
        ### Initial setup
        W = self.W
        n_components = W.shape[1]
        nrows = np.round(np.sqrt(n_components))
        nrows = nrows.astype(int)
        if nrows ** 2 == n_components:
            ncols = nrows
        else:
            ncols = nrows + 1

        if save_folder is None:
            save_folder = "Network_dictionary"

        k = self.k1 + self.k2 + 1

        importance = np.sum(self.code, axis=1) / sum(sum(self.code))
        idx = np.argsort(importance)
        idx = np.flip(idx)

        ### Make gridspec
        fig1 = plt.figure(figsize=(6, 6), constrained_layout=False)
        gs1 = fig1.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.05, hspace=0.05)

        for i in range(nrows * ncols - 3):
            if i == 0:
                ax = fig1.add_subplot(gs1[:2, :2])
            elif i < 2 * ncols - 3:  ### first two rows of the dictionary plot
                if i < ncols - 1:
                    ax = fig1.add_subplot(gs1[0, i + 1])
                else:
                    ax = fig1.add_subplot(gs1[1, i - (ncols - 1) + 2])
            else:
                i1 = i + 3
                a = i1 // ncols
                b = i1 % ncols
                ax = fig1.add_subplot(gs1[a, b])

            ax.imshow(self.W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
            # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
            # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(title)
        fig1.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig1.savefig(save_folder + '/' + save_filename)
        plt.show()

    def omit_chain_edges(self, X):
        ### input is (k^2 x N) matrix X
        ### make all entries corresponding to the edges of the conditioned chain motif zero
        ### This may be applied to patches matrix and also to the dictionary matrix

        k = self.k1 + self.k2 + 1  # size of the network patch
        ### Reshape X into (k x k x N) tensor
        X1 = X.copy()
        X1 = X1.reshape(k, k, -1)

        ### for each slice along mode 2, make the entries along |x-y|=1 be zero
        for i in np.arange(X1.shape[-1]):
            for x in itertools.product(np.arange(k), repeat=2):
                if np.abs(x[0] - x[1]) == 1:
                    X1[x[0], x[1], i] = 0

        return X1.reshape(k ** 2, -1)

    def reconstruct(self,
                            recons_iter=100,
                            if_save_history=True,
                            if_construct_WtdNtwk=True,
                            use_checkpoint_refreshing=False,
                            ckpt_epoch=1000,
                            jump_every=None,
                            omit_chain_edges=False,  ### Turn this on for denoising
                            omit_folded_edges=True,
                            if_save_wtd_reconstruction=False,
                            edge_threshold=0.5,
                            return_weighted=False,
                            edges_added=None,
                            if_keep_visit_statistics=False,
                            print_patches=False,
                            save_filename=None,
                            save_folder=None,
                            verbose=True):
        print('reconstructing given network...')
        '''
        NNetwork version of the reconstruction algorithm (custom Neighborhood Network package for scalable Glauber chain sampling)
        Using large "ckpt_epoch" improves reconstruction accuracy but uses more memory
        edges_added = list of false edges added to the original network to be denoised by reconstruction
        '''
        if save_folder is None:
            save_folder = "Network_dictionary"

        G = self.G
        self.G_recons = Wtd_NNetwork()
        self.G_recons_baseline = Wtd_NNetwork()  ## reconstruct only the edges used by the Glauber chain
        self.G_overlap_count = Wtd_NNetwork()
        self.G_recons_baseline.add_nodes(nodes=[v for v in G.vertices])
        self.G_recons.add_nodes(nodes=[v for v in G.vertices])
        self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

        B = self.path_adj(self.k1, self.k2)
        k = self.k1 + self.k2 + 1  # size of the network patch
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb = self.tree_sample(B, x0)
        emb_history = emb.copy()
        code_history = np.zeros(2 * self.n_components)

        ### Extend the learned dictionary for the flip-symmetry of the path embedding
        atom_size, num_atoms = self.W.shape
        W_ext = np.empty((atom_size, 2 * num_atoms))
        W_ext[:, 0:num_atoms] = self.W[:, 0:num_atoms]
        W_ext[:, num_atoms:(2 * num_atoms)] = np.flipud(self.W[:, 0:num_atoms])

        W_ext_reduced = W_ext

        ### Set up paths and folders
        default_folder = 'Temp_save_graphs'
        default_name_recons = 'temp_wtd_edgelist_recons'
        default_name_recons_baseline = 'temp_wtd_edgelist_recons_baseline'
        path_recons = default_folder + '/' + default_name_recons + '.txt'
        path_recons_baseline = default_folder + '/' + default_name_recons_baseline + '.txt'

        t0 = time()
        c = 0

        if omit_chain_edges:
            ### omit all chain edges from the extended dictionary
            W_ext_reduced = self.omit_chain_edges(W_ext)

        if(verbose):
            f = trange
        else:
            f = np.arange

        for t in f(recons_iter):
            meso_patch = self.get_single_patch_glauber(B, emb, omit_folded_edges=omit_folded_edges)
            patch = meso_patch[0]
            emb = meso_patch[1]
            if (jump_every is not None) and (t % jump_every == 0):
                x0 = np.random.choice(np.asarray([i for i in G.vertices]))
                emb = self.tree_sample(B, x0)

            # meso_patch[2] = nofolding_indicator matrix

            # print('patch', patch.reshape(k, k))
            if omit_chain_edges:
                ### omit all chain edges from the patches matrix
                patch_reduced = self.omit_chain_edges(patch)

                coder = SparseCoder(dictionary=W_ext_reduced.T,  ### Use extended dictioanry
                                    transform_n_nonzero_coefs=None,
                                    transform_alpha=0,
                                    transform_algorithm='lasso_lars',
                                    positive_code=True)
                # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
                # This only occurs when sparse coding a single array
                code = coder.transform(patch_reduced.T)
            else:
                coder = SparseCoder(dictionary=W_ext.T,  ### Use extended dictioanry
                                    transform_n_nonzero_coefs=None,
                                    transform_alpha=0,
                                    transform_algorithm='lasso_lars',
                                    positive_code=True)
                code = coder.transform(patch.T)

            if print_patches and edges_added is not None:
                P = patch.reshape(k, k)
                for x in itertools.product(np.arange(k), repeat=2):
                    a = emb[x[0]]
                    b = emb[x[1]]
                    if [a, b] in edges_added:
                        P[x[0], x[1]] = -P[x[0], x[1]]
                print('!!!!! Current sampled patch:\n', P.astype(int))

            if if_save_history:
                emb_history = np.vstack((emb_history, emb))
                code_history = np.vstack((code_history, code))

            patch_recons = np.dot(W_ext, code.T).T
            patch_recons = patch_recons.reshape(k, k)

            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                # edge = [str(a), str(b)] ### Use this when nodes are saved as strings, e.g., '154' as in DNA networks
                edge = [a, b]  ### Use this when nodes are saved as integers, e.g., 154 as in FB networks

                # print('!!!! meso_patch[2]', meso_patch[2])

                # if not (omit_folded_edges and meso_patch[2][x[0],x[1]]==0):
                #    print('!!!!!!!!! reconstruction masked')
                #    print('!!!!! meso_patch[2]', meso_patch[2])
                if self.G_overlap_count.has_edge(a, b) == True:
                    # print(G_recons.edges)
                    # print('ind', ind)
                    j = self.G_overlap_count.get_edge_weight(a, b)
                else:
                    j = 0

                if self.G_recons.has_edge(a, b) == True:
                    new_edge_weight = (j * self.G_recons.get_edge_weight(a, b) + patch_recons[x[0], x[1]]) / (j + 1)
                else:
                    new_edge_weight = patch_recons[x[0], x[1]]

                if if_construct_WtdNtwk:
                    if new_edge_weight > 0 and not (omit_chain_edges and np.abs(x[0] - x[1]) == 1):
                        self.G_recons.add_edge(edge, weight=new_edge_weight, increment_weights=False)
                        ### Add the same edge to the baseline reconstruction
                        ### if x[0] and x[1] are adjacent in the chain motif
                    if np.abs(x[0] - x[1]) == 1:
                        # print('baseline edge added!!')
                        self.G_recons_baseline.add_edge(edge, weight=1, increment_weights=False)

                    if not (omit_chain_edges and np.abs(x[0] - x[1]) == 1):
                        self.G_overlap_count.add_edge(edge, weight=j + 1, increment_weights=False)

                if if_keep_visit_statistics and not (omit_chain_edges and np.abs(x[0] - x[1]) == 1):
                    if not self.G_overlap_count.has_colored_edge(edge[0], edge[1]):
                        self.G_overlap_count.add_colored_edge([edge[0], edge[1], np.abs(x[0] - x[1])])
                        ### np.abs(x[0]-x_{1}) = distance on the chain motif
                    else:
                        # colored_edge_weight = self.G_overlap_count.get_colored_edge_weight(edge[0], edge[1])
                        # colored_edge = edge + colored_edge_weight + [np.abs(x[0]-x[1])]
                        # self.G_overlap_count.add_colored_edge(colored_edge)
                        colored_edge_weight = self.G_overlap_count.get_colored_edge_weight(edge[0], edge[1])[0]
                        self.G_overlap_count.add_colored_edge(
                            [edge[0], edge[1], (j * colored_edge_weight + np.abs(x[0] - x[1])) / (j + 1)])

            has_saved_checkpoint = False
            # progress status, saving reconstruction checkpoint, and memory refreshing
            if t % 1000 == 0:
                self.result_dict.update({'homomorphisms_history': emb_history})
                self.result_dict.update({'code_history': code_history})
                #print('iteration %i out of %i' % (t, recons_iter))

                # print('num edges in G_count', len(self.G_overlap_count.get_edges()))
                # print('num edges in G_recons', len(self.G_recons.get_edges()))

                if use_checkpoint_refreshing and t % ckpt_epoch == 0:
                    ### print out current memory usage
                    pid = os.getpid()
                    py = psutil.Process(pid)
                    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
                    print('memory use:', memoryUse)

                    ### Threshold and simplify the current reconstruction graph
                    G_recons_simplified = self.G_recons.threshold2simple(threshold=edge_threshold)
                    G_recons_combined = Wtd_NNetwork()
                    G_recons_combined.add_edges(edges=G_recons_simplified.get_edges(),
                                                edge_weight=1,
                                                increment_weights=True)

                    ### Do the same thing for the baseline reconstruction
                    G_recons_simplified_baseline = self.G_recons_baseline.threshold2simple(threshold=edge_threshold)
                    G_recons_combined_baseline = Wtd_NNetwork()
                    G_recons_combined_baseline.add_edges(edges=G_recons_simplified.get_edges(),
                                                         edge_weight=1,
                                                         increment_weights=True)

                    ### Load and combine with the saved edges and reconstruction counts
                    if has_saved_checkpoint:
                        G_recons_combined.load_add_wtd_edges(path=path_recons, increment_weights=True)
                        G_recons_combined_baseline.load_add_wtd_edges(path=path_recons_baseline, increment_weights=True)

                    ### Save current graphs
                    G_recons_combined.save_wtd_edgelist(default_folder=default_folder,
                                                        default_name=default_name_recons)
                    G_recons_combined_baseline.save_wtd_edgelist(default_folder=default_folder,
                                                                 default_name=default_name_recons_baseline)

                    has_saved_checkpoint = True

                    ### Clear up the edges of the current graphs

                    # self.G_overlap_count.clear_edges()
                    # self.G_recons.clear_edges()
                    self.G_recons = Wtd_NNetwork()
                    self.G_recons_baseline = Wtd_NNetwork()
                    self.G_overlap_count = Wtd_NNetwork()
                    self.G_recons.add_nodes(nodes=[v for v in G.vertices])
                    self.G_recons_baseline.add_nodes(nodes=[v for v in G.vertices])
                    self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

                # print('num edges in G_recons', len(self.G_recons.get_edges()))

        self.result_dict.update({'NDR iterations': recons_iter})
        self.result_dict.update({'omit_chain_edges for NDR': omit_chain_edges})

        ### Save weigthed reconstruction (To get the full version, turn off use_checkpoint_refreshing)
        if if_save_wtd_reconstruction:
            self.result_dict.update({'Edges in weighted reconstruction': self.G_recons.get_wtd_edgelist()})

        ### See how many false edges are ever reconstructed (applies only for denoising purpose)
        if (edges_added is not None) and (if_keep_visit_statistics == True):
            c = 0
            c_true = 0
            wt = 0
            wt_true = 0
            avg_n_visits2false_edge = 0
            avg_n_visits2true_edge = 0
            visit_counts_false = []
            visit_counts_true = []
            recons_weights_false = []
            recons_weights_true = []
            avg_dist_on_chain_false = []
            avg_dist_on_chain_true = []

            false_edge_reconstructed = []

            H = Wtd_NNetwork()
            H.add_edges(edges_added)
            edges_added = H.get_edges()  ### make it ordered pairs

            for edge in self.G.edges:
                if edge in edges_added:
                    if self.G_recons.has_edge(edge[0], edge[1]):
                        c += 1
                        wt += self.G_recons.get_edge_weight(edge[0], edge[1])
                        false_edge_reconstructed.append(edge)
                        avg_n_visits2false_edge += self.G_overlap_count.get_edge_weight(edge[0], edge[1])
                        recons_weights_false.append(self.G_recons.get_edge_weight(edge[0], edge[1]))
                        visit_counts_false.append(self.G_overlap_count.get_edge_weight(edge[0], edge[1]))
                        if if_keep_visit_statistics:
                            colored_edge_weight = self.G_overlap_count.get_colored_edge_weight(edge[0], edge[1])[0]
                            # avg_dist_on_chain_false.append(sum(colored_edge_weight)/len(colored_edge_weight))
                            avg_dist_on_chain_false.append(colored_edge_weight)
                            if colored_edge_weight <= 2:
                                print('!!!!! On-chain distance for false edge=', colored_edge_weight)
                    else:
                        recons_weights_false.append(0)

                else:
                    if self.G_recons.has_edge(edge[0], edge[1]):
                        c_true += 1
                        wt_true += self.G_recons.get_edge_weight(edge[0], edge[1])
                        false_edge_reconstructed.append(edge)
                        avg_n_visits2true_edge += self.G_overlap_count.get_edge_weight(edge[0], edge[1])
                        recons_weights_true.append(self.G_recons.get_edge_weight(edge[0], edge[1]))
                        visit_counts_true.append(self.G_overlap_count.get_edge_weight(edge[0], edge[1]))
                        if if_keep_visit_statistics:
                            colored_edge_weight = self.G_overlap_count.get_colored_edge_weight(edge[0], edge[1])[0]
                            # avg_dist_on_chain_true.append(sum(colored_edge_weight)/len(colored_edge_weight))
                            avg_dist_on_chain_true.append(colored_edge_weight)
                            if colored_edge_weight <= 2:
                                print('!!!!! On-chain distance for true edge=', colored_edge_weight)
                    else:
                        recons_weights_true.append(0)

            ### Get rid of the top 2% largest elements
            a = len(visit_counts_true) // 50
            visit_counts_true = sorted(visit_counts_true, reverse=True)[a:]
            b = len(visit_counts_true) // 50
            visit_counts_false = sorted(visit_counts_false, reverse=True)[b:]

            print('!!! n_false_edges', len(recons_weights_false))
            print('!!! n_true_edges', len(recons_weights_true))
            print('!!! max visits to false edges', max(visit_counts_false))
            print('!!! max visits to true edges', max(visit_counts_true))

            if not if_keep_visit_statistics:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=False)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), constrained_layout=False)
                # print('!!! avg distance on chain motif for false edges',
                #      sum(avg_dist_on_chain_false) / len(avg_dist_on_chain_false))
                # print('!!! avg distance on chain motif for false edges',
                #      sum(avg_dist_on_chain_true) / len(avg_dist_on_chain_true))

            ax[0].hist(recons_weights_false, bins='auto', alpha=0.3, label='Weights on F edges')
            ax[0].hist(recons_weights_true, bins='auto', alpha=0.3, label='Weights on T edges')
            ax[0].set_xlabel('# weights in reconstruction')
            ax[0].legend()

            ax[1].hist(visit_counts_false, bins='auto', alpha=0.3, label='Visit counts on F edges')
            ax[1].hist(visit_counts_true, bins='auto', alpha=0.3, label='Visit counts on T edges')
            ax[1].set_xlabel('# Glauber chain visits')
            ax[1].legend()

            if if_keep_visit_statistics:
                ax[2].hist(avg_dist_on_chain_false, bins='auto', alpha=0.3, label='Avg dist. on chain motif of F edges')
                ax[2].hist(avg_dist_on_chain_true, bins='auto', alpha=0.3, label='Avg dist. on chain motif of T edges')
                ax[2].set_xlabel('# Avg dist. on chain motif')
                ax[2].legend()

            if save_filename is not None:
                fig.savefig(save_folder + "/!!!denoising_histogram_" + save_filename + ".pdf")
            else:
                fig.savefig(save_folder + "/!!!denoising_histogram_.pdf")

            print('# of false edges ever reconstructed= %i out of %i' % (c, len(edges_added)))
            print('ratio of false edges ever reconstructed=', c / len(edges_added))
            print('avg reconstructed weight of false edges', wt / len(edges_added))
            print('avg reconstructed weight of true edges', wt_true / len(G.edges))
            print('avg_n_visits2false_edge', avg_n_visits2false_edge / len(edges_added))
            print('avg_n_visits2true_edge', avg_n_visits2true_edge / len(G.edges))

            self.result_dict.update({'False edges added': edges_added})
            self.result_dict.update({'False edges ever reconstructed': false_edge_reconstructed})
            self.result_dict.update({'ratio of false edges ever reconstructed': c / len(edges_added)})
            self.result_dict.update({'avg reconstructed weight of false edges': wt / len(edges_added)})
            self.result_dict.update({'avg reconstructed true of false edges': wt_true / len(G.edges)})
            self.result_dict.update({'recons_weights_false': recons_weights_false})
            self.result_dict.update({'recons_weights_true': recons_weights_true})
            self.result_dict.update({'visit_counts_false': visit_counts_false})
            self.result_dict.update({'visit_counts_true': visit_counts_true})

        ### Finalize the simplified reconstruction graph
        G_recons_final = self.G_recons.threshold2simple(threshold=edge_threshold)
        G_recons_final_baseline = self.G_recons_baseline.threshold2simple(threshold=edge_threshold)
        if use_checkpoint_refreshing:
            ### Finalizing reconstruction
            G_recons_combined = Wtd_NNetwork()
            if not self.if_wtd_network:
                G_recons_combined.add_edges(edges=G_recons_final.get_edges(),
                                            edge_weight=1,
                                            increment_weights=True)
                G_recons_combined.load_add_wtd_edges(path=path_recons, increment_weights=True)
                G_recons_final = G_recons_combined.threshold2simple(threshold=edge_threshold)
            else:
                G_recons_combined.add_wtd_edges(edges=self.G_recons.get_wtd_edgelist(),
                                                increment_weights=True)
                G_recons_combined.load_add_wtd_edges(path=path_recons, increment_weights=True)
                G_recons_final = G_recons_combined

            ### Finalizing baseline reconstruction
            G_recons_combined_baseline = Wtd_NNetwork()
            if not self.if_wtd_network:
                G_recons_combined_baseline.add_edges(edges=G_recons_final_baseline.get_edges(),
                                                     edge_weight=1,
                                                     increment_weights=True)
                G_recons_combined_baseline.load_add_wtd_edges(path=path_recons_baseline, increment_weights=True)
                G_recons_final_baseline = G_recons_combined_baseline.threshold2simple(threshold=edge_threshold)
            else:
                G_recons_combined_baseline.add_wtd_edges(edges=self.G_recons_baseline.get_wtd_edgelist(),
                                                         increment_weights=True)
                G_recons_combined.load_add_wtd_edges(path=path_recons_baseline, increment_weights=True)
                G_recons_final_baseline = G_recons_combined_baseline

            self.G_recons = G_recons_final
            self.G_recons_baseline = G_recons_final_baseline
            print('Num edges in recons', len(G_recons_final_baseline.get_edges()))
            print('Num edges in recons_baseline', len(G_recons_final_baseline.get_edges()))

        self.result_dict.update({'Edges reconstructed': G_recons_final.get_edges()})
        self.result_dict.update({'Edges reconstructed in baseline': G_recons_final_baseline.get_edges()})

        print('Reconstructed in %.2f seconds' % (time() - t0))
        # print('result_dict', self.result_dict)
        if if_save_history:
            self.result_dict.update({'homomorphisms_history': emb_history})
            self.result_dict.update({'code_history': code_history})
        if(return_weighted):
            return self.G_recons
        else:
            return G_recons_final

    def network_completion(self, filename, threshold=0.5, recons_iter=100, foldername=None):
        print('reconstructing given network...')
        '''
        Networkx version of the network completion algorithm
        Scale the reconstructed matrix B by np.max(A) and compare with the original network. 
        '''

        G = self.G
        G_recons = G
        G_overlap_count = nx.DiGraph()
        G_overlap_count.add_nodes_from([v for v in G])
        B = self.path_adj(self.k1, self.k2)
        k = self.k1 + self.k2 + 1  # size of the network patch
        x0 = np.random.choice(np.asarray([i for i in G]))
        emb = self.tree_sample(B, x0)
        t0 = time()
        c = 0

        if foldername is None:
            foldername = "Network_dictionary"

        for t in np.arange(recons_iter):
            patch, emb = self.get_single_patch_glauber(B, emb)
            coder = SparseCoder(dictionary=self.W.T,
                                transform_n_nonzero_coefs=None,
                                transform_alpha=0,
                                transform_algorithm='lasso_lars',
                                positive_code=True)
            # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
            # This only occurs when sparse coding a single array
            code = coder.transform(patch.T)
            patch_recons = np.dot(self.W, code.T).T
            patch_recons = patch_recons.reshape(k, k)

            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                ind1 = int(G_overlap_count.has_edge(a, b) == True)
                if ind1 == 1:
                    # print(G_recons.edges)
                    # print('ind', ind)
                    j = G_overlap_count[a][b]['weight']
                    new_edge_weight = (j * G_recons[a][b]['weight'] + patch_recons[x[0], x[1]]) / (j + 1)
                else:
                    j = 0
                    new_edge_weight = patch_recons[x[0], x[1]]

                ind2 = int(G_recons.has_edge(a, b) == True)
                if ind2 == 0:
                    G_recons.add_edge(a, b, weight=new_edge_weight)
                    G_overlap_count.add_edge(a, b, weight=j + 1)
            ### Only repaint upper-triangular

            # progress status
            # print('iteration %i out of %i' % (t, recons_iter))
            if 1000 * t / recons_iter % 1 == 0:
                print(t / recons_iter * 100)

        ### Round the continuum-valued Recons matrix into 0-1 matrix.
        G_recons_simple = nx.Graph()
        # edge_list = [edge for edge in G_recons.edges]
        for edge in G_recons.edges:
            [a, b] = edge
            conti_edge_weight = G_recons[a][b]['weight']
            binary_edge_weight = np.where(conti_edge_weight > threshold, 1, 0)
            if binary_edge_weight > 0:
                G_recons_simple.add_edge(a, b)

        self.G_recons = G_recons_simple
        ### Save reconstruction
        path_recons = foldername + '/' + str(foldername) + '/' + str(filename)
        nx.write_edgelist(G_recons,
                          path=path_recons,
                          data=False,
                          delimiter=",")
        print('Reconstruction Saved')
        print('Reconstructed in %.2f seconds' % (time() - t0))
        return G_recons_simple

    def compute_recons_accuracy_old(self, if_baseline=False):
        ### Compute reconstruction error
        G = self.G
        G_original = NNetwork()
        G_original.add_nodes(self.G.vertices)
        G_original.add_edges(self.G.get_edges())
        edges_original = G_original.get_edges()

        G_recons = NNetwork()
        G_recons.add_nodes(self.G.vertices)
        G_recons.add_edges(self.G_recons.get_edges())
        edges_recons = G_recons.get_edges()

        common_edges = G.intersection(G_recons)

        recons_accuracy = len(common_edges) / len(G_original.get_edges())
        print('# edges of original ntwk=', len(G_original.get_edges()))
        self.result_dict.update({'# edges of original ntwk': len(G_original.get_edges())})

        print('# edges of reconstructed ntwk=', len(G_recons.get_edges()))
        print('Jaccard reconstruction accuracy=', recons_accuracy)
        self.result_dict.update({'# edges of reconstructed ntwk=': len(G_recons.get_edges())})
        self.result_dict.update({'reconstruction accuracy=': recons_accuracy})

        if if_baseline:
            G_recons_baseline = NNetwork()
            G_recons_baseline.add_nodes(self.G.vertices)
            G_recons_baseline.add_edges(self.G_recons_baseline.get_edges())
            edges_recons_baseline = G_recons_baseline.get_edges()

            print('# edges of reconstructed baseline ntwk=', len(self.G_recons_baseline.get_edges()))
            common_edges_baseline = G.intersection(G_recons_baseline)
            recons_accuracy_baseline = len(common_edges_baseline) / len(G.get_edges())
            print('reconstruction accuracy for baseline=', recons_accuracy_baseline)
            self.result_dict.update(
                {'# edges of reconstructed baseline ntwk=': len(self.G_recons_baseline.get_edges())})
            self.result_dict.update({'reconstruction accuracy for baseline=': recons_accuracy_baseline})

        return recons_accuracy

    def compute_recons_accuracy(self, G_recons, if_baseline=False, edges_added=None):
        ### Compute reconstruction error
        G = self.G
        G_recons.add_nodes(G.vertices)
        common_edges = G.intersection(G_recons)
        recons_accuracy = len(common_edges) / (len(G.get_edges()) + len(G_recons.get_edges()) - len(common_edges))
        print('# edges of original ntwk=', len(G.get_edges()))
        self.result_dict.update({'# edges of original ntwk': len(G.get_edges())})

        print('# edges of reconstructed ntwk=', len(G_recons.get_edges()))
        print('Jaccard reconstruction accuracy=', recons_accuracy)
        self.result_dict.update({'# edges of reconstructed ntwk=': len(G_recons.get_edges())})
        self.result_dict.update({'reconstruction accuracy=': recons_accuracy})

        if if_baseline:
            print('# edges of reconstructed baseline ntwk=', len(self.G_recons_baseline.get_edges()))
            common_edges_baseline = G.intersection(self.G_recons_baseline)
            recons_accuracy_baseline = len(common_edges_baseline) / (
                    len(G.get_edges()) + len(self.G_recons_baseline.get_edges()) - len(common_edges_baseline))
            print('reconstruction accuracy for baseline=', recons_accuracy_baseline)
            self.result_dict.update(
                {'# edges of reconstructed baseline ntwk=': len(self.G_recons_baseline.get_edges())})
            self.result_dict.update({'reconstruction accuracy for baseline=': recons_accuracy_baseline})

        return recons_accuracy

    def compute_A_recons(self, G_recons):
        ### Compute reconstruction error
        G_recons.add_nodes_from(self.G.vertices)
        A_recons = nx.to_numpy_matrix(G_recons, nodelist=self.G.vertices)
        ### Having "nodelist=G.nodes" is CRUCIAL!!!
        ### Need to use the same node ordering between A and G for A_recons and G_recons.
        return A_recons


#### helper functions


def show_array(arr):
    ### Plots heatmap of an array
    ### Used to plot network adjcency matrix
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.xaxis.set_ticks_position('top')
    ax.imshow(arr, cmap='viridis', interpolation='nearest')  ### Without 'nearest', 1's look white not yellow.
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=10)
    # ax.tick_params(axis='x', which='minor', labelsize=8)
    plt.show()


def compute_max_component_stats(path):
    G = nx.Graph()
    edgelist = np.genfromtxt(path, delimiter=',', dtype=str)
    for e in edgelist:
        G.add_edge(e[0], e[1], weight=1)
    Gc = max(nx.connected_components(G), key=len)
    G_conn = G.subgraph(Gc)
    print('num_nodes in max_comp of G=', len(G_conn.nodes))
    print('num_edges in max_comp of G=', len(G_conn.edges))


