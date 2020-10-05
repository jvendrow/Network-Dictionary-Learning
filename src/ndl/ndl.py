# from utils.onmf.onmf import Online_NMF
from ndl.onmf import Online_NMF
from ndl.NNetwork import NNetwork, Wtd_NNetwork
from ndl.utils import utils
import numpy as np
import itertools
from time import time
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt
import networkx as nx
import os
import psutil
import matplotlib.gridspec as gridspec
import sys
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
                 sample_size=1000,
                 k=21,
                 alpha=None,
                 is_glauber_dict=True,
                 is_glauber_recons=True,
                 Pivot_exact_MH_rule=False,
                 ONMF_subsample=True,
                 batch_size=10,
                 if_wtd_network=False,
                 if_tensor_ntwk=False,
                 omit_folded_edges=False):

        """
        Constructor for the NetDictLearner Class

        Parameters
        ----------
        G: Wtd_NNetwork object
            Network to use for learning and reconstruction.

        n_components: int
            The number of element to include in the network dictionary.

        MCMC_iterations: int
            The number of monte carlo markov chain iterations to run
            for sampling the network during learning.

        sample_size: int
           Number of sample patches that form the minibatch matrix X_t at
           iterations t.

        k: int
            Length of chain motif to use for sampling.

        alpha: int
            By default None. If not none, L1 regularizer for code
            matrix H, which is th solution to the following minimization 
            problem:
                || X - WH||_F^2 + alpha * ||H||_1, 
            where the columns of X contain the sample patches and the columns
            of W form the network dictionary.

        is_glauber_dict: bool
            By default, True. If True, use glauber chain sampling to 
            sample patches during dictionary learning. Otherwise, use 
            pivon chain for sampling.

        is_glauber_recons: bool
            By default, True. If True, use glauber chain sampling to 
            sample patches during network reconstruction. Otherwise, 
            use pivon chain for sampling.

        ONMF_subsample: bool
            By default, True. If True, during the dictionary update step
            from W_{t-1} to W_t, subsample columns of X_t, the sample patches taken
            at iterations t. Else, use the entire matrix X_t.
        
        batch_size: int
             number of patches used for training dictionaries per ONMF iteration.


        omit_folded_edges: bool
            By default, True. If True, ignores edges that are 'folded,' meaning that
            they are already represented within each patch in another entry, caused
            by the MCMC motif folding on itself.

        """
        self.G = G  ### Full netowrk -- could have positive or negagtive edge weights (as a NNetwork or Wtd_NNetwork class)
        if if_tensor_ntwk:
            self.G.set_clrd_edges_signs()
            ### Each edge with weight w is assigned with tensor weight [+(w), -(w)] stored in the field colored_edge_weight

        self.n_components = n_components
        self.MCMC_iterations = MCMC_iterations
        self.sub_iterations = sub_iterations
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.if_tensor_ntwk = if_tensor_ntwk  # if True, input data is a 3d array
        self.omit_folded_edges = omit_folded_edges  # if True, get induced k by k patch without off-chain edges appearing
        ### due to folding of the underlying motif (e.g., completely folded k-chain --> no checkerboard pattern)
        self.W = np.random.rand((k + 1) ** 2, n_components)
        if if_tensor_ntwk:
            self.W = np.random.rand(G.color_dim * (k + 1) ** 2, n_components)

        self.k = k
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


    def train_dict(self, jump_every=20, verbose=True):
        """
        Performs the Network Dictionary Learning algorithm to train a dictionary
        of latent motifs that aim approximate any given 'patch' of the network.

        Parameters
        ----------
        jump_every: int
            By default, 20. The number of MCMC iterations to perform before
            resampling a patch to encourage coverage of the network.

        verbose: bool
            By default, True. If true, displays a progress bar for training. 

        Returns
        -------
        W: NumPy array, of size k^2 x r.
            The learned dictionary. Each of r columns contains a flattened latent 
            motif of shape k x k. 
        """

        if(verbose):
            print('training dictionaries from patches...')

        G = self.G
        B = self.path_adj(self.k)
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
                self.ntf = Online_NMF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      alpha=self.alpha,
                                      mode=2,
                                      learn_joint_dict=True,
                                      subsample=self.ONMF_subsample)  # max number of possible patches
                self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                self.H = code
            else:
                self.ntf = Online_NMF(X, self.n_components,
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
        self.result_dict.update({'Dictionary learned': self.W})
        self.result_dict.update({'Motif size': self.k})
        self.result_dict.update({'Code learned': self.code})
        self.result_dict.update({'Code COV learned': self.At})
        # print(self.W)
        return self.W

    def display_dict(self,
                     title="Dictionary",
                     path=None,
                     show=True,
                     make_first_atom_2by2=False,
                     show_importance=False):

        """
        Displays the learned dictionary, stored in self.W

        Parameters
        ----------
        title: str
            The title for the plot of the dictionary elements

        path: str
            By defualt, None. If not None, the path in which to 
            save the dictionary plot. 

        show: bool
            By default, True. Whether to show the dictionary plot,
            using plt.show()

        make_first_atom_2by2: bool
            By default, None. If True, increase the size of the top
            atom to emphasize it, as it has the highest 'importance'

        show_importance: bool
            By defualt, False. If True, list the 'importance' of the
            dictionary element under each element, calculated based
            on the code matrix H. 
        """

        W = self.W
        n_components = W.shape[1]
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if rows ** 2 == n_components:
            cols = rows
        else:
            cols = rows + 1


        k = self.k

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
            if type(path) != type(None):
                fig.savefig(path)

            if show:
                plt.show()
 

        else:
            if not self.if_tensor_ntwk:
                figsize = (5, 5)
                if show_importance:
                    figsize = (5, 6)

                fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,
                                        subplot_kw={'xticks': [], 'yticks': []})

                k = self.k  # number of nodes in the motif F
                for ax, j in zip(axs.flat, range(n_components)):
                    ax.imshow(self.W.T[idx[j]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    if show_importance:
                        ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=13)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    # use gray_r to make black = 1 and white = 0

                plt.suptitle(title)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)

                if type(path) != type(None):
                    fig.savefig(path)
                
                if show:
                    plt.show()

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

                if type(path) != type(None):
                    fig.savefig(path)
                
                if show:
                    plt.show()


    def reconstruct(self,
                            recons_iter=1000,
                            if_save_history=True,
                            ckpt_epoch=1000,
                            jump_every=None,
                            omit_chain_edges=False,  ### Turn this on for denoising
                            omit_folded_edges=True,
                            edge_threshold=0.5,
                            return_weighted=True,
                            verbose=True,
                            verbose_memory=False):


        """
        Reconstructs the network self.G using the learned dictionary self.W
        using the 'Network Reconstruction' algorithm. When using this fuction
        for the denoising application, we recommend: omit_chain_edges=True.

        Paremeters
        ----------
        recon_iters: int
            By default, 1000. The number if reconstruction iterations used to
            run the reconstruction algorithm. Higher iterations tend to give
            higher accuracy by allowing more time for convergence.

        if_save_history: bool
            By default, True. If True, save the history of each homomosphism
            sampled during reconstruction.

        use_checkpoint_refreshing: bool
            By default, False. If True, every ckpt_epoch iterations, save the current
            reconstruction and reset, and combine all results at the end. We recommend 
            turning this on to save memory. 

        ckpt_epoch: int
            Number of epochs between checkpoint refreshing, if the parameter above is 
            set to True.

        jump_every: int
            By default, None. If not None, the homomorphism is re-initialized every
            jump_every iterations to encourage visiting the full network. This is
            recommended when using Glauber chain for sampling.

        omit_chain_edges: bool
            By default, False. If True, omits chain edges during reconstruction
            to prevent reconstructing edges directly along the motif chain, which
            aids during denoising.

        omit_folded_edges: bool
            By default, True. If True, ignores edges that are 'folded,' meaning that
            they are already represented within each patch in another entry, caused
            by the MCMC motif folding on itself.

        edge_threshold: float
            If return_weighted is set to false, we set all edge weights above edge_threshold
            to 1, and set all others to 0, before returning the network.

        return_weighted: bool
            By default True. If True, return then return the weighted reconstructed graph.
            If False, return a simple graph thresholded at edge_threshold.

        verbose: bool
            By default, True. If True, shows a progress bar for reconstruction iterations
            completed.

        verbose_memory: bool
            By default, False. If True, shows details information about memory usage
            during every checkpoint refreshing iteration.

        Returns
        -------
        G_recons: Wtd_NNetwork
            The reconstructed network. If return_weighted=False, G_recons is
            thresholded to form a simple graph.
        """

        print('reconstructing given network...')
        
        self.result_dict.update({'NDR iterations': recons_iter})
        self.result_dict.update({'omit_chain_edges for NDR': omit_chain_edges})

        G = self.G
        self.G_recons = Wtd_NNetwork()
        self.G_recons_baseline = Wtd_NNetwork()  ## reconstruct only the edges used by the Glauber chain
        self.G_overlap_count = Wtd_NNetwork()
        self.G_recons_baseline.add_nodes(nodes=[v for v in G.vertices])
        self.G_recons.add_nodes(nodes=[v for v in G.vertices])
        self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

        B = self.path_adj(self.k)
        k = self.k  # size of the network patch
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
        default_folder = './Temp_save_graphs'
        default_name_recons = 'temp_wtd_edgelist_recons'
        default_name_recons_baseline = 'temp_wtd_edgelist_recons_baseline'
        default_name_overlap_count = 'temp_overlap_count'
        path_recons = default_folder + '/' + default_name_recons + '.txt'
        path_recons_baseline = default_folder + '/' + default_name_recons_baseline + '.txt'
        path_overlap_count = default_folder + '/' + default_name_overlap_count + '.txt'

        try:
            os.mkdir(default_folder)
        except OSError:
            pass

        t0 = time()
        if omit_chain_edges:
            ### omit all chain edges from the extended dictionary
            W_ext_reduced = self.omit_chain_edges(W_ext)

        has_saved_checkpoint = False

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
                print('homomorphism resampled')


            if omit_chain_edges:
                ### omit all chain edges from the patches matrix
                patch_reduced = self.omit_chain_edges(patch)

                coder = SparseCoder(dictionary=W_ext_reduced.T,  ### Use extended dictioanry
                                    transform_n_nonzero_coefs=None,
                                    transform_alpha=0,
                                    transform_algorithm='lasso_lars',
                                    positive_code=True)

                code = coder.transform(patch_reduced.T)
            else:
                coder = SparseCoder(dictionary=W_ext.T,  ### Use extended dictioanry
                                    transform_n_nonzero_coefs=None,
                                    transform_alpha=0,
                                    transform_algorithm='lasso_lars',
                                    positive_code=True)
                code = coder.transform(patch.T)

            if if_save_history:
                emb_history = np.vstack((emb_history, emb))
                code_history = np.vstack((code_history, code))

            patch_recons = np.dot(W_ext, code.T).T
            patch_recons = patch_recons.reshape(k, k)

            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                edge = [a, b]  ### Use this when nodes are saved as integers, e.g., 154 as in FB networks


                if not (omit_folded_edges and meso_patch[2][x[0], x[1]] == 0):
                    if self.G_overlap_count.has_edge(a, b) == True:
                        j = self.G_overlap_count.get_edge_weight(a, b)
                    else:
                        j = 0

                    if self.G_recons.has_edge(a, b) == True:
                        new_edge_weight = (j * self.G_recons.get_edge_weight(a, b) + patch_recons[x[0], x[1]]) / (j + 1)
                    else:
                        new_edge_weight = patch_recons[x[0], x[1]]


                    if np.abs(x[0] - x[1]) == 1:
                        self.G_recons_baseline.add_edge(edge, weight=1, increment_weights=False)

                    if not (omit_chain_edges and np.abs(x[0] - x[1]) == 1):

                        self.G_overlap_count.add_edge(edge, weight=j + 1, increment_weights=False)

                        if new_edge_weight > 0:
                            self.G_recons.add_edge(edge, weight=new_edge_weight, increment_weights=False)
                            ### Add the same edge to the baseline reconstruction
                            ### if x[0] and x[1] are adjacent in the chain motif


            # print progress status and memory use
            if t % 1000 == 0:
                self.result_dict.update({'homomorphisms_history': emb_history})
                self.result_dict.update({'code_history': code_history})

                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB

                if verbose_memory:
                    print('memory use:', memoryUse)

                    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
                         key= lambda x: -x[1])[:10]:
                        print("{:>30}: {:>8}".format(name, utils.sizeof_fmt(size)))

                    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
                        print("{:>30}: {:>8}".format(name, utils.sizeof_fmt(size)))


            # refreshing memory at checkpoints
            if (ckpt_epoch is not None) and (t % ckpt_epoch == 0):
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
                if verbose_memory:

                    print('memory use:', memoryUse)

                ### Load and combine with the saved edges and reconstruction counts
                if has_saved_checkpoint:

                    self.G_recons_baseline.load_add_wtd_edges(path=path_recons_baseline, increment_weights=True,
                                                              is_dict=True, is_pickle=True)

                    G_overlap_count_new = Wtd_NNetwork()
                    G_overlap_count_new.add_wtd_edges(edges=self.G_overlap_count.wtd_edges, is_dict=True)

                    G_overlap_count_old = Wtd_NNetwork()
                    G_overlap_count_old.load_add_wtd_edges(path=path_overlap_count, increment_weights=False,
                                                           is_dict=True, is_pickle=True)

                    G_recons_new = Wtd_NNetwork()
                    G_recons_new.add_wtd_edges(edges=self.G_recons.wtd_edges, is_dict=True)

                    self.G_recons = Wtd_NNetwork()
                    self.G_recons.load_add_wtd_edges(path=path_recons, increment_weights=False, is_dict=True,
                                                     is_pickle=True)

                    for edge in G_recons_new.wtd_edges.keys():
                        edge = eval(edge)
                        count_old = G_overlap_count_old.get_edge_weight(edge[0], edge[1])
                        count_new = self.G_overlap_count.get_edge_weight(edge[0], edge[1])

                        old_edge_weight = self.G_recons.get_edge_weight(edge[0], edge[1])
                        new_edge_weight = G_recons_new.get_edge_weight(edge[0], edge[1])

                        if old_edge_weight is not None:
                            new_edge_weight = (count_old / (count_old + count_new)) * old_edge_weight + (
                                        count_new / (count_old + count_new)) * new_edge_weight

                        elif count_old is not None:
                            new_edge_weight = (count_new / (count_old + count_new)) * new_edge_weight

                        self.G_recons.add_edge(edge, weight=new_edge_weight, increment_weights=False)
                        G_overlap_count_old.add_edge(edge=edge, weight=count_new, increment_weights=True)

                    self.G_overlap_count = G_overlap_count_old


                ### Save current graphs
                self.G_recons.save_wtd_edges(path_recons)
                self.G_overlap_count.save_wtd_edges(path_overlap_count)
                self.G_recons_baseline.save_wtd_edges(path_recons_baseline)

                has_saved_checkpoint = True

                ### Clear up the edges of the current graphs
                self.G_recons = Wtd_NNetwork()
                self.G_recons_baseline = Wtd_NNetwork()
                self.G_overlap_count = Wtd_NNetwork()
                self.G_recons.add_nodes(nodes=[v for v in G.vertices])
                self.G_recons_baseline.add_nodes(nodes=[v for v in G.vertices])
                self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])
                G_overlap_count_new = Wtd_NNetwork()
                G_overlap_count_old = Wtd_NNetwork()
                G_recons_new = Wtd_NNetwork()

        if ckpt_epoch is not None:
            self.G_recons = Wtd_NNetwork()
            self.G_recons.load_add_wtd_edges(path=path_recons, increment_weights=True, is_dict=True, is_pickle=True)
            self.G_recons_baseline = Wtd_NNetwork()
            self.G_recons_baseline.load_add_wtd_edges(path=path_recons_baseline, increment_weights=True, is_dict=True, is_pickle=True)

        ### Save weigthed reconstruction into full results dictionary
        self.result_dict.update({'Edges in weighted reconstruction': self.G_recons.wtd_edges})
        self.result_dict.update({'Edges reconstructed in baseline': self.G_recons_baseline.wtd_edges})

        if(verbose):
            print('Reconstructed in %.2f seconds' % (time() - t0))
        # print('result_dict', self.result_dict)
        if if_save_history:
            self.result_dict.update({'homomorphisms_history': emb_history})
            self.result_dict.update({'code_history': code_history})
        if(return_weighted):
            return self.G_recons
        else:
            return self.G_recons.threshold2simple(threshold=edge_threshold)


    #Helper Functions
    #-----------------------------------------------------------------------------------

    def omit_chain_edges(self, X):

        """
        Sets all entries corresponding to the edges of the conditioned chain
        motif to 0. Can be applied to patch matrices or the dictionary matrix.

        Parameters
        ----------
        X: NumPy matrix, size k^2 x N
            Patch or dictionary matrix for which to remove chain edges

        Returns
        -------
        X: NumPy matrix, size k^2 x N
            Patch or dictionary matrix with removed chain edges
        """

        k = self.k  # size of the network patch
        ### Reshape X into (k x k x N) tensor
        X1 = X.copy()
        X1 = X1.reshape(k, k, -1)

        ### for each slice along mode 2, make the entries along |x-y|=1 be zero
        for i in np.arange(X1.shape[-1]):
            for x in itertools.product(np.arange(k), repeat=2):
                if np.abs(x[0] - x[1]) == 1:
                    X1[x[0], x[1], i] = 0

        return X1.reshape(k ** 2, -1)


    def list_intersection(self, lst1, lst2):
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    def path_adj(self, k):
        # generates adjacency matrix for the path motif of k nodes
        A = np.eye(k, k=1, dtype=int)
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
        # k = length of chain
        # updates the current embedding using pivot rule

        x0 = emb[0]  # current location of pivot
        x0 = self.RW_update(x0, Pivot_exact_MH_rule=self.Pivot_exact_MH_rule)  # new location of the pivot
        B = self.path_adj(self.k)

        emb_new = self.tree_sample(B, x0)  # new path embedding
        return emb_new

    def RW_update(self, x, Pivot_exact_MH_rule=False):
        # G = simple graph
        # x = RW is currently at site x
        # stationary distribution = uniform
        # Pivot_exact_MH_rule = True --> RW is updated so that the Pivot chain is sampled from the exact conditional distribution
        # otherwise the pivot of the Pivot chain performs random walk with uniform distribution as its stationary distribution

        N = self.G
        length = self.k - 1  # number of edges in the chain motif
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

        """
        Parameters
        ----------
        B: NumPy array
            Adjacency matrix of motif
        emb: Current embedding F rightarrow G

        Returns
        -------
        X: NumPy array
            Sampled patch
        emb: Numpy array
            Current embedding F rightarrow G

        Mx: Numpy array
            if omit_folded_edges=True, an indicator
            of which edges are 'folded'
        """
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

