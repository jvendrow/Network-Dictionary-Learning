import numpy as np
import networkx as nx
from ndl.NNetwork import Wtd_NNetwork
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def recons_accuracy(G, G_recons):
    """
    Calculates the Jaccard index (accuracy) between the orignal graph and the reconstructed graph

    Parameters
    ----------
    G: Wtd_NNetwork object
        The original network.

    G_recons: Wtd_NNetwork object
        The reconstructed network. 

    Returns
    -------
    recons_accuracy: float
        The Jaccard index (accuracy) between the graphs
    """

    G_recons.add_nodes(G.vertices)
    common_edges = G.intersection(G_recons)
    recons_accuracy = len(common_edges) / (len(G.get_edges()) + len(G_recons.get_edges()) - len(common_edges))

    return recons_accuracy


def convex_hull_roc(x, y):
    
    """
    Calculates the convex hull of an ROC curve.
    
    Parameters
    ----------
    x: list, type float in [0,1]
        The x coordinates for the ROC curve given by points (x,y).
        
    y: list, type float in [0,1]
        The y coordinates for the ROC curve given by points (x,y).
        
    Returns
    -------
    x_hull: list, type float in [0,1]
        The x coordinates for the new ROC curve given by points 
        (x_hull,y_hull), the convex hull or curve (x,y).
        
    y_hull: list, type float in [0,1]
        The y coordinates for the new ROC curve given by points 
        (x_hull,y_hull), the convex hull or curve (x,y).
        
    acc: float
        The AUC for the convex hull of the ROC curve, calculated by:
        sklearn.metrics.auc(x_hull,y_hull).
    """

    x = np.array([0] + list(x) + [1,1])
    y = np.array([0] + list(y) + [1,0])
    hull = ConvexHull(np.vstack((x, y)).T)
    vert = hull.vertices
    vert.sort()
    
    vert.sort()
    
    x_hull = [x[i] for i in vert if not (x[i]==1 and y[i]==0)]
    y_hull = [y[i] for i in vert if not (x[i]==1 and y[i]==0)]

    acc = auc(x_hull,y_hull)
    return x_hull, y_hull, acc



def corrupt(G, path_save=None,
                           delimiter=',',
                           parameter=0.1,
                           noise_nodes=None,
                           noise_type='ER'):
    """
    Corrupts a graph G with additive or subtractive noise. 
    
    Options:
        noise_type='ER': ADDS nx.erdos_renyi_graph(noise_nodes, parameter)
        noise_type='WS': ADDS nx.watts_strogatz_graph(noise_nodes, 2 * parameter // noise_nodes, 0.3)
        noise_type='BA': ADDS nx.barabasi_albert_graph(noise_nodes, parameter)
        noise_type='negative': DELETES (parameter %) edges from the graph chosen randomly
        
    Parameters
    ----------
    G: Wtd_NNetwork object
        Graph onto which to apply corruption.

    path_save: string
        By default, None. If not none, path for saving an edgelist of the
        corrupted graph.

    delimiter: string
        By default, ','. If path_save is not None, delimiter to use
        when saving edgelist.

    parameter: float
        Parameter to use when applying corrputon (See options above).

    noise_nodes: list
        If not None, the subset of nodes onto which to apply corruption.
        If None, noise_nodes is all the nodes in the graph, which we recommend.

    noise_type: string
        The type of noise to apply to the graph G (See options above).

    Returns
    -------
    G_corrupt: Wtd_NNetwork object
       The corrupted graph form from G

    edges_changed: list
        The list of edges added if additive noisen or the
        edges deleted if subtractive noise
    """

    noise_sign = "added"

    edges_added = []
    node_list = [v for v in G.nodes()]

    
    # randomly sample nodes from original graph
    if(noise_nodes == None):
        noise_nodes = len(node_list)
        d = node_list
    else:
        sample = np.random.choice(node_list, noise_nodes, replace=False)
        d = {n: sample[n] for n in range(0, noise_nodes)}  ### set operation
    G_noise = nx.Graph()
    # Generate corrupt network
    if noise_type == 'ER':
        G_noise = nx.erdos_renyi_graph(noise_nodes, parameter)

    elif noise_type == 'WS':
        # number of edges in WS(n, d, p) = (d/2) * n, want this to be "parameter".
        G_noise = nx.watts_strogatz_graph(noise_nodes, 2 * parameter // noise_nodes, 0.3)
        # G_noise = nx.watts_strogatz_graph(100, 50, 0.4)
    elif noise_type == 'BA':
        G_noise = nx.barabasi_albert_graph(noise_nodes, parameter)


    edges = list(G_noise.edges)

    G_new = nx.Graph()

    edgelist = np.random.permutation(G.get_edges())
    for e in edgelist:
        G_new.add_edge(e[0], e[1], weight=1)

    # Overlay corrupt edges onto graph
    for edge in edges:

        if not (G.has_edge(d[edge[0]], d[edge[1]])):
            edges_added.append([d[edge[0]], d[edge[1]]])
            G_new.add_edge(d[edge[0]], d[edge[1]], weight=1)


    if noise_type == 'negative':
        ### take a minimum spanning tree and add back edges except ones to be deleted
        noise_sign = "deleted"
        full_edge_list = G.get_edges()
        G_diminished = nx.Graph(full_edge_list)
        Gc = max(nx.connected_components(G_diminished), key=len)
        G_diminished = G_diminished.subgraph(Gc).copy()
        full_edge_list = [e for e in G_diminished.edges]

        G_new = nx.Graph()
        G_new.add_nodes_from(G_diminished.nodes())
        mst = nx.minimum_spanning_edges(G_diminished, data=False)
        mst_edgelist = list(mst)  # MST edges
        G_new = nx.Graph(mst_edgelist)

        edges_non_mst = []
        for edge in full_edge_list:
            if edge not in mst_edgelist:
                edges_non_mst.append(edge)

        idx_array = np.random.choice(range(len(edges_non_mst)), int(len(edges_non_mst)*parameter), replace=False)
        edges_deleted = [full_edge_list[i] for i in idx_array]
        for i in range(len(edges_non_mst)):
            if i not in idx_array:
                edge = edges_non_mst[i]
                G_new.add_edge(edge[0], edge[1])

    edges_changed = edges_added
    if noise_type == 'negative':
        edges_changed = edges_deleted

    # Change this according to the location you want to save it
    if(type(path_save) != type(None)):
        nx.write_edgelist(G_new, path, data=False, delimiter=',')

    ### Output network as Wtd_NNetwork class
    G_corrupt = Wtd_NNetwork()
    G_corrupt.add_wtd_edges(G_new.edges())

    return G_corrupt, edges_changed


def permute_nodes(path_load, path_save):
    # Randomly permute node labels of a given graph
    edgelist = np.genfromtxt(path_load, delimiter=',', dtype=int)
    edgelist = edgelist.tolist()
    G = nx.Graph()
    for e in edgelist:
        G.add_edge(e[0], e[1], weight=1)

    node_list = [v for v in G.nodes]
    permutation = np.random.permutation(np.arange(1, len(node_list) + 1))

    G_new = nx.Graph()
    for e in edgelist:
        G_new.add_edge(permutation[e[0] - 1], permutation[e[1] - 1], weight=1)

    nx.write_edgelist(G, path_save, data=False, delimiter=',')

    return G_new


def calculate_AUC(x, y):

    """
    Simple function to calculate AUC of an ROC curve, given
    a set of points (x,y)

    Parameters
    ----------
    x: list
        The x coordinates of the points of the ROC curve.

    y: list
        The y coordinates of the points of the ROC curve.

    Returns
    -------
    total: the AUC for the set of points (x,y)
    """

    total = 0
    for i in range(len(x) - 1):
        total += np.abs((y[i] + y[i + 1]) * (x[i] - x[i + 1]) / 2)

    return total


def auc_roc(G_original,
                    G_corrupted,
                    G_recons,
                    path_save=None,
                    noise_type="positive",
                    convex_hull=True):
    """
    Calculate the AUC and plot the ROC curve for a corruption task,
    given the original, corrupted, and reconstructed graph.

    Parameters
    ----------
    G_original: Wtd_NNetwork object
        The original graph G.

    G_corrupted: Wtd_NNetwork object
        The corrupted graph G.

    G_recon: Wtd_NNetwork object
        The reconstructed graph G. This graph should be a reconstruction
        of G_corrupted using a dictionary learned from G_corrupted, so it
        uses no information from G_original.

    path_save: string
        By default, None. If not none, path for saving image of ROC curve.

    noise_type: string
        Either "positive" or "negative". The type of corruption added to the 
        graph, "positive" of additive noise and "negative" if subtractive noise.

    convex_hull: bool
        If true, takes the convex hull of the ROC curve.

    Returns
    -------
    ac: float
        The AUC of the ROC curve.

    fpr: list
        The false positive rate (x axis) coordinates of the ROC.

    tpr: list
        The true positive rate (y axis) coordinates of the ROC.
    """
    edgelist_original = G_original.get_edges()

    edgelist_full = G_corrupted.get_edges()

    y_true = []
    y_pred = []

    j = 0
    if noise_type == "positive":
        for edge in edgelist_full:
            j += 1

            pred = G_recons.get_edge_weight(edge[0], edge[1])

            if pred == None:
                y_pred.append(0)
            else:
                y_pred.append(pred)

            if edge in edgelist_original:
                y_true.append(1)
            else:
                y_true.append(0)
    elif noise_type == "negative":
        V = G_original.nodes()

        for i in np.arange(len(V)):
            for j in np.arange(i, len(V)):
                if not G_corrupted.has_edge(V[i], V[j]):
                    pred = G_recons.get_edge_weight(V[i], V[j])
                    if pred == None:
                        y_pred.append(0)
                    else:
                        y_pred.append(pred)

                    if G_original.has_edge(V[i], V[j]):
                        y_true.append(1)
                    else:
                        y_true.append(0)

    else:
        raise ValueError("Expected noise_type = 'positive' or 'negative but got noise_type={}".format(noise_type))


    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    if not convex_hull:
        ac = calculate_AUC(fpr, tpr)

    else:
        fpr, tpr, ac = convex_hull_roc(fpr, tpr)

    fig, axs = plt.subplots(1, 1, figsize=(4.3, 5))

    axs.plot(fpr, tpr)
    axs.legend(["ROC (AUC = %f.2)" % ac])
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8, wspace=0.1, hspace=0.1)

    if path_save is not None:
        fig.savefig(path_save)

    return ac, fpr, tpr

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
