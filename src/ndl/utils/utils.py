import numpy as np
import networkx as nx
from ndl import Wtd_NNetwork
from sklearn.metrics import roc_curve
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def recons_accuracy(G, G_recons, if_baseline=False, edges_added=None, verbose=True):
    ### Compute reconstruction error
    G_recons.add_nodes(G.vertices)
    common_edges = G.intersection(G_recons)
    recons_accuracy = len(common_edges) / (len(G.get_edges()) + len(G_recons.get_edges()) - len(common_edges))

    if(verbose):
        print('# edges of original ntwk=', len(G.get_edges()))
        
        print('# edges of reconstructed ntwk=', len(G_recons.get_edges()))
        print('Jaccard reconstruction accuracy=', recons_accuracy)

    if if_baseline:
        print('# edges of reconstructed baseline ntwk=', len(self.G_recons_baseline.get_edges()))
        common_edges_baseline = G.intersection(self.G_recons_baseline)
        recons_accuracy_baseline = len(common_edges_baseline) / (
                len(G.get_edges()) + len(self.G_recons_baseline.get_edges()) - len(common_edges_baseline))
        print('reconstruction accuracy for baseline=', recons_accuracy_baseline)

    return recons_accuracy



def rocch(fpr0, tpr0):
    """
    @author: Dr. Fayyaz Minhas (http://faculty.pieas.edu.pk/fayyaz/)
    Construct the convex hull of a Receiver Operating Characteristic (ROC) curve
        Input:
            fpr0: List of false positive rates in range [0,1]
            tpr0: List of true positive rates in range [0,1]
                fpr0,tpr0 can be obtained from sklearn.metrics.roc_curve or
                    any other packages such as pyml
        Return:
            F: list of false positive rates on the convex hull
            T: list of true positive rates on the convex hull
                plt.plot(F,T) will plot the convex hull
            auc: Area under the ROC Convex hull
    """
    fpr = np.array([0] + list(fpr0) + [1.0, 1, 0])
    tpr = np.array([0] + list(tpr0) + [1.0, 0, 0])
    hull = ConvexHull(np.vstack((fpr, tpr)).T)
    vert = hull.vertices
    vert = vert[np.argsort(fpr[vert])]
    F = [0]
    T = [0]
    for v in vert:
        ft = (fpr[v], tpr[v])
        if ft == (0, 0) or ft == (1, 1) or ft == (1, 0):
            continue
        F += [fpr[v]]
        T += [tpr[v]]
    F += [1]
    T += [1]
    auc = np.trapz(T, F)
    return F, T, auc


def corrupt(G, path_save=None,
                           delimiter=',',
                           parameter=0.1,
                           noise_nodes=None,
                           noise_type='ER'):
    ### noise_type = 'ER' (Erdos-Renyi), 'WS' (Watts-Strongatz), 'BA' (Barabasi-Albert), '-ER_edges' (Delete ER edges)
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
    # SW = nx.watts_strogatz_graph(70,50,0.05)
    if noise_type == 'ER':
        G_noise = nx.erdos_renyi_graph(noise_nodes, parameter)
    elif noise_type == 'ER_edges':
        G_noise = nx.gnm_random_graph(noise_nodes, parameter)

    elif noise_type == 'WS':
        # number of edges in WS(n, d, p) = (d/2) * n, want this to be "parameter".
        G_noise = nx.watts_strogatz_graph(noise_nodes, 2 * parameter // noise_nodes, 0.3)
        print('!!! # edges in WS', len(G_noise.edges))
        # G_noise = nx.watts_strogatz_graph(100, 50, 0.4)
    elif noise_type == 'BA':
        G_noise = nx.barabasi_albert_graph(noise_nodes, parameter)
    elif noise_type == 'lattice':
        G_noise = nx.generators.lattice.grid_2d_graph(noise_nodes, noise_nodes)

    # some other possible corruption networks:
    # SW = nx.watts_strogatz_graph(150,149,0.3)
    # BA = nx.barabasi_albert_graph(100, 50)
    # n = range(1,101)
    # L = nx.generators.lattice.grid_2d_graph(40, 40)

    edges = list(G_noise.edges)
    # print(len(edges))

    G_new = nx.Graph()

    edgelist = np.random.permutation(G.get_edges())
    for e in edgelist:
        G_new.add_edge(e[0], e[1], weight=1)

    # Overlay corrupt edges onto graph
    for edge in edges:

        # for lattice graphs
        # ------------------------------------
        # edge1 = edge[0][0] * 40 + edge[0][1]
        # edge2 = edge[1][0] * 40 + edge[1][1]

        # if not (G.has_edge(d[edge1], d[edge2])):
        #    edges_added.append([d[edge1], d[edge2]])
        #    G.add_edge(d[edge1], d[edge2], weight=1)
        # ---------------------------------------
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
    if noise_type == '-ER_edges':
        edges_changed = edges_deleted

    # Change this according to the location you want to save it
    if(path_save != None):
        nx.write_edgelist(G_new, path, data=False, delimiter=',')

    ### Output network as Wtd_NNetwork class
    G_out = Wtd_NNetwork()
    G_out.add_wtd_edges(G_new.edges())

    return G_out, edges_changed


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
        # print('new edge', permutation[e[0]-1], permutation[e[1]-1])

    nx.write_edgelist(G, path_save, data=False, delimiter=',')

    return G_new


def calculate_AUC(x, y):

    total = 0
    for i in range(len(x) - 1):
        total += np.abs((y[i] + y[i + 1]) * (x[i] - x[i + 1]) / 2)

    return total


def auc_roc(G_original,
                    G_corrupted,
                    G_recons,
                    path_save=None,
                    flip_TF=False,
                    noise_type="positive",
                    verbose=True):
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
                if not flip_TF:
                    y_pred.append(pred)
                else:
                    y_pred.append(1 - pred)

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
                        if not flip_TF:
                            y_pred.append(pred)
                        else:
                            y_pred.append(1 - pred)

                    if G_original.has_edge(V[i], V[j]):
                        y_true.append(1)
                    else:
                        y_true.append(0)

    else:
        raise ValueError("Expected noise_type = 'positive' or 'negative but good noise_type={}".format(noise_type))


    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    F, T, ac = rocch(fpr, tpr)

    if(verbose):
        print("AUC with convex hull: ", ac)
        print("AUC without convex hull: ", calculate_AUC(fpr, tpr))

    fig, axs = plt.subplots(1, 1, figsize=(4.3, 5))
    axs.plot(F, T)
    axs.plot(fpr, tpr)
    axs.legend(["Convex hull ROC (AUC = %f.2)" % ac, "Original ROC (AUC = %f.2)" % calculate_AUC(fpr, tpr)])
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8, wspace=0.1, hspace=0.1)

    if path_save is not None:
        fig.savefig(path_save)

    return F, T, ac, fpr, tpr, thresholds

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
