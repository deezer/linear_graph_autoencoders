import networkx as nx
import numpy as np
import scipy.sparse as sp
import warnings

warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

def compute_kcore(adj, nb_core):
    """ Computes the k-core version of a graph - See IJCAI 2019 paper
    for theoretical details on k-core decomposition
    :param adj: sparse adjacency matrix of the graph
    :param nb_core: a core number, from 0 to the "degeneracy"
                    (i.e. max core value) of the graph
    :return: sparse adjacency matrix of the nb_core-core subgraph, together
             with the list of nodes from this core
    """
    # Preprocessing on graph
    G = nx.from_scipy_sparse_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    # K-core decomposition
    core_number = nx.core_number(G)
    # nb_core subgraph
    kcore = nx.k_core(G, nb_core, core_number)
    # Get list of nodes from this subgraph
    nodes_kcore = kcore.nodes
    # Adjacency matrix of this subgraph
    adj_kcore = nx.adjacency_matrix(kcore)
    return adj_kcore, nodes_kcore

def expand_embedding(adj, emb_kcore, nodes_kcore, nb_iterations):
    """ Algorithm 2 'Propagation of latent representation' from IJCAI 2019 paper
    Propagates embeddings vectors computed on k-core to the remaining nodes
    of the graph (i.e. the nodes outside of the k-core)
    :param adj: sparse adjacency matrix of the graph
    :param emb_kcore: n*d embedding matrix computed from Graph AE/VAE
                      for nodes in k-core
    :param nodes_kcore: list of nodes in k-core
    :param nb_iterations: number of iterations "t" for fix-point iteration
                          strategy of Algorithm 2
    :return: n*d matrix of d-dim embeddings for all nodes of the graph
    """
    # Initialization
    num_nodes = adj.shape[0]
    emb = sp.csr_matrix((num_nodes, emb_kcore.shape[1]))
    emb[nodes_kcore,:] = emb_kcore
    adj = adj.tocsr()
    embedded_nodes = []
    new_embedded_nodes = np.array(nodes_kcore)

    # Assign latent space representation to nodes that were not in k-core
    while len(new_embedded_nodes) > 0:
        embedded_nodes = np.hstack((embedded_nodes, new_embedded_nodes))
        # Get nodes from V2 set
        reached_nodes = np.setdiff1d(np.where((adj[new_embedded_nodes,:].sum(0) != 0)), embedded_nodes)
        # Nodes from V1 (newly embedded) and V2
        new_embedded_nodes_union_reached = np.union1d(new_embedded_nodes, reached_nodes)
        # Adjacency matrices normalization by total degree in (A1,A2)
        adj_1_2 = adj[reached_nodes,:][:,new_embedded_nodes_union_reached]
        degrees = np.array(adj_1_2.sum(1))
        degree_mat = sp.diags(np.power(degrees, -1.0).flatten())
        adj_1 = degree_mat.dot(adj[reached_nodes,:][:,new_embedded_nodes])
        adj_2 = degree_mat.dot(adj[reached_nodes,:][:,reached_nodes])

        # Iterations
        z_1 = emb[new_embedded_nodes,:]
        z_2 = np.random.random_sample((len(reached_nodes), emb.shape[1]))
        for j in range(nb_iterations):
            z_2 = adj_1.dot(z_1) + adj_2.dot(z_2)
        emb[reached_nodes,:] += z_2
        # Update new_embedded_nodes
        new_embedded_nodes = reached_nodes

    # Handle isolated nodes
    emb[emb.getnnz(1) == 0] = np.mean(emb_kcore, axis=0)
    # Return embedding
    return emb.toarray()