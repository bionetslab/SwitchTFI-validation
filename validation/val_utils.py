
import pandas as pd
import numpy as np
import networkx as nx

from typing import *
from itertools import combinations
from switchtfi.tf_ranking import grn_to_nx


def compare_grns(grn_list: List[pd.DataFrame],
                 tf_target_keys: Tuple[str, str] = ('TF', 'target')):

    vertex_jis = []
    edge_jis = []
    tf_jis = []
    target_jis = []
    for i, j in combinations(range(len(grn_list)), 2):
        vertex_jis.append(calc_ji_vertices(grn1=grn_list[i], grn2=grn_list[j], tf_target_keys=tf_target_keys))
        edge_jis.append(calc_ji_edges(grn1=grn_list[i], grn2=grn_list[j], tf_target_keys=tf_target_keys))
        tf_jis.append(ji(a=set(grn_list[i][tf_target_keys[0]].to_list()),
                         b=set(grn_list[j][tf_target_keys[0]].to_list())))
        target_jis.append(ji(a=set(grn_list[i][tf_target_keys[1]].to_list()),
                             b=set(grn_list[j][tf_target_keys[1]].to_list())))

    vertex_jis_mean = sum(vertex_jis) / len(vertex_jis)
    edge_jis_mean = sum(edge_jis) / len(edge_jis)
    tf_jis_mean = sum(tf_jis) / len(tf_jis)
    target_jis_mean = sum(target_jis) / len(target_jis)

    return vertex_jis_mean, edge_jis_mean, tf_jis_mean, target_jis_mean, vertex_jis, edge_jis, tf_jis, target_jis


def calc_ji_vertices(grn1: pd.DataFrame,
                     grn2: pd.DataFrame,
                     tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> float:
    vertices1 = set(np.unique(grn1[list(tf_target_keys)].to_numpy()))
    vertices2 = set(np.unique(grn2[list(tf_target_keys)].to_numpy()))
    return ji(vertices1, vertices2)


def calc_ji_edges(grn1: pd.DataFrame,
                  grn2: pd.DataFrame,
                  tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> float:
    edges1 = set([(grn1.at[i, tf_target_keys[0]], grn1.at[i, tf_target_keys[1]]) for i in range(grn1.shape[0])])
    edges2 = set([(grn2.at[i, tf_target_keys[0]], grn2.at[i, tf_target_keys[1]]) for i in range(grn2.shape[0])])
    return ji(a=edges1, b=edges2)


def ji(a: set, b: set) -> float:
    return len(a.intersection(b)) / len(a.union(b))


def compare_gene_sets(res_df_list: List[pd.DataFrame],
                      top_k: int = 10,
                      gene_key: str = 'gene'):

    top_k_jis = []
    for i, j in combinations(range(len(res_df_list)), 2):
        top_k_jis.append(ji(a=get_top_k_set(res_df=res_df_list[i], top_k=top_k, gene_key=gene_key),
                            b=get_top_k_set(res_df=res_df_list[j], top_k=top_k, gene_key=gene_key)))

    top_k_jis_mean = sum(top_k_jis) / len(top_k_jis)

    return top_k_jis_mean, top_k_jis


def get_top_k_set(res_df: pd.DataFrame,
                  top_k: int = 10,
                  gene_key: str = 'gene') -> Set[str]:
    return set(res_df[gene_key].tolist()[0:top_k])


def compare_grn_vs_rand_background(base_grn: pd.DataFrame,
                                   transition_grn: pd.DataFrame,
                                   n: int = 100,
                                   tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> Tuple[list, list]:
    ccs_list = [None] * n
    n_vertices = [0] * n

    # Define auxiliary function that samples a random subnetwork of tgrn with the same number of edges as bgrn
    def rand_subnet_generator(bgrn: pd.DataFrame,
                              tgrn: pd.DataFrame,
                              mode: str = 'simple',
                              random_state: int = 1725149318) -> pd.DataFrame:

        if mode == 'simple':
            # Sample uniform at random among all edges of bgrn
            randgrn = bgrn.sample(n=tgrn.shape[0], replace=False, random_state=random_state, axis=0)
        elif mode == 'advanced':
            # 1) Sample TFs (as many as there are in tgrn)
            # 2) Sample uniform at random among edges starting at sampled TFs
            n_tfs_t = np.unique(tgrn[tf_target_keys[0]].to_numpy()).shape[0]
            tfs_b = np.unique(bgrn[tf_target_keys[0]].to_numpy())
            sampled_tfs_b = np.random.choice(tfs_b, size=n_tfs_t, replace=False)
            sampled_tf_bool_b = np.isin(bgrn[tf_target_keys[0]].to_numpy(), sampled_tfs_b)
            dummy_bgrn = bgrn[sampled_tf_bool_b].copy()
            randgrn = dummy_bgrn.sample(n=tgrn.shape[0], replace=False, axis=0)
        else:
            randgrn = pd.DataFrame()

        return randgrn

    # Subsample n times and calculate metrics
    for i in range(n):
        # Randomly select edges from the base GRN -> random subnetwork
        rand_subnet = rand_subnet_generator(bgrn=base_grn, tgrn=transition_grn, mode='simple', random_state=i)
        # Turn random subnetwork into undirected Networkx graph
        g = grn_to_nx(grn=rand_subnet, edge_attributes=None, tf_target_keys=tf_target_keys).to_undirected()
        # Store list of connected components of graph
        ccs_list[i] = list(nx.connected_components(g))
        # Store number of vertices in random subnetwork
        n_vertices[i] = g.number_of_nodes()

    return ccs_list, n_vertices



