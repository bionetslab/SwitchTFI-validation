
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import splicejac as sp
import matplotlib.pyplot as plt
import networkx as nx
import os

from typing import *


def get_splicejac_driver_genes(adata: sc.AnnData,
                               top_k: Union[int, None] = 10,
                               top_n_jacobian: int = 50,
                               layer_key: Union[str, None] = None,
                               splicejac_pp: bool = True,
                               compute_velocity: bool = True,
                               cluster_pair: Tuple[str, str] = ('Pre-endocrine', 'Beta'),
                               cluster_obs_key: str = 'clusters',
                               plot: bool = False) -> Tuple[List[str], pd.DataFrame, sc.AnnData]:
    """
    This function uses the spliceJAC method to infer gene regulatory networks (GRNs) and identify driver genes
    involved in specific cell state transitions.

    Args:
        adata (sc.AnnData): The input AnnData object containing scRNA-seq data.
        top_k (Union[int, None], optional): The number of top driver genes to return. Defaults to 10.
        top_n_jacobian (int, optional): The number of genes to use for the estimation of the Jacobian. Defaults to 50.
        layer_key (Union[str, None], optional): The key of the data layer to use for calculations.
            If None, the main matrix is used.
        splicejac_pp (bool, optional): Whether to apply preprocessing as per spliceJAC's tutorial. Defaults to True.
        compute_velocity (bool, optional): Whether to compute RNA velocity before identifying driver genes.
            Defaults to True.
        cluster_pair (Tuple[str, str], optional): The pair of cell clusters between which the transition is analyzed.
            Defaults to ('Pre-endocrine', 'Beta').
        cluster_obs_key (str, optional): The key in `adata.obs` for cluster annotations. Defaults to 'clusters'.
        plot (bool, optional): Whether to generate plots for transition genes. Defaults to False.

    Returns:
        Tuple[List[str], pd.DataFrame, sc.AnnData]: A tuple containing a list of top driver genes, a DataFrame
        with the driver gene results, and the modified AnnData object with spliceJAC results.
    """

    adata = adata.copy()
    if top_k is None:
        top_k = top_n_jacobian - 1

    if layer_key is not None:
        # Change main data layer to passed layer
        adata.layers['dummy'] = adata.X.copy()
        adata.X = adata.layers[layer_key].copy()

    if splicejac_pp:
        if layer_key is None:
            # Preprocess according to splicejac tutorial
            scv.pp.filter_genes(adata, min_shared_counts=20)
            scv.pp.normalize_per_cell(adata)
            scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)  # Extract highly variable genes
            sc.pp.log1p(adata)
        else:
            # Use passed layer (already preprocessed), just select highly variable genes
            scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)  # Extract highly variable genes

    # For following workflow see: https://splicejac.readthedocs.io/en/latest/notebooks/Transitions.html
    if compute_velocity:
        # Compute velocity
        scv.tl.velocity(adata)
        scv.tl.velocity_graph(adata)
        adata.uns['neighbors']['distances'] = adata.obsp['distances']
        adata.uns['neighbors']['connectivities'] = adata.obsp['connectivities']

    # Infer state specific GRNs (based on jacobian)
    sp.tl.estimate_jacobian(adata, n_top_genes=top_n_jacobian)
    # Analyze role of genes in GRNs
    sp.tl.grn_statistics(adata)

    # Identify top differentially expressed genes for each cell stage
    sc.tl.rank_genes_groups(adata, cluster_obs_key, method='t-test')
    # Analyze driver genes for specified transition
    sp.tl.transition_genes(adata, cluster_pair[0], cluster_pair[1])

    # Extract ranked list of genes, see splicejac git - plot_trans_genes()
    # (https://github.com/federicobocci/spliceJAC/blob/main/splicejac/plot/instability.py)
    # Get splicejac transition weights
    weight = adata.uns['transitions'][cluster_pair[0] + '-' + cluster_pair[1]]['weights']
    # Get genes
    genes = list(adata.var_names)
    # Sort weights -> indices of sorting
    ind = np.argsort(weight)
    # Get list of genes and corresponding weight
    data, trans_genes = [], []
    for i in range(top_k):
        trans_genes.append(genes[ind[weight.size - top_k + i]])
        data.append(weight[ind[weight.size - top_k + i]])

    res_df = pd.DataFrame()
    res_df['gene'] = trans_genes[::-1]
    res_df['splicejac_weight'] = data[::-1]

    print(res_df)
    print(trans_genes[::-1])

    if plot:
        sp.pl.plot_trans_genes(adata, cluster_pair[0], cluster_pair[1], top_trans_genes=top_k)
        plt.show()

    return trans_genes[::-1][0:top_k], res_df, adata


# Auxiliary ############################################################################################################
def extract_splicejac_grn(adata: sc.AnnData,
                          grn_adj_uns_key: str = 'average_jac',
                          clusters: Tuple[str, ...] = ('Pre-endocrine', 'Beta'),
                          weight_quantile: float = 0.5) -> pd.DataFrame:
    """
    Extract a transition specific gene regulatory network (GRN) from spliceJAC's results.

    This function extracts a GRN for the given clusters based on the Jacobian matrices stored in the
    AnnData object. It filters the GRN by keeping only edges with weights above or below a specified
    quantile threshold and returns the GRN as a DataFrame.

    Args:
        adata (sc.AnnData): The input AnnData object containing SpliceJAC results.
        grn_adj_uns_key (str, optional): The key in `adata.uns` where the Jacobian matrices for the starting and
            terminal states are stored. Defaults to 'average_jac'.
        clusters (Tuple[str, ...], optional): The clusters/states for which the GRN should be extracted.
            Defaults to ('Pre-endocrine', 'Beta').
        weight_quantile (float, optional): The quantile threshold for filtering the edge weights.
            Positive weights above this quantile and negative weights below 1 - this quantile are retained.
            Defaults to 0.5.

    Returns:
        pd.DataFrame: A DataFrame representing the combined GRN for the specified clusters, with transcription
        factors (TF) and their targets.
    """

    n = adata.n_vars
    grn_list = []
    for cluster in clusters:
        adj = adata.uns[grn_adj_uns_key][cluster][0][0:n, n:].copy().T

        # Compute weight_quantile-quantile of positive and negative weights
        q_pos = np.quantile(adj[adj > 0], weight_quantile)
        q_neg = np.quantile(adj[adj < 0], 1 - weight_quantile)

        # Set weights that are less extreme than the respective quantile values to zero
        adj[(adj > q_neg) & (adj < q_pos)] = 0

        # Create networkx graph from adjacency matrix and annotate with weights and gene names
        g = nx.from_numpy_array(adj, parallel_edges=False, create_using=nx.DiGraph)
        nx.relabel_nodes(g, dict(zip(range(adata.n_vars), adata.var_names.to_list())), copy=False)

        grn_list.append(nx.to_pandas_edgelist(g, source='TF', target='target'))

    grn = combine_grns(grn_list=grn_list,
                       n_occurrence_thresh=1,
                       result_folder=None,
                       tf_target_keys=('TF', 'target'),
                       verbosity=0)

    return grn


def combine_grns(grn_list: List[pd.DataFrame],
                 n_occurrence_thresh: int,
                 result_folder: Union[None, str],
                 tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                 verbosity: int = 0,
                 **kwargs) -> pd.DataFrame:

    # Get edges of all grns into one array
    edges_list = [grn[list(tf_target_keys)].to_numpy(dtype=str) for grn in grn_list]
    edges = np.vstack(edges_list)

    # Get unique edges and their number of occurrences
    unique_edges, n_occurrences = np.unique(edges, return_counts=True, axis=0)

    # Get edges that occur more or equally often than 'n_occurrence_thresh'
    keep_bool = (n_occurrences >= n_occurrence_thresh)
    keep_edges = unique_edges[keep_bool, :]

    grn = pd.DataFrame(keep_edges, columns=list(tf_target_keys))

    if result_folder is not None:
        prefix = kwargs.get('fn_prefix')  # Get prefix for filename, returns None if no root is passed in kwargs
        if prefix is None:
            prefix = ''
        final_grn_p = os.path.join(result_folder, f'{prefix}combined_grn.csv')
        grn.to_csv(final_grn_p)

    if verbosity >= 1:
        print(grn.head())
        genes = np.unique(grn[['TF', 'target']].to_numpy())
        print(f'# The combined GRN has {genes.shape[0]} vertices and {grn.shape[0]} edges')

    return grn


def splicejac_full_workflow(adata: sc.AnnData,
                            cell: str = 'Beta'):
    # adata = sc.read_h5ad(filename='../data/anndata/preendocrine_beta.h5ad')
    # adata = sc.read_h5ad(filename='../data/anndata/preendocrine_alpha.h5ad')

    # Preprocess according to splicejac tutorial
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)  # Extract highly variable genes
    sc.pp.log1p(adata)

    # Compute velocity
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    adata.uns['neighbors']['distances'] = adata.obsp['distances']
    adata.uns['neighbors']['connectivities'] = adata.obsp['connectivities']

    # Use PAGA to infer cell state transitions
    scv.tl.paga(adata, groups='clusters')

    # Save df of PAGA identified transitions for later use by spliceJAC:
    adata.uns['PAGA_paths'] = scv.get_df(adata, 'paga/transitions_confidence').T
    print(adata.uns['PAGA_paths'])
    # Plot
    scv.pl.paga(adata, basis='umap', size=50, alpha=.6, min_edge_width=2, node_size_scale=1.5, legend_loc='on data')

    # Infer state specific GRNs (based on jacobian)
    sp.tl.estimate_jacobian(adata, n_top_genes=50)
    grn_plots = True
    if grn_plots:
        # Plot the gene-gene interaction matrices of each cell state in a heatmap-style, activation=red, inhibition=blue
        sp.pl.visualize_jacobian(adata)
        plt.show()
        # Inspect stability of cell states -> plot eigenvalues of Jacobian, neg~stability, pos~instability -> pos percentage
        sp.pl.eigen_spectrum(adata)
        plt.show()
        # Plot state specific GRNs
        sp.pl.visualize_network(adata, 'Pre-endocrine', weight_quantile=.98, pos_style='circle', plot_interactive=False)
        plt.show()
        sp.pl.visualize_network(adata, 'Pre-endocrine', weight_quantile=.01, pos_style='circle', plot_interactive=False)
        plt.show()

        sp.pl.visualize_network(adata, cell, weight_quantile=.98, pos_style='spring',
                                plot_interactive=False, figsize=(8, 6))
        plt.show()
        sp.pl.visualize_network(adata, cell, weight_quantile=.90, pos_style='spring',
                                plot_interactive=False, figsize=(8, 6))
        plt.show()

        # Analyze role of genes in GRNs
        sp.tl.grn_statistics(adata)

        # Scatter-plot of weighted sum of incoming/outgoing edges in the GRN
        sp.pl.plot_signaling_hubs(adata, 'Pre-endocrine')
        plt.show()
        sp.pl.plot_signaling_hubs(adata, cell)
        plt.show()

        # Compare variablity GRN-statistics across states
        # (different options: statistics to compare, comparison metric; see doc ...)
        sp.pl.gene_variation(adata, method='inter_range')
        plt.show()
        # Inspect signalling role of most variable genes
        sp.pl.gene_var_detail(adata, measure='centrality')
        plt.show()
        # Scatterplot: inter-state gene variation vs average gene expression across clusters
        # => key genes for further investigation
        sp.pl.gene_var_scatter(adata)
        plt.show()

    # Analyze role of genes in GRNs
    sp.tl.grn_statistics(adata)

    # With knowledge of the transitions, we can quantify the change in signaling role
    sp.pl.plot_signaling_change(adata, 'Pre-endocrine', cell, figsize=(5, 4))
    plt.show()

    # Identify top differentially expressed genes for each cell stage
    sc.tl.rank_genes_groups(adata, 'clusters', method='t-test')

    # Analyze driver genes for all transitions identified by PAGA
    # sp.tl.trans_from_PAGA(adata)
    # For specific transition
    sp.tl.transition_genes(adata, 'Pre-endocrine', cell)
    # Plot to 10
    sp.pl.plot_trans_genes(adata, 'Pre-endocrine', cell)
    plt.show()

    # Plot core GRN
    # (top marker genes of the starting cell state (based on DE scores),
    # the top transition driver genes leading to the final state)
    sp.pl.core_GRN(adata, 'Pre-endocrine', cell, figsize=(5, 5))
    plt.show()
    # Scatter plot of DE scores vs transition scores -> genes are marker and/or driver
    sp.pl.scatter_scores(adata, 'Pre-endocrine', cell)
    plt.show()

    # Plot ‘Differential GRN’, highlighting the interactions with more dramatic change
    sp.pl.diff_network(adata, 'Pre-endocrine', cell, pos_style='circle',
                       figsize=(7, 5), weight_quantile=0.975)
    plt.show()

    # Plot more quantitative visualization of these interactions
    sp.pl.diff_interactions(adata, 'Pre-endocrine', cell, title='Top 10 Differential Interactions',
                            loc='upper center')
    plt.show()

    # Plot a ‘Conserved GRN’ of gene-gene interactions that remained relatively unchanged
    sp.pl.conserved_grn(adata, 'Pre-endocrine', cell, pos_style='circle',
                        weight_quantile=0.95, figsize=(7, 5))
    plt.show()

    # Visualize the strength of these interactions can be visualize quantitatively in a barplot
    sp.pl.top_conserved_int(adata, 'Pre-endocrine', cell,
                            title='Top 10 Conserved Interactions')
    plt.show()

    print(adata)


def grn_extraction_workflow():
    cell = 'beta'
    adata = sc.read_h5ad(filename=f'../data/anndata/preendocrine_{cell}.h5ad')

    cell = cell.capitalize()

    # Preprocess according to splicejac tutorial
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)  # Extract highly variable genes
    sc.pp.log1p(adata)

    # Compute velocity
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    adata.uns['neighbors']['distances'] = adata.obsp['distances']
    adata.uns['neighbors']['connectivities'] = adata.obsp['connectivities']

    # Use PAGA to infer cell state transitions
    scv.tl.paga(adata, groups='clusters')

    # Save df of PAGA identified transitions for later use by spliceJAC:
    adata.uns['PAGA_paths'] = scv.get_df(adata, 'paga/transitions_confidence').T
    print(adata.uns['PAGA_paths'])

    # Infer state specific GRNs (based on jacobian)
    sp.tl.estimate_jacobian(adata, n_top_genes=50)

    extract_splicejac_grn(adata=adata,
                          clusters=('Pre-endocrine', cell),  # ('Beta',)
                          weight_quantile=0.8)

    sp.pl.visualize_network(adata, 'Pre-endocrine', weight_quantile=.98, pos_style='circle', plot_interactive=False)
    plt.show()
    sp.pl.visualize_network(adata, cell, weight_quantile=.98, pos_style='circle', plot_interactive=False)
    plt.show()
