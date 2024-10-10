
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import cellrank as cr
import scipy.stats as st

from typing import *
from switchtfi.utils import anndata_to_numpy


def compute_rna_velocity(adata: sc.AnnData,
                         scvelo_pp: bool = True,
                         layer_key: Union[str, None] = None,
                         plot: bool = False) -> sc.AnnData:
    # Plot percentages of un-/spiced reads
    if plot:
        scv.pl.proportions(adata)

    if scvelo_pp:
        # See: https://cellrank.readthedocs.io/en/latest/notebooks/tutorials/kernels/200_rna_velocity.html
        scv.pp.filter_and_normalize(
            adata, min_shared_counts=20, n_top_genes=2000, subset_highly_variable=False
        )

        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, random_state=0)

    if layer_key is not None:
        # Change main data layer to passed layer
        adata.layers['dummy'] = adata.X.copy()
        adata.X = adata.layers[layer_key].copy()

    # Compute moments
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    # Estimate model parameters
    scv.tl.recover_dynamics(adata, n_jobs=16)
    # Compute velocities
    scv.tl.velocity(adata, mode="dynamical")

    if layer_key is not None:
        # Change back layers
        adata.X = adata.layers['dummy'].copy()

    return adata


def compute_rna_velo_transition_matrix(adata: sc.AnnData,
                                       layer_key: Union[str, None] = 'log1p_norm',
                                       result_folder: Union[str, None] = None,
                                       plot: bool = False,
                                       **kwargs) -> cr.kernels.Kernel:
    if layer_key is not None:
        # Change main data layer to passed layer
        adata.layers['dummy'] = adata.X.copy()
        adata.X = adata.layers[layer_key].copy()

    # Set up velocity kernel
    vk = cr.kernels.VelocityKernel(adata)
    # Compute (cel-cell) transition matrix
    vk.compute_transition_matrix()

    # Combine velocity kernel with connectivity kernel => robustness
    ck = cr.kernels.ConnectivityKernel(adata)
    ck.compute_transition_matrix()
    combined_kernel = 0.8 * vk + 0.2 * ck

    if plot:
        # vk.plot_random_walks(start_ixs={"clusters": "Pre-endocrine"}, max_iter=200, seed=0)
        vk.plot_random_walks(start_ixs={"clusters": "Pre-endocrine"}, max_iter=200, seed=0)
        vk.plot_projection()

    if result_folder is not None:
        prefix = kwargs.get('fn_prefix')
        if prefix is None:
            prefix = ''
        vk.write_to_adata()
        k_p = os.path.join(result_folder, f'{prefix}vk.h5ad')
        adata.write(k_p, compression='gzip')

    return vk


def compute_rna_velo_transition_matrix2(adata: sc.AnnData,
                                        combine_w_gene_expr_similarity: bool = True,
                                        result_folder: Union[str, None] = None,
                                        plot: bool = False,
                                        **kwargs):
    prefix = kwargs.get('fn_prefix')
    if prefix is None:
        prefix = ''

    # Set up velocity kernel
    vk = cr.kernels.VelocityKernel(adata)
    # Compute (cel-cell) transition matrix
    vk.compute_transition_matrix()

    # Combine velocity kernel with connectivity kernel => robustness
    if combine_w_gene_expr_similarity:
        ck = cr.kernels.ConnectivityKernel(adata)
        ck.compute_transition_matrix()

        combined_kernel = 0.8 * vk + 0.2 * ck

        if plot:
            combined_kernel.plot_projection()
            combined_kernel.plot_random_walks(start_ixs={"clusters": "Pre-endocrine"}, max_iter=200, seed=0)

        if result_folder is not None:
            vk.write_to_adata()
            ck.write_to_adata()
            combined_kernel.write_to_adata()
            k_p = os.path.join(result_folder, f'{prefix}vk_ck_combk.h5ad')
            adata.write(k_p, compression='gzip')

        return vk, ck, combined_kernel

    if plot:
        vk.plot_projection()
        vk.plot_random_walks(start_ixs={"clusters": "Pre-endocrine"}, max_iter=200, seed=0)

    if result_folder is not None:
        vk.write_to_adata()
        k_p = os.path.join(result_folder, f'{prefix}vk.h5ad')
        adata.write(k_p, compression='gzip')

    return vk


def identify_initial_terminal_states(cr_kernel: cr.kernels.Kernel,
                                     cluster_obs_key: str = 'clusters',
                                     allow_overlap: bool = False,
                                     initial_terminal_state: Union[Tuple[str, str], None] = None,
                                     plot: bool = False) -> cr.estimators.GPCCA:
    # Initialize estimator
    gpcca = cr.estimators.GPCCA(cr_kernel)

    if initial_terminal_state is None:
        print('########################')
        # Fit estimator -> soft assignment of cells to macrostates, transition probabilities mtrx among these macrostates
        gpcca.fit(cluster_key=cluster_obs_key, n_states=2)  # Todo: n_states=2
        # Predict terminal states
        gpcca.predict_terminal_states(allow_overlap=allow_overlap)
        # Predict initial states
        gpcca.predict_initial_states(allow_overlap=allow_overlap)

    else:
        gpcca.compute_schur()
        gpcca.compute_macrostates(cluster_key=cluster_obs_key, n_states=2)
        gpcca.set_terminal_states(states=initial_terminal_state[1])
        gpcca.set_initial_states(states=initial_terminal_state[0])

    if plot:
        gpcca.plot_macrostates(which="all", discrete=True, legend_loc="on data", s=100)
        gpcca.plot_macrostates(which="terminal", legend_loc="on data", s=100)
        # Plot terminal state that cell most likely belongs to != fate probability
        gpcca.plot_macrostates(which="terminal", discrete=False)
        # Plot initial state that cell most likely belongs to != fate probability
        gpcca.plot_macrostates(which="initial", legend_loc="on data", s=100)

    return gpcca


def estimate_fate_probabilities(cr_estimator: cr.estimators.GPCCA,
                                plot: bool = False):
    # Initial and terminal states must have been identified beforehand ..
    cr_estimator.compute_fate_probabilities()
    if plot:
        cr_estimator.plot_fate_probabilities(same_plot=False)
        cr_estimator.plot_fate_probabilities(same_plot=True)

    return cr_estimator


def uncover_driver_genes(cr_estimator: cr.estimators.GPCCA,
                         verbosity: int = 0) -> Tuple[pd.DataFrame, cr.estimators.GPCCA]:

    cr_estimator.compute_eigendecomposition()
    res_df = cr_estimator.compute_lineage_drivers(cluster_key="clusters")
    if verbosity >= 1:
        print('###### Top-10 putative driver genes: ######')
        print(res_df[0:10])

    return res_df, cr_estimator


def visualize_expression_trends(adata: sc.AnnData,
                                genes: list[str]):
    # Driver genes must have been inferred beforehand
    # Initialize model for GAM fitting
    model = cr.models.GAMR(adata, n_knots=6, smoothing_penalty=10.0)

    cr.pl.gene_trends(
        adata,
        model=model,
        data_key='log1p_norm',
        genes=genes,
        same_plot=True,
        ncols=2,
        time_key="palantir_pseudotime",
        hide_cells=False,
        weight_threshold=(1e-3, 1e-3),
    )
    plt.show()

    return


def visualize_expression_trends2(adata: sc.AnnData,
                                 genes: list[str]):

    for gene in genes:
        bdata = adata[anndata_to_numpy(adata[:, gene]) != 0, :]
        model = cr.models.GAMR(bdata, n_knots=6, smoothing_penalty=10.0)
        cr.pl.gene_trends(
            bdata,
            model=model,
            data_key='log1p_norm',
            genes=[gene],
            same_plot=True,
            ncols=2,
            time_key="palantir_pseudotime",
            hide_cells=False,
            weight_threshold=(1e-3, 1e-3),
        )
        plt.show()

    return


def visualize_expression_trends3(adata: sc.AnnData,
                                 genes: list[str]):

    for gene in genes:
        bdata = adata[anndata_to_numpy(adata[:, gene]) != 0, :]
        model = cr.models.GAM(bdata, n_knots=6, smoothing_penalty=10.0)
        cr.pl.gene_trends(
            bdata,
            model=model,
            data_key='log1p_norm',
            genes=[gene],
            same_plot=True,
            ncols=2,
            time_key="palantir_pseudotime",
            hide_cells=False,
            weight_threshold=(1e-3, 1e-3),
        )

        print(model.x_test)
        print(model.y_test)
        plt.plot(model.x_test.flatten(), model.y_test.flatten(), color='red')
        plt.show()

    return


def visualize_expression_cascades_via_heatmaps(adata: sc.AnnData,
                                               driver_genes_df: pd.DataFrame):
    # Driver genes must have been inferred beforehand
    # Initialize model for GAM fitting
    model = cr.models.GAMR(adata, n_knots=6, smoothing_penalty=10.0)

    # plot heatmap
    cr.pl.heatmap(
        adata,
        model=model,
        cluster_key="clusters",
        show_fate_probabilities=True,
        data_key="log1p_norm",
        genes=driver_genes_df.head(40).index,
        time_key="palantir_pseudotime",
        figsize=(12, 10),
        show_all_genes=True,
        weight_threshold=(1e-3, 1e-3),
    )
    plt.show()

    return


def cluster_gene_expression_trends(adata: sc.AnnData,
                                   genes: Union[list[str], None] = None):
    # Initialize model for GAM fitting
    model = cr.models.GAMR(adata, n_knots=6, smoothing_penalty=10.0)

    if genes is None:
        genes = adata.var_names

    cr.pl.cluster_trends(
        adata,
        model=model,
        data_key="log1p_norm",
        genes=genes,
        lineage='Beta',
        time_key="palantir_pseudotime",
        weight_threshold=(1e-3, 1e-3),
        n_jobs=32,
        random_state=0,
        clustering_kwargs={"resolution": 0.2, "random_state": 0},
        neighbors_kwargs={"random_state": 0},
    )
    plt.show()

    # Get Anndata object of shape genes x ?
    gdata = adata.uns["lineage_Beta_trend"].copy()
    print(gdata)
    print(gdata.X)

    # Merge annotations
    cols = ["means", "dispersions"]
    gdata.obs = gdata.obs.merge(
        right=adata.var[cols], how="left", left_index=True, right_index=True
    )
    print(gdata)

    # Analyze gene clusters
    sc.tl.umap(gdata, random_state=0)
    sc.pl.embedding(gdata, basis="umap", color=["clusters", "means"], vmax="p95")
    plt.show()

    return


def get_cellrank_driver_genes(adata: sc.AnnData,
                              top_k: Union[int, None] = 10,
                              compute_velocity: bool = True,
                              scvelo_pp: bool = True,
                              layer_key: Union[str, None] = None,
                              initial_terminal_state: Union[tuple[str, str], None] = None,
                              plot: bool = False) -> Tuple[list[str], pd.DataFrame]:
    """
    Identify driver genes associated with cell fate transitions using CellRank.

    This function follows the standard CellRank workflow as detailed in their documentation.

    Args:
        adata (sc.AnnData): The input AnnData object containing scRNA-seq data.
        top_k (Union[int, None], optional): The number of top driver genes to return. If None, all driver genes are returned.
            Defaults to 10.
        compute_velocity (bool, optional): Whether to compute RNA velocity before identifying driver genes. Defaults to True.
        scvelo_pp (bool, optional): Whether to apply scVelo preprocessing for RNA velocity calculation. Defaults to True.
        layer_key (Union[str, None], optional): The key of the data layer to use for velocity computation. If None, the main matrix is used.
            Defaults to None.
        initial_terminal_state (Union[tuple[str, str], None], optional): Tuple specifying the initial and terminal states for GPCCA.
            If None, these states are inferred. Defaults to None.
        plot (bool, optional): Whether to generate plots for RNA velocity, transition matrix, and fate probabilities. Defaults to False.

    Returns:
        Tuple[list[str], pd.DataFrame]: A tuple containing a list of top driver gene names and a DataFrame with all driver gene results.
    """

    if compute_velocity:
        # Compute RNA-velocity
        adata = compute_rna_velocity(adata=adata, scvelo_pp=scvelo_pp, layer_key=layer_key, plot=plot)

    # Compute transition matrix based on RNA-velocity
    vk = compute_rna_velo_transition_matrix(adata=adata, layer_key=layer_key, plot=plot)

    # Compute terminal and initial states using the Generalized Perron Cluster Cluster Analysis (GPCCA) estimator
    gpcca_estimator = identify_initial_terminal_states(cr_kernel=vk,
                                                       initial_terminal_state=initial_terminal_state, plot=plot)

    # Compute fate probabilities, correlate fate probabilities with gene expression => Driver genes
    gpcca_estimator = estimate_fate_probabilities(cr_estimator=gpcca_estimator, plot=plot)
    res_df, gpcca_estimator = uncover_driver_genes(cr_estimator=gpcca_estimator, verbosity=0)

    res_df.reset_index(inplace=True, names='gene')

    if top_k is None:
        top_k = res_df.shape[0] - 1

    return res_df['gene'].to_list()[0:top_k], res_df


# Auxiliary ############################################################################################################
def get_root_cell(adata: sc.AnnData,
                  verbosity: int = 0) -> Tuple[sc.AnnData, int]:
    # Cellranks 'identify_initial_terminal_states()' must have been run before
    root = np.argmax(adata.obs['init_states_fwd_probs'].to_numpy())
    print(adata.obs['init_states_fwd_probs'].to_numpy())
    adata.uns['iroot'] = root
    if verbosity >= 1:
        print(f'# The cell with the highest initial state probability is {root}')
    return adata, int(root)


def cellrank_full_workflow(adata: sc.AnnData,
                           verbosity: int = 0,
                           plot: bool = False):

    # Compute RNA-velocity and transition matrix computed based on RNA-velocity
    adata = compute_rna_velocity(adata=adata, plot=plot)
    vk = compute_rna_velo_transition_matrix(adata=adata, plot=plot)

    # Compute terminal and initial states using the Generalized Perron Cluster Cluster Analysis (GPCCA) estimator
    gpcca_estimator = identify_initial_terminal_states(cr_kernel=vk, plot=plot)

    # Compute fate probabilities, correlate fate probabilities with gene expression => Driver genes
    gpcca_estimator = estimate_fate_probabilities(cr_estimator=gpcca_estimator, plot=plot)
    res_df, gpcca_estimator = uncover_driver_genes(cr_estimator=gpcca_estimator, verbosity=verbosity)

    # Compute PALANTIR pseudo-time using the cell with the highest initial state probability as the root
    from .pseudotime_inference import calculate_palantir_pt
    get_root_cell(adata=adata, verbosity=verbosity)
    adata = calculate_palantir_pt(adata=adata, root=None, plot=plot)

    # Compute and plot gene expression trends in pseudo-time
    visualize_expression_trends(adata=adata, genes=["Pax4", "Pdx1", "Fev", 'Nkx6-1', 'Elf3', 'Foxa3', 'Fos', "Rpl18a"])
    # visualize_expression_trends2(adata=adata, genes=["Pax4", "Pdx1", "Fev", 'Nkx6-1',
    #                                                  'Elf3', 'Foxa3', 'Fos', "Rpl18a"])
    # visualize_expression_trends(adata=adata, genes=["Pax4", "Pdx1"])
    # visualize_expression_trends2(adata=adata, genes=["Pax4"])
    # visualize_expression_trends3(adata=adata, genes=["Pax4"])

    visualize_expression_cascades_via_heatmaps(adata=adata, driver_genes_df=res_df)
    # cluster_gene_expression_trends(adata=adata,
    #                                genes=["Pax4", "Pdx1", "Fev", 'Nkx6-1', 'Elf3', 'Foxa3', 'Fos', "Rpl18a"])

    cluster_gene_expression_trends(adata=adata,
                                   genes=res_df.head(40).index.to_list())

