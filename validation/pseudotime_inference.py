
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import palantir as pt
import pandas as pd

from typing import *

from cellrank.kernels import CytoTRACEKernel
from validation.cellrank_workflow import (
    compute_rna_velocity, compute_rna_velo_transition_matrix, identify_initial_terminal_states, get_root_cell
)


def calculate_cytotrace_pt(
        adata: sc.AnnData,
        cluster_obs_key: str = 'clusters',
        plot: bool = False
) -> sc.AnnData:

    # Store original spliced counts elsewhere
    if 'spliced' in adata.layers.keys() and 'unspliced' in adata.layers.keys():
        adata.layers['og_spliced'] = adata.layers['spliced'].copy()
        adata.layers['og_unspliced'] = adata.layers['unspliced'].copy()

    adata.layers['spliced'] = adata.X
    adata.layers['unspliced'] = adata.X
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    ctk = CytoTRACEKernel(adata)
    ctk.compute_cytotrace()

    if plot:
        try:
            sc.pl.umap(adata, color=cluster_obs_key, legend_loc='on data')
        except KeyError:
            pass
        sc.pl.umap(adata, color='ct_pseudotime')

    return adata


def calculate_diffusion_pt(
        adata: sc.AnnData,
        root: Union[int, None] = None,
        cluster_obs_key: str = 'clusters',
        plot: bool = False
) -> sc.AnnData:

    # Get root cell if one was passed
    if root is not None:
        adata.uns['iroot'] = root
    else:
        if 'iroot' in adata.uns_keys():
            print(f'Using already existing root cell {adata.uns["iroot"]}')
        else:
            raise Exception(f"A root cell must be defined and stored at adata.uns['iroot']")

    # Calculate diffusion map and diffusion pseudotime
    sc.tl.diffmap(adata, n_comps=15)
    sc.tl.dpt(adata, n_dcs=10, n_branchings=0)

    if plot:
        try:
            sc.pl.umap(adata, color=cluster_obs_key, legend_loc='on data')
        except KeyError:
            pass
        sc.pl.umap(adata, color='dpt_pseudotime', show=False)
        xy = adata[adata.uns['iroot'], :].obsm['X_umap'].flatten()
        plt.plot(xy[0], xy[1], color='red', marker='o', markersize=6, zorder=10, label='root')
        plt.show()

    return adata


def calculate_palantir_pt(
        adata: sc.AnnData,
        root: Union[int, None] = None,
        layer_key: Union[str, None] = None,
        cluster_obs_key: str = 'clusters',
        plot: bool = False
) -> sc.AnnData:

    """
    Calculate cell wise pseudotime values using the Palantir method.

    This function computes diffusion maps and Palantir pseudotime values for single-cell RNA-seq data stored
    in an AnnData object. The root cell can be specified manually, or an existing root stored in
    `adata.uns['iroot']` can be used. Optionally, UMAP visualizations are generated to highlight
    the root cell and plot the pseudotime ordering.

    Args:
        adata (sc.AnnData): The input AnnData object containing single-cell RNA-seq data.
        root (Union[int, None], optional): Index of the root cell for pseudotime calculation. If None,
            the root must be predefined in `adata.uns['iroot']`. Defaults to None.
        layer_key (Union[str, None], optional): The key of the data layer to use for pseudotime calculation.
            If None, the main expression matrix (`adata.X`) is used. Defaults to None.
        cluster_obs_key (str, optional): The key in `adata.obs` representing clusters. Used for plotting only.
            Defaults to 'clusters'.
        plot (bool, optional): Whether to generate UMAP and pseudotime visualizations. Defaults to False.

    Returns:
        sc.AnnData: The AnnData object with Palantir pseudotime and diffusion map results added.
    """

    if layer_key is not None:
        # Change main data layer to passed layer
        adata.layers['dummy'] = adata.X.copy()
        adata.X = adata.layers[layer_key].copy()

    # Run diffusion maps
    dm_res = pt.utils.run_diffusion_maps(adata, n_components=5)
    # Compute low dimensional embedding of the data based on the eigen gap
    ms_data = pt.utils.determine_multiscale_space(adata)

    # Get root cell if one was passed
    if root is not None:
        adata.uns['iroot'] = root
    else:
        if 'iroot' in adata.uns_keys():
            print(f'Using already existing root cell {adata.uns["iroot"]}')
        else:
            raise Exception(f"A root cell must be defined and stored at adata.uns['iroot']")

    root_cell_name = adata.obs_names[adata.uns['iroot']]

    if plot:
        try:
            root_cluster = adata.obs[cluster_obs_key][root]
            initial_state = pd.Series([root_cluster], index=[root_cell_name])
            pt.plot.highlight_cells_on_umap(adata, initial_state)
            plt.show()
        except KeyError:
            pass

    # Run palantir
    pr_res = pt.core.run_palantir(adata, root_cell_name, num_waypoints=500)

    if plot:
        pt.plot.plot_palantir_results(adata, s=3)
        plt.show()

        sc.pl.umap(adata, color='palantir_pseudotime', show=False)
        xy = adata[adata.uns['iroot'], :].obsm['X_umap'].flatten()
        plt.plot(xy[0], xy[1], color='red', marker='o', markersize=6, zorder=10, label='root')
        plt.show()

    if layer_key is not None:
        # Change back layers
        adata.X = adata.layers['dummy'].copy()

    return adata


def find_initial_cell(
        adata: sc.AnnData,
        initial_terminal_state: Union[Tuple[str, str], None] = None,
        plot: bool = False
):
    # Compute RNA-velocity and transition matrix computed based on RNA-velocity
    adata = compute_rna_velocity(
        adata=adata,
        scvelo_pp=True,
        layer_key=None,
        plot=plot
    )
    vk = compute_rna_velo_transition_matrix(
        adata=adata,
        layer_key=None,
        plot=plot
    )
    # Compute terminal and initial states using the Generalized Perron Cluster Cluster Analysis (GPCCA) estimator
    gpcca_estimator = identify_initial_terminal_states(
        cr_kernel=vk,
        plot=plot,
        allow_overlap=False,
        initial_terminal_state=initial_terminal_state
    )
    # Annotate adata, root cell = cell with highest terminal state probability
    adata, root = get_root_cell(adata=adata, verbosity=1)

    return adata, root

