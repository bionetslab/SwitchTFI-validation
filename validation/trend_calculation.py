
import numpy as np
import pandas as pd
import scanpy as sc
import mellon as ml
import jax


from typing import *
from pygam import LinearGAM, s

from switchtfi.utils import anndata_to_numpy


def calculate_pygam_gene_trends(
        adata: sc.AnnData,
        gene_names: Union[List[str], None] = None,
        n_splines: int = 4,
        spline_order: int = 2,
        pseudotime_obs_key: str = 'palantir_pseudotime',
        trend_resolution: int = 200,
        layer_key: Union[str, None] = None
) -> sc.AnnData:
    """
    Calculate gene expression trends over pseudotime using PyGAM.

    This function fits Generalized Additive Models (GAMs) to model gene expression trends over pseudotime.
    It can also compute confidence intervals for the predicted trends. The results are stored
    in the AnnData object.

    Args:
        adata (sc.AnnData): The input AnnData object containing gene expression data.
        gene_names (Union[List[str], None], optional): List of gene names to compute trends for. If None, trends for
            all genes are computed. Defaults to None.
        n_splines (int, optional): Number of splines to use for the GAM model. Defaults to 4.
        spline_order (int, optional): Order of the splines in the GAM model. Defaults to 2.
        pseudotime_obs_key (str, optional): The key in `adata.obs` for the pseudotime values.
            Defaults to 'palantir_pseudotime'.
        trend_resolution (int, optional): The number of points to evaluate the trend along pseudotime. Defaults to 200.
        layer_key (Union[str, None], optional): The data layer to use for gene expression.
            If None, the main expression matrix is used. Defaults to None.

    Returns:
        sc.AnnData: The AnnData object with the computed gene trends and confidence intervals stored in
        `varm['gam_gene_trends']` and `uns['gam_confidence_intervals']`.
    """

    pt_vec = adata.obs[pseudotime_obs_key].to_numpy()
    pt_grid = np.linspace(pt_vec.min(), pt_vec.max(), trend_resolution)

    if gene_names is None:
        gene_names = adata.var_names.to_list()

    # gene_trends = pd.DataFrame(np.zeros(adata.shape), columns=adata.var_names)
    try:
        # Check if gene trend and cis were already computed for some genes
        gene_trends = pd.DataFrame(adata.varm['gam_gene_trends'].T, columns=adata.var_names)
        confidence_intervals = adata.uns['gam_confidence_intervals']
    except KeyError:
        gene_trends = pd.DataFrame(np.zeros((trend_resolution, adata.n_vars)), columns=adata.var_names)
        confidence_intervals = {}

    for gene in gene_names:
        expression_vec = anndata_to_numpy(adata[:, gene], layer_key=layer_key).flatten()
        gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order)).fit(
            X=pt_vec.reshape(-1, 1),
            y=expression_vec,
            weights=None
        )
        gene_trends[gene] = gam.predict(X=pt_grid.reshape(-1, 1))

        # Calculate confidence intervals of prediction
        confidence_intervals[gene] = gam.confidence_intervals(X=pt_grid.reshape(-1, 1), width=0.95).T
        # Has dimension: trend_resolution x 2  -> transpose ...

    adata.uns['pt_grid'] = pt_grid
    adata.varm['gam_gene_trends'] = gene_trends.values.T
    adata.uns['gam_confidence_intervals'] = confidence_intervals

    return adata


def calculate_mellon_gene_trends(
        adata: sc.AnnData,
        pseudotime_obs_key: str = 'palantir_pseudotime',
        trend_resolution: int = 200,
        layer_key: Union[str, None] = None
) -> sc.AnnData:
    # Get pseudotime-vector and expression matrix, initialize pseudotime-grid
    pt_vec = adata.obs[pseudotime_obs_key].to_numpy()
    expressions = anndata_to_numpy(adata=adata, layer_key=layer_key)
    pt_grid = np.linspace(pt_vec.min(), pt_vec.max(), trend_resolution)

    # Initialize mellon model
    model = ml.FunctionEstimator(sigma=1, ls=1)
    trends = model.fit_predict(
        x=pt_vec,
        y=expressions,
        Xnew=pt_grid
    )  # (pt_grid.shape[0], expressions.shape[1])
    adata.uns['pt_grid'] = pt_grid
    adata.varm['mellon_gene_trends'] = jax.device_get(trends.T)

    return adata



