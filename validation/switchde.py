import numpy as np
import pandas as pd
import scanpy as sc

import os
import anndata2ri
import logging
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from typing import *

from switchtfi.utils import anndata_to_numpy


def calculate_switch_de_pvalues(
        adata: sc.AnnData,
        zero_inflated: bool = False,
        pt_obs_key: str = 'palantir_pseudotime',
        layer_key: str = 'log1p_norm',
        verbosity: int = 0
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """
    Calculate switchde p-values for analysis of differential expression over pseudotime.

    This function uses the switchde method (implemented in R) to identify genes with significant
    expression changes along a pseudotime trajectory. Switchde offers a model for normal and a model
    for zero-inflated data. The results include gene wise p-values, Benjamini-Hochberg-corrected q-values, and
    parameters from switchde's sigmoid based model.

    Args:
        adata (sc.AnnData): The input AnnData object containing gene expression data.
        zero_inflated (bool, optional): Whether to use a zero-inflated model. Defaults to False.
        pt_obs_key (str, optional): The key in `adata.obs` for the pseudotime values. Defaults to 'palantir_pseudotime'.
        layer_key (str, optional): The data layer to use for gene expression. Defaults to 'log1p_norm'.
        verbosity (int, optional): Level of logging for detailed output. Defaults to 0.

    Returns:
        Tuple[sc.AnnData, pd.DataFrame]: The AnnData object annotated with SwitchDE results and a DataFrame
        containing gene-wise p-values, q-values, and model parameters.
    """

    if layer_key is None:
        data = anndata_to_numpy(adata, layer_key=None).T
    else:
        data = anndata_to_numpy(adata=adata, layer_key=layer_key).T

    genes = adata.var_names.to_numpy()
    pt = adata.obs[pt_obs_key].to_numpy()

    # Activate R interface and set up logging
    anndata2ri.activate()
    pandas2ri.activate()
    rcb.logger.setLevel(logging.ERROR)

    # Set up R environment and transfer raw counts from Python to R
    ro.globalenv['verbosity'] = verbosity
    ro.globalenv['data'] = data
    ro.globalenv['genes'] = genes
    ro.globalenv['pseudotime'] = pt
    ro.globalenv['zero_inflated'] = zero_inflated

    # Run R code using rpy2
    ro.r.source(os.path.join(os.path.dirname(__file__), 'switchde.R'))

    # Retrieve results from R to Python
    out = ro.globalenv['sde']

    # Annotate AnnData with SwitchDE output
    adata.var['switchde_pval'] = out['pval'].to_numpy()
    adata.var['switchde_qval'] = out['qval'].to_numpy()  # BH corrected p-values
    adata.varm['switchde_params'] = np.vstack(
        [2 * out['mu0'].to_numpy(),  # see paper, def sigmoid
        out['k'].to_numpy(),
        out['t0'].to_numpy()]
    ).T

    # Sort genes by q-value
    out.sort_values(by='qval', ascending=True, inplace=True)
    out.reset_index(drop=True)

    if verbosity >= 1:
        print(out)

    return adata, out




