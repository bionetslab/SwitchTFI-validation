
import numpy as np
import scanpy as sc
import scanpy.external as sce
import seaborn as sns
import matplotlib.pyplot as plt

import anndata2ri
import logging
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro

from rpy2.robjects import pandas2ri
from pathlib import Path
from scipy.stats import median_abs_deviation
from typing import *
import os

from switchtfi.utils import anndata_to_numpy


def filter_low_quality_reads(adata: sc.AnnData,
                             species: str = 'mus musculus',
                             pct_counts_mt_threshold: float = 8.0,
                             verbosity: int = 0,
                             plot: bool = False) -> sc.AnnData:
    """
    Filter low-quality cells from an AnnData object based on general count based metrics and mitochondrial counts.

    Args:
        adata (sc.AnnData): The AnnData object to filter.
        species (str, optional): The species of the dataset ('mus musculus' or 'homo sapiens').
            Defaults to 'mus musculus'.
        pct_counts_mt_threshold (float, optional): The threshold for the percentage of mitochondrial
            gene counts used to filter cells. Defaults to 8.0.
        verbosity (int, optional): Level of logging for detailed output. Defaults to 0.
        plot (bool, optional): Whether to plot QC metrics and filtering results. Defaults to False.

    Returns:
        sc.AnnData: The filtered AnnData object with low-quality cells removed.
    """

    if verbosity >= 1:
        print('### Filtering low quality reads ... ###')
    # Annotate genes that are mitochondrial
    if species == 'mus musculus':
        gene_prefix = 'mt-'
    elif species == 'homo sapiens':
        gene_prefix = 'MT-'
    else:
        gene_prefix = ''
    adata.var['mt'] = adata.var_names.str.startswith(gene_prefix)

    # Calculate QC-metrics, adds columns to .var, .obs
    # n_genes_by_counts = number of genes with positive counts in a cell
    # total_counts = total number of count for a cell (library size)
    # pct_counts_mt = proportion of total counts for a cell which are mitochondrial
    sc.pp.calculate_qc_metrics(adata,
                               qc_vars=['mt'],
                               inplace=True,
                               percent_top=[20],
                               log1p=True)

    if plot:
        p1 = sns.displot(adata.obs['total_counts'], bins=100, kde=False)
        # sc.pl.violin(adata, 'total_counts')
        p2 = sc.pl.violin(adata, 'pct_counts_mt')
        p3 = sc.pl.scatter(adata, 'total_counts', 'n_genes_by_counts', color='pct_counts_mt')
        plt.show()

    # Annotate cells that are outliers w.r.t. QC-metrics:
    # number of counts per barcode (count depth), number of genes per barcode, pct of counts in top 20 genes
    adata.obs['outlier'] = (
            is_outlier(adata, obs_key_qc_metric='log1p_total_counts', nmads=5)
            | is_outlier(adata, obs_key_qc_metric='log1p_n_genes_by_counts', nmads=5)
            | is_outlier(adata, obs_key_qc_metric='pct_counts_in_top_20_genes', nmads=5)
    )
    if verbosity >= 1:
        dummy = adata.obs.outlier.value_counts()
        print('# The number of general count based non-/outliers are:')
        print(dummy)

    # Annotate cells that are outliers w.r.t QC- metric:
    # fraction of counts from mitochondrial genes per barcode (or fraction > 8 percent)
    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | \
                              (adata.obs["pct_counts_mt"] > pct_counts_mt_threshold)

    if verbosity >= 1:
        print('# The number of mt-based non-/outliers are:')
        print(adata.obs.mt_outlier.value_counts())

    # Filter adata based on the identified outliers
    if verbosity >= 1:
        n_cells_before = adata.n_obs
        print(f'# Number of cells before filtering: {n_cells_before}')
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

    if verbosity >= 1:
        print(f'# Number of cells after filtering: {adata.n_obs}')
        print(f'# Number of cells removed due to low quality: {n_cells_before - adata.n_obs}')

    if plot:
        p1 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")
        plt.show()

    return adata


def correct_for_ambient_rna(adata: sc.AnnData,
                            unfiltered_adata: sc.AnnData,
                            verbosity: int = 0) -> sc.AnnData:
    """
    Correct for ambient RNA contamination in an AnnData object.

    This function corrects ambient RNA contamination in single-cell RNA-seq data using
    the SoupX R package. It computes Leiden clustering on normalized, log-transformed
    data and uses the resulting clusters to perform correction. The corrected counts
    are stored in the 'soupX_counts' layer, and the original counts are retained in
    the 'counts' layer.

    Args:
        adata (sc.AnnData): The filtered AnnData object to correct.
        unfiltered_adata (sc.AnnData): The unfiltered AnnData object used for ambient RNA correction.
        verbosity (int, optional): Level of logging. Defaults to 0.

    Returns:
        sc.AnnData: The AnnData object with ambient RNA-corrected counts.
    """

    if verbosity >= 1:
        print('### Correcting for ambient RNA ... ###')
    # Copy adata, normalize and apply log
    adata_pp = adata.copy()
    sc.pp.normalize_per_cell(adata_pp)
    sc.pp.log1p(adata_pp)

    # Compute leiden clustering
    sc.pp.pca(adata_pp)
    sc.pp.neighbors(adata_pp)
    sc.tl.leiden(adata_pp, key_added='soupx_groups')

    # Use clusters as groups for SoupX
    soupx_groups = adata_pp.obs['soupx_groups']

    # Delete adata
    del adata_pp

    # Save cell names, gene names and gene by cell matrix (filtered by cell quality, no transformations)
    cells = adata.obs_names
    genes = adata.var_names
    # data = anndata_to_numpy(adata, layer_key=None).T
    data = adata.X.T

    # Save gene by cell matrix (unfiltered, no transformations)
    # data_tod = anndata_to_numpy(unfiltered_adata, layer_key=None).T
    data_tod = unfiltered_adata.X.T

    # Activate R interface and set up logging
    anndata2ri.activate()
    pandas2ri.activate()
    rcb.logger.setLevel(logging.ERROR)
    # Load SoupX R library
    # ro.r.library('SoupX')

    # Set up R environment and transfer variables from Python to R
    ro.globalenv['data'] = data
    ro.globalenv['data_tod'] = data_tod
    ro.globalenv['genes'] = genes
    ro.globalenv['cells'] = cells
    ro.globalenv['soupx_groups'] = soupx_groups

    # Run R code using rpy2
    ro.r.source(os.path.join(os.path.dirname(__file__), 'ambient_rna_correction.R'))

    # Retrieve results from R to Python
    out = ro.globalenv['out']

    # Update adata
    adata.layers['counts'] = adata.X
    adata.layers['soupX_counts'] = out.T
    adata.X = adata.layers['soupX_counts']

    return adata


def filter_uninformative_genes(adata: sc.AnnData,
                               threshold: Tuple[Union[int, float], str] = (20, 'n_cells'),
                               verbosity: int = 0) -> sc.AnnData:
    """
    Filter out uninformative genes from an AnnData object based on expression thresholds.

    This function filters genes using one of three modes: minimum number of cells
    expressing the gene, percentage of cells expressing the gene, or total gene count quantile.
    The filtered genes are removed from the AnnData object.

    Args:
        adata (sc.AnnData): The input AnnData object.
        threshold (Tuple[Union[int, float], str], optional): Filtering threshold and mode.
            Modes include 'n_cells' (min number of cells in which gene is expressed, passed as an integer),
            'percent_cells' (min percentage of cells in which gene is expressed, passed as a decimal in [0,1]),
            and 'gene_count_quantile' (quantile of gene counts, passed as a decimal in [0,1]).
            Defaults to (20, 'n_cells').
        verbosity (int, optional): Level of logging. Defaults to 0.

    Returns:
        sc.AnnData: The AnnData object after gene filtering.
    """

    # (5, 'n_cells'), (0.001, 'percent_cells'), (0.1, 'gene_count_quantile')

    thresh = threshold[0]
    mode = threshold[1]

    if verbosity:
        print('### Filtering uninformative genes ... ###')
        n_genes_before = adata.n_vars
        print(f'# Number of genes before filtering: {n_genes_before}')

    # Filter by number of cells in which gene is expressed
    if mode == 'n_cells':
        sc.pp.filter_genes(adata, min_cells=thresh)

    # Filter by percentage of cells in which gene is expressed
    if mode == 'percent_cells':
        n_cells_perc_thresh = np.ceil(adata.n_obs * thresh)
        sc.pp.filter_genes(adata, min_cells=n_cells_perc_thresh)

    # Filter by total count per gene, q is quantile of total count distribution across genes
    if mode == 'gene_count_quantile':
        counts_per_gene = anndata_to_numpy(adata).sum(axis=0)
        q = np.quantile(counts_per_gene, threshold, method='nearest')
        keep_bool = (counts_per_gene > q)
        adata = adata[:, keep_bool]

    if verbosity >= 1:
        print(f'# Number of genes after filtering: {adata.n_vars}')
        print(f'# Number of genes removed due to low overall count: {n_genes_before - adata.n_vars}')

    return adata


def quality_control(adata: sc.AnnData,
                    species: str = 'mus musculus',
                    cor_amb_rna: bool = True,  # Only to be used, if raw data is from droplet based technology
                    pct_counts_mt_threshold: float = 8.0,
                    gene_expr_threshold: Tuple[Union[int, float], str] = (20, 'n_cells'),
                    verbosity: int = 0,
                    plot: bool = False) -> sc.AnnData:
    # Following best practices according to
    # https://www.sc-best-practices.org/preamble.html
    """
    Perform quality control on an AnnData object.

    This function filters out low-quality cells based on mitochondrial RNA content, removes
    uninformative genes, and optionally corrects for ambient RNA.

    Following best practices according to: https://www.sc-best-practices.org/preamble.html (07.10.2024)

    Args:
        adata (sc.AnnData): The input AnnData object.
        species (str, optional): The species of the dataset, either 'mus musculus' or 'homo sapiens'.
            Defaults to 'mus musculus'.
        cor_amb_rna (bool, optional): Whether to correct for ambient RNA (set to True for droplet-based technologies).
            Defaults to True.
        pct_counts_mt_threshold (float, optional): Mitochondrial gene expression threshold for filtering
            low-quality cells. Must be percentage in [0.0, 100.0]. Defaults to 8.0.
        gene_expr_threshold (Tuple[Union[int, float], str], optional): Filtering threshold and mode for filtering
            uninformative genes. Modes include:
             'n_cells' (min number of cells in which gene is expressed, passed as an integer),
            'percent_cells' (min percentage of cells in which gene is expressed, passed as a decimal in [0,1]),
            and 'gene_count_quantile' (quantile of gene counts, passed as a decimal in [0,1]).
            Defaults to (20, 'n_cells').
        verbosity (int, optional): Level of logging. Defaults to 0.
        plot (bool, optional): Whether to plot QC metrics. Defaults to False.

    Returns:
        sc.AnnData: The AnnData object after quality control.
    """
    assert species in {'mus musculus', 'homo sapiens'}, "'species' can be 'mus musculus', 'homo sapiens'"
    # Store original AnnData
    og_adata = adata.copy()
    # Filter cells with low quality reads
    adata = filter_low_quality_reads(adata=adata,
                                     species=species,
                                     pct_counts_mt_threshold=pct_counts_mt_threshold,
                                     verbosity=verbosity,
                                     plot=plot)
    if cor_amb_rna:
        # Correct for ambient RNA
        adata = correct_for_ambient_rna(adata=adata,
                                        unfiltered_adata=og_adata,
                                        verbosity=verbosity)

    # In tutorial: 20 / 14814 = 0,00135007425408397461860402322128
    # -> use less aggressive threshold of 5 (only have ~1000 cells in data set -> 5 / 1000 = 0.005)
    adata = filter_uninformative_genes(adata=adata,
                                       threshold=gene_expr_threshold,
                                       verbosity=verbosity)

    return adata


def normalize(adata: sc.AnnData,
              plot: bool = False) -> sc.AnnData:
    """
    Normalize and log-transform gene expression data in an AnnData object.

    This function normalizes each cell by its total counts, ensuring all cells have the same
    total count. The normalized data is stored in the 'norm' layer, and a log1p-transformed
    version is stored in the 'log1p_norm' layer. Optionally, histograms of total counts and
    log-transformed counts can be plotted.

    Args:
        adata (sc.AnnData): The input AnnData object.
        plot (bool, optional): Whether to plot histograms of the counts before and after
            log transformation. Defaults to False.

    Returns:
        sc.AnnData: The AnnData object with normalized and log-transformed data in new layers.
    """
    # Normalize each cell by total counts over all genes -> every cell has the same total count after normalization
    scaled_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
    adata.layers['norm'] = scaled_counts['X']
    # log1p transform
    adata.layers['log1p_norm'] = sc.pp.log1p(scaled_counts['X'], copy=True)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        p1 = sns.histplot(adata.X.sum(1), bins=100, kde=False, ax=axes[0])
        axes[0].set_title('Total counts')
        p2 = sns.histplot(adata.layers['log1p_norm'].sum(1), bins=100, kde=False, ax=axes[1])
        axes[1].set_title('Shifted logarithm')
        plt.show()

    return adata


def scale_to_unit_variance(adata: sc.AnnData,
                           layer_key: Union[str, None] = None,
                           plot: bool = False) -> sc.AnnData:
    # Scale gene-columns to unit variance (no 0 centering)
    # => needed if GENIE/GRNboost2 is used for GRN inference!!!

    """
    Scale gene expression data to unit variance in an AnnData object.

    This function scales the gene expression values in the specified layer or
    the main expression matrix (`adata.X`) to unit variance without zero-centering.
    The scaled data is stored in a new layer. Optionally, it plots the sum of gene
    expressions before and after scaling.

    Args:
        adata (sc.AnnData): The input AnnData object.
        layer_key (Union[str, None], optional): The layer to scale. Defaults to the main expression matrix if None.
        plot (bool, optional): Whether to plot histograms of gene expression sums. Defaults to False.

    Returns:
        sc.AnnData: The AnnData object with the scaled data added as a new layer.
    """

    # Get data from desired layer
    if layer_key is None:
        x = adata.X.copy()
        layer_key = 'counts'
    else:
        x = adata.layers[layer_key].copy()

    adata.layers[f'scaled_{layer_key}'] = sc.pp.scale(x, zero_center=False, copy=True)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        p1 = sns.histplot(x.sum(1), bins=100, kde=False, ax=axes[0])
        axes[0].set_title(layer_key)
        p2 = sns.histplot(adata.layers[f'scaled_{layer_key}'].sum(1), bins=100, kde=False, ax=axes[1])
        axes[1].set_title(f'scaled_{layer_key}')
        plt.show()

    return adata


def annotate_highly_deviant_genes(adata: sc.AnnData,
                                  top_k: int = 6000,
                                  verbosity: int = 0,
                                  plot: bool = False) -> sc.AnnData:
    """
    Annotate the top-k highly deviant genes in an AnnData object.

    This function uses an R-based method to compute binomial deviance for
    each gene and annotates the top-k most deviant genes. Optionally,
    it plots gene dispersions vs means.

    Args:
        adata (sc.AnnData): The input AnnData object.
        top_k (int, optional): Number of top deviant genes to annotate. Defaults to 6000.
        verbosity (int, optional): Logging level. Defaults to 0.
        plot (bool, optional): Whether to plot the gene dispersions. Defaults to False.

    Returns:
        sc.AnnData: The AnnData object with highly deviant gene annotations.
    """

    if verbosity >= 1:
        print('### Annotating highly deviant genes ... ###')

    # Activate R interface and set up logging
    anndata2ri.activate()
    pandas2ri.activate()
    rcb.logger.setLevel(logging.ERROR)

    # Set up R environment and transfer raw counts from Python to R
    ro.globalenv["adata"] = adata

    # Run R code using rpy2
    ro.r.source(os.path.join(os.path.dirname(__file__), 'deviant_genes_annotation.R'))

    # Retrieve results from R to Python
    binomial_deviance = ro.r("rowData(sce)$binomial_deviance").T

    # Annotate top-k most deviant genes
    idx = binomial_deviance.argsort()[-top_k:]
    mask = np.zeros(adata.var_names.shape, dtype=bool)
    mask[idx] = True

    # Annotate anndata object
    adata.var["highly_deviant"] = mask
    adata.var["binomial_deviance"] = binomial_deviance

    if plot:
        try:
            # Compute the mean and dispersion for each gene across all cells.
            sc.pp.highly_variable_genes(adata, layer='log1p_norm')
            # Plot dispersions vs mean, color by 'highly_deviant'
            ax = sns.scatterplot(
                data=adata.var, x="means", y="dispersions", hue="highly_deviant", s=5
            )
            ax.set_xlim(None, 1.5)
            ax.set_ylim(None, 3)
            plt.show()

        except KeyError:
            print("WARNING: layer 'log1p_norm' does not exist")

    return adata


def impute_data_magic(adata: sc.AnnData,
                      diffusion_time: Union[int, str] = 1,  # 'auto'
                      layer_key: str = 'log1p_norm',
                      verbosity: int = 0) -> sc.AnnData:
    """
    Perform MAGIC imputation on an AnnData object.
    This function runs the MAGIC algorithm to impute missing values using
    a specified layer or the main expression matrix. The imputed data is
    stored in a new layer called 'magic_imputed'.

    Args:
        adata (sc.AnnData): The AnnData object to impute.
        diffusion_time (Union[int, str], optional): The diffusion time for MAGIC.
            Defaults to 1.
        layer_key (str, optional): The layer to use for imputation. Defaults to 'log1p_norm'.
        verbosity (int, optional): Level of logging. Defaults to 0.

    Returns:
        sc.AnnData: The AnnData object with MAGIC-imputed values added as a new layer.
    """
    if verbosity >= 1:
        print('### Computing MAGIC imputation ... ###')
    # Make copy of adata
    bdata = adata.copy()

    # Set layer as bdata.X
    try:
        bdata.X = bdata.layers[layer_key]
    except KeyError:
        print(f"WARNING: Layer '{layer_key}' does not exist, proceeding with adata.X instead")

    # Run MAGIC with default parameter except for diffusion time t
    sce.pp.magic(adata=bdata,
                 name_list='all_genes',
                 knn=5,
                 decay=1,
                 knn_max=None,
                 t=diffusion_time,
                 n_pca=100,
                 solver='exact',
                 random_state=None,
                 n_jobs=None,
                 verbose=(verbosity >= 1),
                 copy=None)

    # Add new layer to adata
    adata.layers['magic_imputed'] = bdata.X.copy()

    return adata


def additional_calculations(adata: sc.AnnData) -> sc.AnnData:
    # Annotate highly variable genes
    """
    Perform some additional analyses and calculate embeddings and dimensionality reductions on an AnnData object.

    Args:
        adata (sc.AnnData): The input AnnData object to perform calculations on.

    Returns:
        sc.AnnData: The AnnData object with computed results stored in respective annotation fields and layers.

    """
    try:
        sc.pp.highly_variable_genes(adata, layer='log1p_norm')
    except KeyError:
        sc.pp.highly_variable_genes(adata)
        print("WARNING: layer 'log1p_norm' not found, using adata.X instead")

    # Compute PCA
    sc.pp.pca(adata, n_comps=50, use_highly_variable=False)

    # Compute KNN-graph
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')

    # Compute t-SNE
    sc.tl.tsne(adata, use_rep='X_pca')

    # Compute UMAP embedding
    sc.tl.umap(adata)

    # Compute Diffusion Map embedding
    sc.tl.diffmap(adata)

    # Compute force-directed graph drawing embedding
    sc.tl.draw_graph(adata)

    return adata


def data_pipeline(adata: sc.AnnData,
                  res_path: Union[None, str] = None,
                  obs_subset_keys: Union[Tuple[str, list[str]], None] = None,
                  species: str = 'mus musculus',  # 'homo sapiens', 'mus musculus'
                  pct_counts_mt_threshold: float = 8.0,
                  cor_amb_rna: bool = True,  # Only use if Dropseq technology was used
                  gene_expr_threshold: Tuple[Union[int, float], str] = (20, 'n_cells'),
                  top_k_deviant: int = 6000,
                  diffusion_time: int = 1,
                  verbosity: int = 0,
                  plot: bool = False) -> sc.AnnData:
    """
    Preprocessing pipeline according to best practices for single cell RNA-seq data stored in an AnnData object,
    including quality control, normalization, scaling and imputation, and gene annotation. Alterations to the
    data matrix are stored in new layers such that the original and intermediate data matrices remain available.
    Also, highly deviant genes are annotated as well as additional calculations such as KNN-graph computation,
    PCA, UMAP-embedding, ... are performed. The processed object is saved if a result path is provided.

    Args:
        adata (sc.AnnData): The input AnnData object containing the single-cell data to be processed.
        res_path (Union[None, str], optional): Path to save the resulting AnnData object.
            If None, the object is not saved. Defaults to None.
        obs_subset_keys (Union[Tuple[str, list[str]], None], optional): A tuple specifying the
            key in `adata.obs` to subset on and the values to keep. If None, no subsetting
            is performed. Defaults to None.
        species (str, optional): The species being studied. Options are 'mus musculus' or
            'homo sapiens'. Defaults to 'mus musculus'.
        pct_counts_mt_threshold (float, optional): The mitochondrial gene expression threshold
            used during quality control. Cells with a percentage of mitochondrial counts above
            this threshold are filtered out. Defaults to 8.0.
        cor_amb_rna (bool, optional): Whether to correct ambiguous RNA molecules. This should
            be set to True if the data was generated using the Dropseq technology. Defaults to True.
        gene_expr_threshold (Tuple[Union[int, float], str], optional): A tuple specifying the
            threshold for filtering low-expressed genes. The first element defines the threshold,
            and the second specifies the metric. Options are 'n_cells' and an integer, 'percent_cells'
            and a percentage passed as a fraction in [0, 1], 'gene_count_quantile' and a quantile in [0,1].
            Defaults to (20, 'n_cells').
        top_k_deviant (int, optional): Number of top deviant genes to annotate based on variance
            or other criteria. Defaults to 6000.
        diffusion_time (int, optional): The diffusion time parameter for MAGIC-based imputation.
            Defaults to 1.
        verbosity (int, optional): Controls the level of output messages during processing. Set
            to 0 for no output, 1 for basic output, and higher for more detailed logging. Defaults to 0.
        plot (bool, optional): Whether to generate plots during the processing steps. Defaults to False.

    Returns:
        sc.AnnData: The processed AnnData object containing filtered, normalized, scaled, and imputed data.

    Raises:
        ValueError: If expected keys or data formats are not present in the AnnData object.
    """
    # Subset Anndata if any keys and values to keep are passed
    if obs_subset_keys is not None:
        adata = subset_obs(adata=adata,
                           obs_key=obs_subset_keys[0],
                           keep_vals=obs_subset_keys[1])

    # Perform quality control
    adata = quality_control(adata=adata,
                            species=species,
                            pct_counts_mt_threshold=pct_counts_mt_threshold,
                            cor_amb_rna=cor_amb_rna,
                            gene_expr_threshold=gene_expr_threshold,
                            verbosity=verbosity,
                            plot=plot)

    # Normalize data (cellcount / total count, log1p-transform)
    adata = normalize(adata=adata, plot=plot)

    # Scale data to unit variance and add as additional layers
    adata = scale_to_unit_variance(adata=adata, layer_key=None, plot=plot)
    adata = scale_to_unit_variance(adata=adata, layer_key='norm', plot=plot)
    adata = scale_to_unit_variance(adata=adata, layer_key='log1p_norm', plot=plot)

    # Compute imputed data matrix
    adata = impute_data_magic(adata=adata,
                              diffusion_time=diffusion_time,
                              layer_key='log1p_norm',
                              verbosity=verbosity)

    # Annotate highly deviant genes
    adata = annotate_highly_deviant_genes(adata=adata, top_k=top_k_deviant, verbosity=verbosity, plot=plot)

    # Perform additional calculations
    adata = additional_calculations(adata=adata)

    if res_path is not None:
        adata.write_h5ad(filename=Path(res_path))

    if verbosity >= 1:
        print('### Preprocessed AnnData object')
        print(adata)

    return adata


# Auxiliary functions ##################################################################################################
def subset_obs(adata: sc.AnnData,
               obs_key: str,
               keep_vals: list[Any]) -> sc.AnnData:
    """
   Subset an AnnData object based on values in a specific observation key.

   This function filters the AnnData object by retaining only the cells (rows)
   where the values in the specified observation key (`obs_key`) match one of
   the values in the `keep_vals` list.

   Args:
       adata (sc.AnnData): The input AnnData object containing the single-cell data to subset.
       obs_key (str): The key in `adata.obs` to subset by. This should correspond to a column
           in the observation metadata (adata.obs).
       keep_vals (list[Any]): A list of values to keep. Only rows (cells) where the value in
           `obs_key` matches one of these values will be retained.

   Returns:
       sc.AnnData: The subsetted AnnData object containing only the cells that match the
       specified `keep_vals` in the `obs_key` column.
   """
    adata = adata[np.isin(adata.obs[obs_key].array, keep_vals), :]

    return adata


def is_outlier(adata: sc.AnnData,
               obs_key_qc_metric: str,
               nmads: int = 5):
    """
    Identify outliers in a QC metric using Median Absolute Deviations, MAD = median(|x_i - median(x)|).

    This function flags outliers in a given observation key (`obs_key_qc_metric`) of an AnnData object
    based on a threshold determined by the number of median absolute deviations (MAD) from the median.

    Args:
        adata (sc.AnnData): The AnnData object containing the QC metric to analyze.
        obs_key_qc_metric (str): The key in `adata.obs` corresponding to the QC metric to check for outliers.
        nmads (int, optional): The number of MADs from the median to define an outlier. Defaults to 5.

    Returns:
        pd.Series: A boolean series where True indicates an outlier.
    """
    # Uses Median Absolute Deviations as outlier criterion for QC-metric x_i, x
    # MAD = median(|x_i - median(x)|)
    m = adata.obs[obs_key_qc_metric]
    outlier = (m < np.median(m) - nmads * median_abs_deviation(m)) | \
              (np.median(m) + nmads * median_abs_deviation(m) < m)

    return outlier
