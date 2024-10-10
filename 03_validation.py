
import scanpy as sc
import numpy as np
import scipy as sci
import pandas as pd
import os
import glob
import json

from pathlib import Path
from typing import *

from switchtfi.data import preendocrine_alpha, preendocrine_beta, erythrocytes
from switchtfi.utils import load_grn_json


def main_pseudotime_inference():
    # ### Script for inferring a pseudotime value for each cell in the dataset
    from validation.pseudotime_inference import calculate_palantir_pt
    from switchtfi.utils import csr_to_numpy
    # ### Load the previously preprocessed scRNA-seq data stored as an AnnData object
    # (also available via the SwitchTFI functions)
    adata = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
    bdata = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
    cdata = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')
    # adata = preendocrine_alpha()
    # bdata = preendocrine_beta()
    # cdata = erythrocytes()

    # ### Determine the root cells for pseudotime inference by the expression of well known marker genes
    # - Pre-endocrine alpha cell transition data: high expression of Fev
    # - Pre-endocrine beta cell transition data: high expression of Fev
    # - Erythrocytes differentiation data: low expression of Gata1

    plot = True
    if plot:
        sc.pl.umap(adata, color='Fev')
        sc.pl.umap(adata, color='clusters')
        sc.pl.umap(bdata, color='Fev')
        sc.pl.umap(bdata, color='clusters')
        sc.pl.umap(cdata, color='Gata1')
        sc.pl.umap(cdata, color='paul15_clusters')

    afev_expression = csr_to_numpy(adata[:, 'Fev'].X).flatten()
    apreend_bool = (adata.obs['clusters'].to_numpy() == 'Pre-endocrine')
    apreend_fev_expression = np.ma.masked_equal(afev_expression, apreend_bool)
    aroot = int(np.argmax(apreend_fev_expression))

    bfev_expression = csr_to_numpy(bdata[:, 'Fev'].X).flatten()
    bpreend_bool = (bdata.obs['clusters'].to_numpy() == 'Pre-endocrine')
    bpreend_fev_expression = np.ma.masked_equal(bfev_expression, bpreend_bool)
    broot = int(np.argmax(bpreend_fev_expression))

    gata1_expression = cdata[:, 'Gata1'].X.flatten()
    mep_bool = (cdata.obs['paul15_clusters'].to_numpy() == '7MEP')
    mep_gata1_expression = np.ma.masked_equal(gata1_expression, mep_bool)
    eryroot = int(np.argmin(mep_gata1_expression))

    # ### Calculate cell wise pseudotime values with the Palantir method
    adata = calculate_palantir_pt(adata=adata, root=aroot, layer_key='log1p_norm',
                                  cluster_obs_key='clusters', plot=plot)
    bdata = calculate_palantir_pt(adata=bdata, root=broot, layer_key='log1p_norm',
                                  cluster_obs_key='clusters', plot=plot)
    cdata = calculate_palantir_pt(adata=cdata, root=eryroot, layer_key='log1p_norm',
                                  cluster_obs_key='prog_off', plot=plot)

    # ### Save results
    adata.write_h5ad(filename=Path('./results/03_validation/anndata/pt_pre-endocrine_alpha.h5ad'))
    bdata.write_h5ad(filename=Path('./results/03_validation/anndata/pt_pre-endocrine_beta.h5ad'))
    cdata.write_h5ad(filename=Path('./results/03_validation/anndata/pt_erythrocytes.h5ad'))


def main_switchde_analysis():
    # ### Script for determining differential gene expression along a previously calculated pseudotemporal trajectory

    # ### Set the path to the R version installed in the Conda environment
    conda_env_path = '/data/bionets/ac07izid/miniconda3/envs/dtfi'
    r_home_path = os.path.join(conda_env_path, "lib", "R")
    # Set the R_HOME environment variable
    os.environ["R_HOME"] = r_home_path
    # Import after setting R version
    from validation.switchde import calculate_switch_de_pvalues

    # ### Load the previously preprocessed scRNA-seq data stored as an AnnData object
    # (also available via the SwitchTFI functions)
    adata = sc.read_h5ad('./results/03_validation/anndata/pt_pre-endocrine_alpha.h5ad')
    bdata = sc.read_h5ad('./results/03_validation/anndata/pt_pre-endocrine_beta.h5ad')
    erydata = sc.read_h5ad('./results/03_validation/anndata/pt_erythrocytes.h5ad')

    # ### Perform the differential expression analysis with the switchde method
    data_list = [adata, bdata, erydata]
    fn_list = ['pre-endocrine_alpha', 'pre-endocrine_beta', 'erythrocytes']
    for i in [0, 1, 2]:

        resdata, res = calculate_switch_de_pvalues(adata=data_list[i].copy(),
                                                   zero_inflated=False,
                                                   layer_key='magic_imputed',
                                                   verbosity=1)
        res.to_csv(
            f'./results/03_validation/anndata/switchde_magic_nozeroinflated_{fn_list[i]}.csv')
        resdata.write_h5ad(Path(
            f'./results/03_validation/anndata/switchde_magic_nozeroinflated_{fn_list[i]}.h5ad'))

        resdata, res = calculate_switch_de_pvalues(adata=data_list[i].copy(),
                                                   zero_inflated=True,
                                                   layer_key='log1p_norm',
                                                   verbosity=1)
        res.to_csv(
            f'./results/03_validation/anndata/switchde_log1p_norm_zeroinflated_{fn_list[i]}.csv')
        resdata.write_h5ad(Path(
            f'./results/03_validation/anndata/switchde_log1p_norm_zeroinflated_{fn_list[i]}.h5ad'))


def main_trend_calculation():
    # ### Script for calculating gene trends in pseudotime
    from validation.trend_calculation import calculate_pygam_gene_trends

    # ### Load AnnData objects containing scRNA-seq data for which pseudotime values were previously computed
    adata = sc.read_h5ad('./results/03_validation/anndata/pt_pre-endocrine_alpha.h5ad')
    bdata = sc.read_h5ad('./results/03_validation/anndata/pt_pre-endocrine_beta.h5ad')
    erydata = sc.read_h5ad('./results/03_validation/anndata/pt_erythrocytes.h5ad')

    # ### Calculate gene trends in pseudotime with the pygam implementation of generative additive models (GAMs)
    adata = calculate_pygam_gene_trends(adata=adata,
                                        gene_names=None,
                                        n_splines=4,
                                        spline_order=2,
                                        pseudotime_obs_key='palantir_pseudotime',
                                        trend_resolution=200,
                                        layer_key='magic_imputed')
    bdata = calculate_pygam_gene_trends(adata=bdata,
                                        gene_names=None,
                                        n_splines=4,
                                        spline_order=2,
                                        pseudotime_obs_key='palantir_pseudotime',
                                        trend_resolution=200,
                                        layer_key='magic_imputed')
    erydata = calculate_pygam_gene_trends(adata=erydata,
                                          gene_names=None,
                                          n_splines=4,
                                          spline_order=2,
                                          pseudotime_obs_key='palantir_pseudotime',
                                          trend_resolution=200,
                                          layer_key='magic_imputed')

    adata.write_h5ad(Path('./results/03_validation/anndata/trend_pre-endocrine_alpha.h5ad'))
    bdata.write_h5ad(Path('./results/03_validation/anndata/trend_pre-endocrine_beta.h5ad'))
    erydata.write_h5ad(Path('./results/03_validation/anndata/trend_erythrocytes.h5ad'))


def main_save_cellrank_driver_genes():
    # ### Script for computing and saving transition driver genes with the CellRank method
    from validation.cellrank_workflow import get_cellrank_driver_genes

    # ### Load the previously preprocessed scRNA-seq data stored as an AnnData object
    # (also available via the SwitchTFI functions)
    adata = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
    bdata = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
    erydata = sc.read_h5ad('./results/03_validation/anndata/pt_erythrocytes.h5ad')
    # adata = preendocrine_alpha()
    # bdata = preendocrine_beta()
    # erydata = erythrocytes()

    # ### Start analysis with CellRank
    plot = True
    atop_k_list, ares_df = get_cellrank_driver_genes(adata=adata,
                                                     top_k=None,
                                                     compute_velocity=True,
                                                     scvelo_pp=True,
                                                     layer_key=None,
                                                     initial_terminal_state=('Pre-endocrine', 'Alpha'),
                                                     plot=plot)
    btop_k_list, bres_df = get_cellrank_driver_genes(adata=bdata,
                                                     top_k=None,
                                                     compute_velocity=True,
                                                     scvelo_pp=True,
                                                     layer_key=None,
                                                     initial_terminal_state=('Pre-endocrine', 'Beta'),
                                                     plot=plot)
    print('### Alpha')
    print(atop_k_list)
    print('### Beta')
    print(btop_k_list)

    res_p = './results/03_validation/driver_genes'

    ares_df.to_csv(os.path.join(res_p, 'alpha_cellrank_driver_genes.csv'))
    bres_df.to_csv(os.path.join(res_p, 'beta_cellrank_driver_genes.csv'))

    # ### For the erythrocytes dataset there are no un-/spliced counts available,
    # use CellRank pseudotime kernel instead
    import cellrank as cr
    from validation.cellrank_workflow import identify_initial_terminal_states, estimate_fate_probabilities, \
        uncover_driver_genes

    # Set log transformed data, set as main data matrix of anndata, keep only 2000 highly variable genes
    erydata.layers['dummy'] = erydata.X.copy()
    erydata.X = erydata.layers['log1p_norm'].copy()

    # Use pseudotime to compute cellrank transition matrix
    pk = cr.kernels.PseudotimeKernel(erydata, time_key='palantir_pseudotime')
    pk.compute_transition_matrix()
    pk.plot_projection(basis='X_umap', color='palantir_pseudotime', legend_loc='right')

    # Compute terminal and initial states using CellRanks Generalized Perron Cluster Cluster Analysis (GPCCA) estimator
    gpcca_estimator = identify_initial_terminal_states(cr_kernel=pk, cluster_obs_key='prog_off',
                                                       initial_terminal_state=('prog', 'off'), plot=plot)

    # Compute fate probabilities, correlate fate probabilities with gene expression => Driver genes
    gpcca_estimator = estimate_fate_probabilities(cr_estimator=gpcca_estimator, plot=plot)
    eryres_df, gpcca_estimator = uncover_driver_genes(cr_estimator=gpcca_estimator, verbosity=0)

    eryres_df.reset_index(inplace=True, names='gene')
    erytop_k_list = eryres_df['gene'].to_list()

    print('### Erythrocytes')
    print(erytop_k_list)

    eryres_df.to_csv(os.path.join(res_p, 'ery_cellrank_driver_genes.csv'))


def main_save_splicejac_driver_genes():
    # ### Script for computing and saving transition driver genes with the spliceJAC method
    from validation.splicejac_workflow import get_splicejac_driver_genes, extract_splicejac_grn
    # ### Load the previously preprocessed scRNA-seq data stored as an AnnData object
    # (also available via the SwitchTFI functions)
    adata = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
    bdata = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
    # adata = preendocrine_alpha()
    # bdata = preendocrine_beta()

    # ### Start spliceJAC analysis
    # Define upper bound (= 0.9 * number of cells in cluster) for the number of genes
    # that are included in the spliceJAC analysis.
    # print(adata.obs['clusters'].value_counts())
    # print(bdata.obs['clusters'].value_counts())
    # pre-e, alpha: 565, 339 -> 0.9 * 339 = 305,1;  pre-e, beta: 556, 446 -> 0.9 * 446 = 401,4
    # See tutorial: https://splicejac.readthedocs.io/en/latest/notebooks/GRN%20Inference.html (07.10.2024)
    atop_n_jacobian = 300
    btop_n_jacobian = 400

    atop_k_list, ares_df, adata = get_splicejac_driver_genes(adata=adata,
                                                             top_k=None,
                                                             top_n_jacobian=atop_n_jacobian,
                                                             layer_key=None,
                                                             splicejac_pp=True,
                                                             compute_velocity=True,
                                                             cluster_pair=('Pre-endocrine', 'Alpha'))
    btop_k_list, bres_df, bdata = get_splicejac_driver_genes(adata=bdata,
                                                             top_k=None,
                                                             top_n_jacobian=btop_n_jacobian,
                                                             layer_key=None,
                                                             splicejac_pp=True,
                                                             compute_velocity=True,
                                                             cluster_pair=('Pre-endocrine', 'Beta'))
    print('### Alpha')
    print(atop_k_list)
    print('### Beta')
    print(btop_k_list)

    ares_df.to_csv('./results/03_validation/driver_genes/alpha_splicejac_driver_genes.csv')
    bres_df.to_csv('./results/03_validation/driver_genes/beta_splicejac_driver_genes.csv')

    # ### Extract GRN from spliceJACs results
    for q in [0.1, 0.25, 0.5, 0.75, 0.95]:
        agrn = extract_splicejac_grn(adata=adata,
                                     grn_adj_uns_key='average_jac',
                                     clusters=('Pre-endocrine', 'Alpha'),
                                     weight_quantile=q)
        bgrn = extract_splicejac_grn(adata=bdata,
                                     grn_adj_uns_key='average_jac',
                                     clusters=('Pre-endocrine', 'Beta'),
                                     weight_quantile=q)

        grn_res_p = './results/03_validation/driver_genes/spjc_grns'
        agrn.to_csv(os.path.join(grn_res_p, f'alpha_q{q}_spjc_grn.csv'))
        bgrn.to_csv(os.path.join(grn_res_p, f'beta_q{q}_spjc_grn.csv'))


def main_save_drivaer_driver_genes():
    # ### Script for computing and saving transition driver genes with the DrivAER method

    # ### Data conversion
    # Since DrivAER relies on an outdated version of AnnData,
    # it is not compatible with the previously saved AnnData objects from preprocessing.
    # We save the data matrices as numpy arrays and reconstruct the original AnnnData object
    # using the deprecated version.

    def save_new_anndata_as_np(ad: sc.AnnData,
                               layer: Union[str, None] = None,
                               cluster_obs_key: str = 'clusters',
                               prefix: str = 'alpha',
                               rp: Union[str, None] = './results/03_validation/driver_genes_drivaer_aux_data'):

        if layer is None:
            data_mtrx = ad.X
        else:
            data_mtrx = ad.layers[layer]
        if isinstance(data_mtrx, sci.sparse.csr_matrix):
            data_mtrx = data_mtrx.toarray()

        print(data_mtrx)

        cell_anno_names = ad.obs_names.to_numpy()
        cell_anno_cluster = ad.obs[cluster_obs_key].to_numpy()
        # cell_anno_pt = ad.obs['palantir_pseudotime'].to_numpy()
        gene_anno_names = ad.var_names.to_numpy()

        if layer is None:
            layer = 'raw'

        np.savez(os.path.join(rp, f'np_{prefix}_{layer}.npz'),
                 data_mtrx=data_mtrx,
                 cell_anno_names=cell_anno_names,
                 cell_anno_cluster=cell_anno_cluster,
                 # cell_anno_pt=cell_anno_pt,
                 gene_anno_names=gene_anno_names)

    def load_np_save_old_anndata(layer: Union[str, None] = None,
                                 prefix: str = 'alpha',
                                 rp: Union[str, None] = './results/03_validation/driver_genes_drivaer_aux_data'):

        if layer is None:
            layer = 'raw'

        arrays = np.load(os.path.join(rp, f'np_{prefix}_{layer}.npz'), allow_pickle=True)

        data_mtrx = arrays['data_mtrx']
        cell_anno_names = arrays['cell_anno_names']
        cell_anno_cluster = arrays['cell_anno_cluster']
        # cell_anno_pt = arrays['cell_anno_pt']
        # cell_anno_pt = np.random.uniform(low=0, high=1, size=data_mtrx.shape[0])
        gene_anno_names = arrays['gene_anno_names']

        ad = sc.AnnData(data_mtrx)
        ad.obs_names = cell_anno_names
        ad.obs['clusters'] = cell_anno_cluster
        # ad.obs['palantir_pseudotime'] = cell_anno_pt
        ad.var_names = gene_anno_names

        ad.write_h5ad(filename=Path(os.path.join(rp, f'ad_{prefix}_{layer}.h5ad')))

    def correct_dtype(ad: sc.AnnData,
                      cluster_obs_key: str = 'clusters') -> sc.AnnData:
        ad.var_names = np.array([byte.decode('ascii') for byte in ad.var_names.to_numpy()])
        ad.obs[cluster_obs_key] = np.array([byte.decode('ascii') for byte in ad.obs[cluster_obs_key].to_numpy()])

        return ad

    # Define path where data is saved
    data_p = './results/03_validation/driver_genes/drivaer_aux_data'
    # Run with recent scanpy version to store AnnData object as separate numpy arrays
    save_new_to_np = False
    if save_new_to_np:
        # Load AnnData object with scRNAseq data, use previously preprocessed data or the switchtfi function
        adata = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
        bdata = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
        erydata = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')
        # adata = preendocrine_alpha()
        # bdata = preendocrine_beta()
        # cdata = erythrocytes()
        save_new_anndata_as_np(ad=adata, layer=None, prefix='alpha', rp=data_p)
        save_new_anndata_as_np(ad=bdata, layer=None, prefix='beta', rp=data_p)
        save_new_anndata_as_np(ad=erydata, layer=None, cluster_obs_key='prog_off', prefix='ery', rp=data_p)

    # Run with drivaer env (old scanpy version) to load stored numpy arrays and create an Anndata object from them
    load_np_save_old = False
    if load_np_save_old:
        load_np_save_old_anndata(layer=None, prefix='alpha', rp=data_p)
        load_np_save_old_anndata(layer=None, prefix='beta', rp=data_p)
        load_np_save_old_anndata(layer=None, prefix='ery', rp=data_p)

    # ### Run DrivAER analysis
    drivaer_analysis = True
    if drivaer_analysis:
        from validation.drivaer_workflow import get_drivaer_driver_genes

        # Load AnnDatas and correct dtype (screwed up when saving and loading with old scanpy version)
        adata = correct_dtype(sc.read_h5ad(os.path.join(data_p, 'ad_alpha_raw.h5ad')))
        bdata = correct_dtype(sc.read_h5ad(os.path.join(data_p, 'ad_beta_raw.h5ad')))
        erydata = correct_dtype(sc.read_h5ad(os.path.join(data_p, 'ad_ery_raw.h5ad')))

        ad_list = [adata, bdata, erydata]

        # Load GRNs
        agrn = pd.read_csv('./results/01_grn_inf/endocrine/alpha/ngrnthresh9_alpha_pyscenic_combined_grn.csv',
                           index_col=[0])
        bgrn = pd.read_csv('./results/01_grn_inf/endocrine/beta/ngrnthresh9_beta_pyscenic_combined_grn.csv',
                           index_col=[0])
        erygrn = pd.read_csv('./results/01_grn_inf/hematopoiesis/ngrnthresh9_erythrocytes_pyscenic_combined_grn.csv',
                             index_col=[0])

        grn_list = [agrn, bgrn, erygrn]

        name_list = ['alpha', 'beta', 'ery']

        for i in [0, 1, 2]:

            top_k_list, res_df, drivaer_out = get_drivaer_driver_genes(adata=ad_list[i],
                                                                       grn=grn_list[i],
                                                                       top_k=None,
                                                                       dim_red_method='dca',
                                                                       verbosity=1)

            print(top_k_list)

            res_df.to_csv(f'./results/03_validation/driver_genes/{name_list[i]}_drivaer_driver_genes.csv')


def main_sig_thresh_selection():

    # ### Script for determining the change in SwitchTFI's predicted top driver genes for different FWER thresholds

    # ### Define auxiliary function for selecting the FWER threshold hyperparameter
    from switchtfi.pvalue_calculation import remove_insignificant_edges
    from switchtfi.tf_ranking import rank_tfs
    from validation.val_utils import compare_gene_sets

    def hyper_param_selection_helper(grn: pd.DataFrame,
                                     thresh_list: List[float],
                                     ranking_method: str = 'pr',
                                     top_k: Tuple[int, ...] = (10,)):
        pruned_grn_list = []
        res_list = []

        # Iterate over FWER thresholds
        for alpha in thresh_list:

            # Prune GRN based on FWER threshold
            dummy_grn = remove_insignificant_edges(grn=grn.copy(deep=True), alpha=alpha, p_value_key='pvals_wy',
                                                   inplace=False)
            pruned_grn_list.append(dummy_grn)

            # Rank transcription factors based on centrality in pruned GRN (= transition GRN)
            if ranking_method == 'pr':
                dummy_res = rank_tfs(grn=dummy_grn, centrality_measure='pagerank')
            elif ranking_method == 'out_deg':
                dummy_res = rank_tfs(grn=dummy_grn, centrality_measure='out_degree', reverse=False, weight_key='score')
            else:
                dummy_res = pd.DataFrame()

            res_list.append(dummy_res)

        # Iterate over pruned GRNs and determine their sizes
        n_verts = []
        n_tfs = []
        n_targets = []
        n_edges = []
        for i, pg in enumerate(pruned_grn_list):
            print(f'###### Alpha = {thresh_list[i]}: ###### ')
            print(f'## n vertices: {np.unique(pg[["TF", "target"]].to_numpy()).shape[0]}')
            n_verts.append(np.unique(pg[["TF", "target"]].to_numpy()).shape[0])
            print(f'## n TFs: {np.unique(pg[["TF"]].to_numpy()).shape[0]}')
            n_tfs.append(np.unique(pg[["TF"]].to_numpy()).shape[0])
            print(f'## n targets: {np.unique(pg[["target"]].to_numpy()).shape[0]}')
            n_targets.append(np.unique(pg[["target"]].to_numpy()).shape[0])
            print(f'## n edges: {pg.shape[0]}')
            n_edges.append(pg.shape[0])

        grn_df = pd.DataFrame(index=thresh_list)
        grn_df['nvert'] = n_verts
        grn_df['ntf'] = n_tfs
        grn_df['ntarget'] = n_targets
        grn_df['nedges'] = n_edges

        print(grn_df)

        # For multiple ks determine their Jaccard-based similarity of the top-k ranked TFs
        mean_jis = []
        jis = []
        for k in top_k:
            a, b = compare_gene_sets(res_df_list=res_list, top_k=k)
            mean_jis.append(a)
            jis.append(b)
            print(f'###### Avg pairwise JI among top {k}: ######')
            print(f'## JI: {a}')

    # ### Load the unpruned GRNs that were saved as an intermediate result during SwitchTFI analyses
    agrn = load_grn_json(grn_path='./results/02_switchtfi/endocrine/alpha/grn.json')
    bgrn = load_grn_json(grn_path='./results/02_switchtfi/endocrine/beta/grn.json')
    erygrn = load_grn_json(grn_path='./results/02_switchtfi/hematopoiesis/grn.json')

    # ### Run hyperparameter selection analyses
    alphas = [0.05, 0.1, 0.2, 0.5]
    hyper_param_selection_helper(grn=agrn, thresh_list=alphas, ranking_method='pr', top_k=(1, 5, 10, 15, 20))
    hyper_param_selection_helper(grn=agrn, thresh_list=alphas, ranking_method='out_deg', top_k=(1, 5, 10, 15, 20))

    hyper_param_selection_helper(grn=bgrn, thresh_list=alphas, ranking_method='pr', top_k=(1, 5, 10, 15, 20))
    hyper_param_selection_helper(grn=bgrn, thresh_list=alphas, ranking_method='out_deg', top_k=(1, 5, 10, 15, 20))

    hyper_param_selection_helper(grn=erygrn, thresh_list=alphas, ranking_method='pr', top_k=(1, 5, 10, 15, 20))
    hyper_param_selection_helper(grn=erygrn, thresh_list=alphas, ranking_method='out_deg', top_k=(1, 5, 10, 15, 20))


def main_robustness_analysis():

    # ### Script for investigating the robustness of SwitchTFI's results w.r.t. the input GRN
    from switchtfi.fit import fit_model
    from switchtfi.tf_ranking import rank_tfs
    from validation.val_utils import compare_grns, compare_gene_sets

    # ### Run SwitchTFI analyses on the individual GRNs inferred with Scenic
    switchtfi_inference = False
    if switchtfi_inference:
        adata = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
        bdata = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
        erydata = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')

        def helper(ad: sc.AnnData,
                   grn_p: str,
                   res_p: str,
                   clustering_obs_key: str):
            grn_list = []
            # Get list of paths to csv files
            csv_files = glob.glob(grn_p + '/*_pruned_grn.csv')
            for csv_file in csv_files:
                grn_list.append(pd.read_csv(csv_file, index_col=[0]))
            for i, grn in enumerate(grn_list):
                transition_grn, _ = fit_model(adata=ad.copy(),
                                              grn=grn,
                                              layer_key='magic_imputed',
                                              clustering_obs_key=clustering_obs_key,
                                              result_folder=res_p,
                                              fn_prefix=os.path.basename(csv_files[i])[0:2],
                                              verbosity=1,
                                              save_intermediate=True)
                rank_tfs(grn=transition_grn,
                         centrality_measure='out_degree',
                         reverse=False,
                         weight_key='score',
                         result_folder=res_p,
                         fn_prefix=os.path.basename(csv_files[i])[0:2] + 'outdeg_')

        helper(ad=adata,
               grn_p='./results/01_grn_inf/endocrine/alpha',
               res_p='./results/03_validation/robustness/alpha',
               clustering_obs_key='clusters')
        helper(ad=bdata,
               grn_p='./results/01_grn_inf/endocrine/beta',
               res_p='./results/03_validation/robustness/beta',
               clustering_obs_key='clusters')
        helper(ad=erydata,
               grn_p='./results/01_grn_inf/hematopoiesis',
               res_p='./results/03_validation/robustness/erythrocytes',
               clustering_obs_key='prog_off')

    # ### Compare the similarity of the resulting transition GRNs and TF rankings
    # against the similarity of the input GRNs
    def comparison_helper(base_grn_p: str,
                          switchtfi_res_p: str):
        # Load base GRNs
        base_grn_list = []
        csv_files = glob.glob(base_grn_p + '/*_pruned_grn.csv')
        for csv_file in csv_files:
            base_grn_list.append(pd.read_csv(csv_file, index_col=[0]))

        # Load core GRNs
        grn_list = []
        grn_csv_files = glob.glob(switchtfi_res_p + '/*grn.csv')
        for csv_file in grn_csv_files:
            grn_list.append(pd.read_csv(csv_file, index_col=[0]))

        # Load result dataframes
        res_list = []
        res_csv_files = glob.glob(switchtfi_res_p + '/*ranked_tfs.csv')
        for csv_file in res_csv_files:
            res_list.append(pd.read_csv(csv_file, index_col=[0]))

        # Load result dataframes ranked by outdegree
        outdeg_res_list = []
        outdeg_res_csv_files = glob.glob(switchtfi_res_p + '/*outdeg_ranked_tfs.csv')
        for csv_file in outdeg_res_csv_files:
            outdeg_res_list.append(pd.read_csv(csv_file, index_col=[0]))

        base_grn_res = compare_grns(grn_list=base_grn_list)
        grn_res = compare_grns(grn_list=grn_list)
        res_res = compare_gene_sets(res_df_list=res_list, top_k=10)
        outdeg_res_res = compare_gene_sets(res_df_list=outdeg_res_list, top_k=10)

        print('###### Pyscenic GRNs')
        print(f'### vert: {base_grn_res[0]}, edg: {base_grn_res[1]}, tf: {base_grn_res[2]}, target: {base_grn_res[3]}')
        print('###### SwitchTFI GRNs')
        print(f'### vert: {grn_res[0]}, edg: {grn_res[1]}, tf: {grn_res[2]}, target: {grn_res[3]}')
        print('###### Top k TFs')
        print(f'### top 10: {res_res[0]}')
        print('###### Top k outdegree ranked TFs')
        print(f'### top 10: {outdeg_res_res[0]}')

    print('# ### Alpha')
    comparison_helper(base_grn_p='./results/01_grn_inf/endocrine/alpha',
                      switchtfi_res_p='./results/03_validation/robustness/alpha')
    print('# ### Beta')
    comparison_helper(base_grn_p='./results/01_grn_inf/endocrine/beta',
                      switchtfi_res_p='./results/03_validation/robustness/beta')
    print('# ### Erythrocytes')
    comparison_helper(base_grn_p='./results/01_grn_inf/hematopoiesis',
                      switchtfi_res_p='./results/03_validation/robustness/erythrocytes')


# Misc #################################################################################################################
def main_save_topk_genesets():

    def save_first_k_entries_to_file(d: pd.DataFrame,
                                     file_name: str,
                                     column_name: str = 'gene',
                                     k: int = 10) -> None:

        # Extract the first k entries from the specified column
        entries = d[column_name].head(k)

        # Save the entries to a file, each entry on a new line
        with open(file_name, 'w') as file:
            for entry in entries:
                file.write(f"{entry}\n")

    top_k = 20
    out_p = './results/03_validation/digest_results/gene_sets'

    # Save top-k driver genes of other methods to lists
    res_base_p = './results/03_validation/driver_genes/'
    res_fns = ['alpha_cellrank_driver_genes.csv', 'beta_cellrank_driver_genes.csv',
               'alpha_splicejac_driver_genes.csv', 'beta_splicejac_driver_genes.csv',
               'alpha_drivaer_driver_genes.csv', 'beta_drivaer_driver_genes.csv',
               'ery_cellrank_driver_genes.csv', 'ery_drivaer_driver_genes.csv', ]
    for fn in res_fns:
        df = pd.read_csv(os.path.join(res_base_p, fn), index_col=[0])
        out_fn = os.path.join(out_p, f'top{top_k}_' + fn[:-4] + '.txt')
        save_first_k_entries_to_file(d=df, file_name=out_fn, column_name='gene', k=top_k)

    # Save top-k driver genes of SwitchTFI to list
    a_df = pd.read_csv('./results/02_switchtfi/endocrine/alpha/ranked_tfs.csv', index_col=[0])
    out_fn = os.path.join(out_p, f'top{top_k}_alpha_switchtfi_driver_genes.txt')
    save_first_k_entries_to_file(d=a_df, file_name=out_fn, column_name='gene', k=top_k)

    a_df = pd.read_csv('./results/02_switchtfi/endocrine/alpha/outdeg_ranked_tfs.csv', index_col=[0])
    out_fn = os.path.join(out_p, f'top{top_k}_alpha_switchtfi_outdeg_driver_genes.txt')
    save_first_k_entries_to_file(d=a_df, file_name=out_fn, column_name='gene', k=top_k)

    b_df = pd.read_csv('./results/02_switchtfi/endocrine/beta/ranked_tfs.csv', index_col=[0])
    out_fn = os.path.join(out_p, f'top{top_k}_beta_switchtfi_driver_genes.txt')
    save_first_k_entries_to_file(d=b_df, file_name=out_fn, column_name='gene', k=top_k)

    b_df = pd.read_csv('./results/02_switchtfi/endocrine/beta/outdeg_ranked_tfs.csv', index_col=[0])
    out_fn = os.path.join(out_p, f'top{top_k}_beta_switchtfi_outdeg_driver_genes.txt')
    save_first_k_entries_to_file(d=b_df, file_name=out_fn, column_name='gene', k=top_k)

    ery_df = pd.read_csv('./results/02_switchtfi/hematopoiesis/ranked_tfs.csv', index_col=[0])
    out_fn = os.path.join(out_p, f'top{top_k}_ery_switchtfi_driver_genes.txt')
    save_first_k_entries_to_file(d=ery_df, file_name=out_fn, column_name='gene', k=top_k)

    ery_df = pd.read_csv('./results/02_switchtfi/hematopoiesis/outdeg_ranked_tfs.csv', index_col=[0])
    out_fn = os.path.join(out_p, f'top{top_k}_ery_switchtfi_outdeg_driver_genes.txt')
    save_first_k_entries_to_file(d=ery_df, file_name=out_fn, column_name='gene', k=top_k)


def main_print_digest_res():

    def digest_resdf_helper(digest_res_path_list, name_list):
        databases = ['GO.BP', 'GO.CC', 'GO.MF', 'KEGG']
        df = pd.DataFrame()
        name_col = []
        database_col = []
        digest_scores_col = []
        digest_pvals_col = []
        for i, res_p in enumerate(digest_res_path_list):
            with open(res_p, 'r') as f:
                digest_res = json.load(f)
            for database in databases:
                name_col.append(name_list[i])
                database_col.append(database)
                digest_scores_col.append(digest_res['input_values']['values']['JI-based'][database])
                digest_pvals_col.append(digest_res['p_values']['values']['JI-based'][database])

        df['method'] = name_col
        df['database'] = database_col
        df['DIGEST score'] = digest_scores_col
        df['digest_pval'] = digest_pvals_col

        return df

    # Define names of method and color to be used in plots
    method_name_list = ['CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI']

    # ### Define paths to DIGEST results
    digest_base_p = './results/03_validation/digest_results'
    ap1 = os.path.join(digest_base_p, 'top20_alpha_cellrank/b8d14a52-62ad-49ca-871e-9b8456086669_result.json')
    ap2 = os.path.join(digest_base_p, 'top20_alpha_splicejac/694d78ec-0fcd-4055-84fc-d29c53ffbde5_result.json')
    ap3 = os.path.join(digest_base_p, 'top20_alpha_drivaer/c1c20b4f-4c58-4973-9cae-4fd9ced7f7d3_result.json')
    ap4 = os.path.join(digest_base_p, 'top20_alpha_switchtfi_outdeg/2a70a4b4-a78e-4ade-a2a6-e0a9c2d93901_result.json')
    ap5 = os.path.join(digest_base_p, 'top20_alpha_switchtfi/0abd6509-83f7-4952-bde6-717d9b6b0648_result.json')
    apl = [ap1, ap2, ap3, ap4, ap5]

    bp1 = os.path.join(digest_base_p, 'top20_beta_cellrank/542a7e3c-f24d-4856-9a7f-1e2b7c777e8e_result.json')
    bp2 = os.path.join(digest_base_p, 'top20_beta_splicejac/9706eeaa-ba4d-4189-aa71-a6248b60a0f6_result.json')
    bp3 = os.path.join(digest_base_p, 'top20_beta_drivaer/59f0306c-3da6-49bb-8391-55a8c59f1e70_result.json')
    bp4 = os.path.join(digest_base_p, 'top20_beta_switchtfi_outdeg/6adfd6ec-b32d-418e-8451-0e81f1c89914_result.json')
    bp5 = os.path.join(digest_base_p, 'top20_beta_switchtfi/c0932ef3-f281-46d5-acdd-358551495dbe_result.json')
    bpl = [bp1, bp2, bp3, bp4, bp5]

    eryp1 = os.path.join(digest_base_p, 'top20_ery_cellrank/a4077393-4050-42ed-bf1a-a9ab3ea3726b_result.json')
    eryp2 = os.path.join(digest_base_p, 'top20_ery_drivaer/0d0d89e4-ea15-4ee7-8cc2-da8fe8d39a4f_result.json')
    eryp3 = os.path.join(digest_base_p, 'top20_ery_switchtfi_outdeg/70becca4-53b6-4ede-a7f5-419b0e3da377_result.json')
    eryp4 = os.path.join(digest_base_p, 'top20_ery_switchtfi/4dfe96bb-48ff-4d23-b9c3-7e94f8bf3da0_result.json')
    erypl = [eryp1, eryp2, eryp3, eryp4]

    print('# ### Alpha:')
    print(digest_resdf_helper(digest_res_path_list=apl, name_list=method_name_list))

    print('# ### Beta:')
    print(digest_resdf_helper(digest_res_path_list=bpl, name_list=method_name_list))

    print('# ### Ery:')
    print(digest_resdf_helper(digest_res_path_list=erypl, name_list=method_name_list[:1] + method_name_list[2:]))


def main_store_anndata():
    from scipy.sparse import csr_matrix
    import pickle
    import lzma

    def helper(ad: sc.AnnData,
               filename: str):
        if isinstance(ad.X, np.ndarray):
            ad.X = csr_matrix(ad.X)
        for layer_key in ad.layers.keys():
            if isinstance(ad.layers[layer_key], np.ndarray):
                ad.layers[layer_key] = csr_matrix(ad.layers[layer_key])

        print('### Compressing')
        # Serialize (pickle) and compress the AnnData object to a file using lzma
        with lzma.open(filename, 'wb', preset=9) as f:
            pickle.dump(ad, f)

    adata = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
    bdata = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
    erydata = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')

    helper(ad=adata, filename='./switchtfi/d/pre-endocrine_alpha.pickle.xz')
    helper(ad=bdata, filename='./switchtfi/d/pre-endocrine_beta.pickle.xz')
    helper(ad=erydata, filename='./switchtfi/d/erythrocytes.pickle.xz')

    # print('## Decompressing')
    # Deserialize (unpickle) and decompress the AnnData object from the file
    # with lzma.open('./switchtfi/d/pre-endocrine_beta_9.pickle.xz', 'rb') as f:
    #     loaded_adata = pickle.load(f)

    # print(loaded_adata)


def main_get_basegrn_sizes():

    def print_sizes(grn: pd.DataFrame,
                    tf_target_keys: Tuple[str, str] = ('TF', 'target')):
        n_vertices = np.unique(grn[list(tf_target_keys)].to_numpy()).shape[0]
        n_tfs = np.unique(grn[tf_target_keys[0]].to_numpy()).shape[0]
        n_targets = np.unique(grn[tf_target_keys[1]].to_numpy()).shape[0]
        print(f'# ## n vertices: {n_vertices}')
        print(f'# ## n edges: {grn.shape[0]}')
        print(f'# ## n TFs: {n_tfs}')
        print(f'# ## n targets: {n_targets}')


    abasegrn = load_grn_json('./results/02_switchtfi/endocrine/alpha/grn.json')
    bbasegrn = load_grn_json('./results/02_switchtfi/endocrine/beta/grn.json')
    erybasegrn = load_grn_json('./results/02_switchtfi/hematopoiesis/grn.json')
    # agrn = pd.read_csv('./results/02_switchtfi/endocrine/alpha/grn.csv', index_col=0)
    # bgrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)
    # erygrn = pd.read_csv('./results/02_switchtfi/hematopoiesis/grn.csv', index_col=0)

    print('###### Alpha:')
    print_sizes(grn=abasegrn)

    print('###### Beta:')
    print_sizes(grn=bbasegrn)

    print('###### Erythrocytes:')
    print_sizes(grn=erybasegrn)


def main_transition_grn_val():
    import networkx as nx
    import matplotlib.pyplot as plt
    from switchtfi.tf_ranking import grn_to_nx

    def val_helper(base_grn: pd.DataFrame,
                   transition_grn: pd.DataFrame,
                   n: int = 100,
                   tf_target_keys: Tuple[str, str] = ('TF', 'target'),):

        sizes_lcc = np.zeros(n)
        n_edges_lcc = np.zeros(n)
        n_ccs = np.zeros(n)
        perc_of_total_lcc = np.zeros(n)
        n_vert = np.zeros(n)
        n_tf = np.zeros(n)
        n_target = np.zeros(n)
        ccs = [None] * n

        def get_size_lcc(h: nx.Graph) -> Tuple[int, int]:
            connected_components = list(nx.connected_components(h))
            largest_cc = max(connected_components, key=len)
            return len(connected_components), len(largest_cc)

        def rand_subnet_generator(bgrn: pd.DataFrame,
                                  tgrn: pd.DataFrame,
                                  mode: str = 'simple',
                                  tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> pd.DataFrame:

            if mode == 'simple':
                randgrn = bgrn.sample(n=tgrn.shape[0], replace=False, axis=0)
            elif mode == 'advanced':
                n_tfs_t = np.unique(tgrn[tf_target_keys[0]].to_numpy()).shape[0]
                tfs_b = np.unique(bgrn[tf_target_keys[0]].to_numpy())
                sampled_tfs_b = np.random.choice(tfs_b, size=n_tfs_t, replace=False)
                sampled_tf_bool_b = np.isin(bgrn[tf_target_keys[0]].to_numpy(), sampled_tfs_b)
                dummy_bgrn = bgrn[sampled_tf_bool_b].copy()
                randgrn = dummy_bgrn.sample(n=tgrn.shape[0], replace=False, axis=0)
            else:
                randgrn = pd.DataFrame()

            return randgrn

        for i in range(n):
            # Randomly select edges from the base GRN -> random subnetwork
            rand_subnet = rand_subnet_generator(bgrn=base_grn, tgrn=transition_grn, mode='simple',
                                                tf_target_keys=tf_target_keys)
            n_vert[i] = np.unique(rand_subnet[list(tf_target_keys)].to_numpy()).shape[0]
            n_tf[i] = np.unique(rand_subnet[tf_target_keys[0]].to_numpy()).shape[0]
            n_target[i] = np.unique(rand_subnet[tf_target_keys[1]].to_numpy()).shape[0]

            # Turn random subnetwork into undirected Networkx graph
            g = grn_to_nx(grn=rand_subnet, edge_attributes=None, tf_target_keys=tf_target_keys).to_undirected()
            ccs[i] = list(nx.connected_components(g))
            # Find size of the largest connected component
            a, b = get_size_lcc(h=g)
            n_ccs[i] = a
            sizes_lcc[i] = b
            perc_of_total_lcc[i] = b / g.number_of_nodes()
            # Get the number of edges in the lcc
            lcc = max(nx.connected_components(g), key=len)
            lcc_subgraph = g.subgraph(lcc)
            n_edges_lcc[i] = lcc_subgraph.number_of_edges()

        # Find size of lcc of transition GRN
        tg = grn_to_nx(grn=transition_grn, edge_attributes=None, tf_target_keys=tf_target_keys).to_undirected()
        nccs_tgrn, size_lcc_tgrn = get_size_lcc(h=tg)
        perc_of_total_lcc_tgrn = size_lcc_tgrn / tg.number_of_nodes()

        def calc_emp_pval(val: float,
                          val_vec: np.ndarray,
                          geq: bool = True) -> float:
            if geq:
                p_val = (val <= val_vec).sum()
            else:
                p_val = (val >= val_vec).sum()

            p_val /= (val_vec.shape[0] + 1)

            return p_val

        # Plot histogram of lcc sizes
        plt.hist(sizes_lcc, bins=15, edgecolor='black')
        plt.axvline(size_lcc_tgrn, color='red', linewidth=2,
                    label=f'val: {size_lcc_tgrn}, '
                          f'p-val: {calc_emp_pval(val=size_lcc_tgrn, val_vec=sizes_lcc, geq=False)}')
        plt.legend()
        plt.title('Lcc sizes')
        plt.show()

        # plot histogram of percentage of lcc vertices among all vertices in (sub-)network
        plt.hist(perc_of_total_lcc, bins=15, edgecolor='black', color='green')
        plt.axvline(perc_of_total_lcc_tgrn, color='red', linewidth=2,
                    label=f'val: {perc_of_total_lcc_tgrn}, '
                          f'p-val: {calc_emp_pval(val=perc_of_total_lcc_tgrn, val_vec=perc_of_total_lcc, geq=True)}')
        plt.legend()
        plt.title('Percentages of lcc nodes in network')
        plt.show()

        # Plot bar plot with number of connected components
        from collections import Counter
        # Count occurrences of each integer
        counts = Counter(n_ccs)
        # Create lists for the integers and their frequencies
        integers = list(counts.keys())
        frequencies = list(counts.values())
        plt.bar(integers, frequencies, width=1.0, edgecolor='black', color='orange')
        plt.axvline(nccs_tgrn, color='red', linewidth=2,
                    label=f'val: {nccs_tgrn}, '
                          f'p-val: {calc_emp_pval(val=nccs_tgrn, val_vec=n_ccs, geq=False)}')
        plt.legend()
        plt.xlabel('Number of connected components')
        plt.ylabel('Frequency')
        plt.show()

        # Plot histogram of percentage of TFs among vertices in (sub-)network
        tfperc = n_tf / n_vert
        tfperc_tgrn = np.unique(transition_grn[tf_target_keys[0]]).shape[0] / np.unique(transition_grn[list(
            tf_target_keys)]).shape[0]
        plt.hist(tfperc, bins=15, edgecolor='black', color='pink')
        plt.axvline(tfperc_tgrn, color='red', linewidth=2,
                    label=f'val: {tfperc_tgrn}, '
                          f'p-val: {calc_emp_pval(val=tfperc_tgrn, val_vec=tfperc, geq=False)}')
        plt.legend()
        plt.title('Percentages of TF in the network')
        plt.show()

        # Plot histogram of percentage of targets among vertices in (sub-)network
        targetperc = n_target / n_vert
        targetperc_tgrn = np.unique(transition_grn[tf_target_keys[1]]).shape[0] / np.unique(transition_grn[list(
            tf_target_keys)]).shape[0]
        plt.hist(targetperc, bins=15, edgecolor='black', color='purple')
        plt.axvline(targetperc_tgrn, color='red', linewidth=2,
                    label=f'val: {targetperc_tgrn}, '
                          f'p-val: {calc_emp_pval(val=targetperc_tgrn, val_vec=targetperc, geq=True)}')
        plt.legend()
        plt.title('Percentages of targets in the network')
        plt.show()

        # Plot histogram of number of vertices in (sub-)network
        nvert_tgrn = np.unique(transition_grn[list(tf_target_keys)]).shape[0]
        plt.hist(n_vert, bins=15, edgecolor='black', color='yellow')
        plt.axvline(nvert_tgrn, color='red', linewidth=2,
                    label=f'val: {nvert_tgrn}, '
                          f'p-val: {calc_emp_pval(val=nvert_tgrn, val_vec=n_vert, geq=False)}')
        plt.legend()
        plt.title('Number of vertices in the network')
        plt.show()

        # Plot histogram with number of edges in lcc
        lcc_tg = max(nx.connected_components(tg), key=len)
        lcc_subgraph_tg = tg.subgraph(lcc_tg)
        n_edges_lcc_tg = lcc_subgraph_tg.number_of_edges()

        plt.hist(n_edges_lcc / tg.number_of_edges(), bins=15, edgecolor='black', color='cyan')
        plt.axvline(n_edges_lcc_tg / tg.number_of_edges(), color='red', linewidth=2,
                    label=f'val: {n_edges_lcc_tg / tg.number_of_edges()}, '
                          f'p-val: {calc_emp_pval(val=n_edges_lcc_tg / tg.number_of_edges(), val_vec=n_edges_lcc / tg.number_of_edges(), geq=True)}')
        plt.legend()
        plt.title('Percentages of edges in the lcc')
        plt.show()

        # Plot histogram number of ccs weighted by size
        scores = [0] * n
        for i, cc in enumerate(ccs):  # iterate over iterators of components
            scores[i] = sum((len(c) / n_vert[i]) ** 2 for c in cc)
        scores = np.array(scores)

        components_tg = list(nx.connected_components(tg))
        score_tgrn = sum((len(c) / tg.number_of_nodes()) ** 2 for c in components_tg)

        plt.hist(scores, bins=15, edgecolor='black', color='olive')
        plt.axvline(score_tgrn, color='red', linewidth=2,
                    label=f'val: {score_tgrn}, '
                          f'p-val: {calc_emp_pval(val=score_tgrn, val_vec=scores, geq=True)}')
        plt.legend()
        plt.title('Scores')
        plt.show()

        # Plot histogram of harmonic mean of component sizes
        harmonic_means = [0] * n
        for i, cc in enumerate(ccs):
            harmonic_means[i] = len(cc) / sum(1 / len(c) for c in cc)

        harmonic_mean_tg = len(components_tg) / sum(1 / len(c) for c in components_tg)

        plt.hist(harmonic_means, bins=15, edgecolor='black', color='gray')
        plt.axvline(harmonic_mean_tg, color='red', linewidth=2,
                    label=f'val: {harmonic_mean_tg}, '
                          f'p-val: {calc_emp_pval(val=harmonic_mean_tg, val_vec=np.array(harmonic_means), geq=True)}')
        plt.legend()
        plt.title('Harmonic means')
        plt.show()

        return

    # erybasegrn = load_grn_json('./results/02_switchtfi/hematopoiesis/grn.json')
    # erygrn = pd.read_csv('./results/02_switchtfi/hematopoiesis/grn.csv', index_col=0)

    # erybasegrn = load_grn_json('./results/02_switchtfi/endocrine/beta/grn.json')
    # erygrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)

    from switchtfi.plotting import plot_grn
    from switchtfi.pvalue_calculation import remove_insignificant_edges

    # erygrn = remove_insignificant_edges(grn=erybasegrn, alpha=0.75, inplace=False)

    erybasegrn = load_grn_json('./results/02_switchtfi/endocrine/alpha/grn.json')
    erygrn = pd.read_csv('./results/02_switchtfi/endocrine/alpha/grn.csv', index_col=0)

    plot_grn(grn=erygrn, plot_folder='./', fn_prefix='zzz_')

    print('#')
    print(erybasegrn)
    print('##')
    print(erygrn)

    print(f'Number of vertices in transition GRN: {np.unique(erygrn[["TF", "target"]]).shape}')

    val_helper(base_grn=erybasegrn, transition_grn=erygrn, n=1000)

    # ## Notes:
    # Fewer expected connected components than expected by chance
    #
    # The number of vertices in the lcc is smaller than expected by chance
    # -> indicates that fewer genes, that are in close mechanistic relation are in lcc of transition GRN
    # BUT! -> Random model is flawed pick  random TFs + Targets => probably similar lcc structure as in transition GRN
    # Get result because SwitchTFI picks TF and basically all its targets, when picking at random get more vertices ...
    #
    # Maybe use random model that more closely preserves GRN structure
    # -> Sample as many TFs as GRN has and then add edges uniform at random ...


def main_get_targets():

    def print_target_lists(grn: pd.DataFrame,
                           tfs: list[str],
                           sort_by: Union[str, None] = None,
                           fp: str = './regulons.txt',
                           tf_target_keys: Tuple[str, str] = ('TF', 'target')):

        if sort_by is not None:
            if sort_by not in grn.columns and sort_by == 'score':
                weights = grn['weight'].to_numpy()
                pvals = grn['pvals_wy'].to_numpy()
                pvals += np.finfo(np.float64).eps
                grn['score'] = -np.log10(pvals) * weights

            grn = grn.sort_values(sort_by, axis=0, ascending=False).reset_index(drop=True)

        with open(fp, 'a') as f:
            f.write('###### Note: TFs sorted according to centrality in transition GRN\n')
            for tf in tfs:
                tf_bool = np.isin(grn[tf_target_keys[0]].to_numpy(), np.array([tf]))
                regulon = grn[tf_bool].copy()
                targets = regulon[tf_target_keys[1]].tolist()
                f.write(f'###### TF: {tf}\n')
                f.write(f'### {targets}\n')
                print(f'###### TF: {tf}')
                print(f'### {targets}')


    agrn = pd.read_csv('./results/02_switchtfi/endocrine/alpha/grn.csv', index_col=0)
    bgrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)
    erygrn = pd.read_csv('./results/02_switchtfi/hematopoiesis/grn.csv', index_col=0)
    print_target_lists(agrn, tfs=['Creb3l2', 'Jund', 'Cdx2', 'Nkx6-1', 'Arx', 'Klf8', 'Maff'], sort_by='score', fp='./regulons_alpha.txt')
    print_target_lists(bgrn, tfs=['Ybx1', 'Fev', 'Yy1', 'Foxa3', 'Etv1', 'Lhx1'], sort_by='score', fp='./regulons_beta.txt')
    print_target_lists(erygrn, tfs=['Mbd2', 'Myc', 'Foxo1', 'Fli1', 'Cux1', 'Myb', 'Elk3', 'Ikzf2'], sort_by='score', fp='./regulons_ery.txt')

    from switchtfi.plotting import plot_grn
    plot_grn(grn=agrn, fn_prefix='alpha_')


if __name__ == '__main__':

    # main_pseudotime_inference()

    # main_switchde_analysis()

    # main_trend_calculation()

    # main_save_cellrank_driver_genes()

    # main_save_splicejac_driver_genes()

    # main_save_drivaer_driver_genes()

    # main_sig_thresh_selection()

    # main_robustness_analysis()

    # ### Miscellaneous scripts, generally uninteresting to reader
    # main_save_topk_genesets()
    # main_print_digest_res()
    # main_store_anndata()
    # main_get_basegrn_sizes()
    # main_transition_grn_val()
    # main_get_targets()

    print('done')
