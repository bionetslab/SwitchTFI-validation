
import numpy as np
import scanpy as sc
import pandas as pd
import DrivAER as dv

from typing import *


def get_regulons(
        grn: pd.DataFrame,
        gene_names: Union[List[str], None] = None,
        additional_info_keys: Union[List[str], None] = None,
        tf_target_keys: Tuple[str, str] = ('TF', 'target')
) -> Dict[str, Dict]:

    # If no gene names are passed, compute regulons of all TFs
    if gene_names is None:
        gene_names = np.unique(grn[tf_target_keys[0]].to_numpy()).tolist()

    regulon_dict = {}
    for gene in gene_names:

        gene_tf_bool = (grn[tf_target_keys[0]].to_numpy() == gene)

        if gene_tf_bool.sum() == 0:
            print(f'WARNING: Gene {gene} is not a TF in the given GRN')

        dummy = {'targets': grn[tf_target_keys[1]].to_numpy()[gene_tf_bool].tolist()}

        if additional_info_keys is not None:
            for key in additional_info_keys:
                dummy[key] = grn[key].to_numpy()[gene_tf_bool].tolist()

        regulon_dict[gene] = dummy

    return regulon_dict


def get_tf_target_pdseries(
        grn: pd.DataFrame,
        tf_target_keys: Tuple[str, str] = ('TF', 'target')
) -> pd.Series:

    tf_target_dict = get_regulons(
        grn=grn,
        gene_names=None,
        tf_target_keys=tf_target_keys
    )

    for key, val in tf_target_dict.items():
        tf_target_dict[key] = val['targets']

    tf_target_series = pd.Series(tf_target_dict)
    tf_target_series.index.name = 'TF'

    return tf_target_series


def get_drivaer_driver_genes(
        adata: sc.AnnData,
        grn: pd.DataFrame,
        top_k: Union[int, None] = 10,
        dim_red_method: str = 'dca',  # 'pca', 'umap', 'tsne'
        clustering_obs_key: str = 'clusters',
        tf_target_keys: Tuple[str, str] = ('TF', 'target'),
        verbosity: int = 0
) -> Tuple[list, pd.DataFrame, tuple]:

    # Extract pandas series from GRN, TF: List[targets]
    # -> Relevance TFs is defined the predictive power of their target gene set w.r.t the condition of interest
    tf_targets = get_tf_target_pdseries(grn=grn, tf_target_keys=tf_target_keys)

    if dim_red_method == 'dca':
        # Train DCA model on gene-sets and train random forest model on DCA embedding
        low_dim_rep, relevance, genes = dv.calc_relevance(
            count=adata,
            pheno=adata.obs[clustering_obs_key],
            tf_targets=tf_targets,
            min_targets=10,
            ae_type='nb-conddisp',
            epochs=50,
            early_stop=3,
            hidden_size=(8, 2, 8),
            verbose=(verbosity >= 1)
        )

    elif dim_red_method == 'pca':
        low_dim_rep, relevance, genes = dv.calc_relevance_pca(
            adata=adata,
            pheno=adata.obs[clustering_obs_key],
            tf_targets=tf_targets,
            min_targets=6
        )

    elif dim_red_method == 'umap':
        low_dim_rep, relevance, genes = dv.calc_relevance_umap(
            adata=adata,
            pheno=adata.obs[clustering_obs_key],
            tf_targets=tf_targets,
            min_targets=6
        )

    elif dim_red_method == 'tsne':
        low_dim_rep, relevance, genes = dv.calc_relevance_tsne(
            adata=adata,
            pheno=adata.obs[clustering_obs_key],
            tf_targets=tf_targets,
            min_targets=6
        )

    else:
        raise ValueError('dim_red_method must be one of dca, pca, umap, tsne')

    drivaer_out = (low_dim_rep, relevance, genes)

    res_df = pd.DataFrame()
    res_df['gene'] = genes
    res_df['relevance'] = relevance

    res_df.sort_values(by='relevance', ascending=False, inplace=True)
    res_df.reset_index(drop=True, inplace=True)

    if top_k is None:
        top_k = res_df.shape[0] - 1

    top_k_list = res_df['gene'].to_list()[0:top_k]

    return top_k_list, res_df, drivaer_out


