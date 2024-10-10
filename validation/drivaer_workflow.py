
import os
import pickle
import scanpy as sc
import numpy as np
import pandas as pd
import DrivAER as dv

from typing import *
from switchtfi.utils import get_regulons


def get_tf_target_pdseries(grn: pd.DataFrame,
                           tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> pd.Series:
    tf_target_dict = get_regulons(grn=grn,
                                  gene_names=None,
                                  tf_target_keys=tf_target_keys)

    for key, val in tf_target_dict.items():
        tf_target_dict[key] = val['targets']

    tf_target_series = pd.Series(tf_target_dict)
    tf_target_series.index.name = 'TF'

    return tf_target_series


def get_drivaer_driver_genes(adata: sc.AnnData,
                             grn: pd.DataFrame,
                             top_k: Union[int, None] = 10,
                             dim_red_method: str = 'dca',  # 'pca', 'umap', 'tsne'
                             clustering_obs_key: str = 'clusters',
                             tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                             verbosity: int = 0) -> Tuple[list, pd.DataFrame, tuple]:

    # Extract pandas series from GRN, TF: List[targets]
    # -> Relevance TFs is defined the predictive power of their target gene set w.r.t the condition of interest
    tf_targets = get_tf_target_pdseries(grn=grn,
                                        tf_target_keys=tf_target_keys)

    if dim_red_method == 'dca':
        # Train DCA model on gene-sets and train random forest model on DCA embedding
        low_dim_rep, relevance, genes = dv.calc_relevance(count=adata,
                                                          pheno=adata.obs[clustering_obs_key],
                                                          tf_targets=tf_targets,
                                                          min_targets=10,
                                                          ae_type='nb-conddisp',
                                                          epochs=50,
                                                          early_stop=3,
                                                          hidden_size=(8, 2, 8),
                                                          verbose=(verbosity >= 1))
    elif dim_red_method == 'pca':
        low_dim_rep, relevance, genes = dv.calc_relevance_pca(adata=adata,
                                                              pheno=adata.obs[clustering_obs_key],
                                                              tf_targets=tf_targets,
                                                              min_targets=6)
    elif dim_red_method == 'umap':
        low_dim_rep, relevance, genes = dv.calc_relevance_umap(adata=adata,
                                                               pheno=adata.obs[clustering_obs_key],
                                                               tf_targets=tf_targets,
                                                               min_targets=6)
    elif dim_red_method == 'tsne':
        low_dim_rep, relevance, genes = dv.calc_relevance_tsne(adata=adata,
                                                               pheno=adata.obs[clustering_obs_key],
                                                               tf_targets=tf_targets,
                                                               min_targets=6)
    else:
        low_dim_rep, relevance, genes = [], [], []

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


def main_drivaer_workflow():

    data_p = '/data/bionets/ac07izid/dtfi/data/drivaer_aux_data/'

    cell_type = 'alpha'  # 'beta'
    fn_ad = f'preendocrine_{cell_type}.h5ad'
    adata = sc.read_h5ad(filename=os.path.join(data_p, fn_ad))
    print(adata)
    print(adata.X)

    print(adata.var_names)
    adata.var_names = np.array([byte.decode('ascii') for byte in adata.var_names.to_numpy()])
    print(adata.var_names)

    print(adata.obs['clusters'])
    adata.obs['clusters'] = np.array([byte.decode('ascii') for byte in adata.obs['clusters'].to_numpy()])
    print(adata.obs['clusters'])

    # Load annotated gene set into pandas series
    # fn_anno = 'c3.all.v2023.2.Hs.symbols.gmt'   # -> newer gene sets exist ...
    # fn = 'C3.gmt'  # 'C3.gmt', 'hallmark.gmt', 'trrust_human.tsv', 'trrust_mouse.tsv'
    # c3_mouse = dv.get_anno(filename=fn, filetype="gmt", conv_mouse=True)
    # print(c3_mouse)
    # print(type(c3_mouse))
    # print(c3_mouse.index)
    # print(c3_mouse.values)

    # Load GRN
    base_p = '/data/bionets/ac07izid/dtfi/results/01_grn_inf/'
    sub_dir = f'endocrine_{cell_type}'  # f'endocrine_{cell_type}2'
    grn_thresh = 18  # 9, 18, 27
    grn_fn = f'{grn_thresh}_thresh_{cell_type}_combined_grn.csv'
    grn_fp = os.path.join(base_p, sub_dir, grn_fn)
    grn = pd.read_csv(grn_fp, index_col=[0])

    # Extract pandas series from GRN, TF: List[targets]
    tf_targets = get_tf_target_pdseries(grn=grn)
    print(tf_targets)

    res_df = dv.calc_relevance(count=adata,
                               pheno=adata.obs['clusters'],
                               tf_targets=tf_targets,
                               min_targets=6,
                               verbose=True)
    print(res_df)

    with open(f'zzz_drivaer_res_{cell_type}.pkl', 'wb') as f:
        pickle.dump(res_df, f)


def main_load_res():
    cell_type = 'beta'
    fp = f'/data/bionets/ac07izid/dtfi/data/drivaer_aux_data/drivaer_res_{cell_type}.pkl'
    with open(fp, 'rb') as f:
        res = pickle.load(f)

    print(res)

    dv.rank_plot(result=res)

    score = pd.DataFrame({'Signature': res[2], 'Relevance Score': res[1]})
    score = score.sort_values('Relevance Score', ascending=False)
    print(score[0:20])
    print(score['Signature'][0:20].to_list())

