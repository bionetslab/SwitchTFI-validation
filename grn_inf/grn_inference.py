
import ctxcore.rnkdb
import pyscenic.utils
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle

from typing import *
from tqdm import tqdm
from arboreto.algo import grnboost2, genie3
from pyscenic.utils import modules_from_adjacencies
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell


def pyscenic_pipeline(adata: sc.AnnData,
                      layer_key: Union[None, str],
                      tf_file: Union[None, str],
                      result_folder: Union[None, str],
                      database_path: str,
                      motif_annotations_path: str,
                      grn_inf_method: str = 'grnboost2',
                      fn_prefix: Union[None, str] = None,
                      verbosity: int = 0,
                      plot: bool = False):
    """
    Run the SCENIC method for gene regulatory network (GRN) inference on scRNA-seq data stored as an AnnData object.

    This function relies on the PyScenic implementation of the Scenic method.
    The method can be broken down into 4 Steps:
    1. Inference of co-expression modules from gene expression data using methods like GRNBoost2.
    2. Pruning of GRNs using cis-regulatory footprints (with RcisTarget).
    3. Extraction of the pruned GRN from the PySCENIC results.
    4. Calculation of cellular regulon enrichment (AUCell) to quantify regulon activity in each cell.

    Args:
        adata (sc.AnnData): The AnnData object containing the single-cell gene expression data.
        layer_key (Union[None, str]): The layer in `adata` to use for gene expression values. If None, the main matrix is used.
        tf_file (Union[None, str]): File containing transcription factor (TF) names. If None, all genes are considered as potential TFs.
        result_folder (Union[None, str]): Folder to save results such as intermediate files and GRN outputs. If None, results are not saved.
        database_path (str): Path to the databases required for cis-regulatory analysis (RcisTarget).
        motif_annotations_path (str): Path to motif annotations used in cis-regulatory analysis.
        grn_inf_method (str, optional): The method for GRN inference ('grnboost2' or 'genie3'). Defaults to 'grnboost2'.
        fn_prefix (Union[None, str], optional): Optional prefix for filenames when saving results. Defaults to None.
        verbosity (int, optional): Level of logging for detailed output. Defaults to 0.
        plot (bool, optional): Whether to generate plots during the pipeline stages. Defaults to False.

    Returns:
        pd.DataFrame: The pruned gene regulatory network (GRN) inferred with Scenic.
    """


    if verbosity >= 1:
        print('###### Starting PySCENIC pipeline ... ######')

    # Extract cell x gene - expression matrix of desired layer
    expression_mtrx = adata.to_df(layer=layer_key)

    # Load list of TFs
    if tf_file is not None:
        tf_names = load_tf_names(tf_file)
        check_tf_gene_set_intersection(tf_names=np.array(tf_names),
                                       gene_names=adata.var_names.to_numpy(),
                                       verbosity=verbosity)
    else:
        tf_names = 'all'

    # Phase I: Inference of co-expression modules ###
    # Infer initial GRN using GRNboost2
    adjacencies = infer_basic_grn(expression_matrix=expression_mtrx,
                                  tf_names=tf_names,
                                  method=grn_inf_method,
                                  result_folder=result_folder,
                                  verbosity=verbosity,
                                  plot=plot,
                                  fn_prefix=fn_prefix)
    # Derive potential regulons from these co-expression modules
    modules = modules_from_grn(adjacencies=adjacencies,
                               expression_matrix=expression_mtrx,
                               result_folder=result_folder,
                               fn_prefix=fn_prefix)

    # Phase II: Prune modules for targets with cis regulatory footprints (aka RcisTarget) ###
    res_df = prune_grn(modules=modules,
                       database_path=database_path,
                       motif_annotations_path=motif_annotations_path,
                       result_folder=result_folder,
                       verbosity=verbosity,
                       fn_prefix=fn_prefix)

    # Phase III: Extract pruned GRN from pyscenic results dataframe
    pruned_grn = pyscenic_result_df_to_grn(pyscenic_result_df=res_df,
                                           result_folder=result_folder,
                                           verbosity=verbosity,
                                           fn_prefix=fn_prefix)

    # Phase IV: Cellular regulon enrichment matrix (aka AUCell)
    # Get regulons from results dataframe
    regulons = pyscenic_res_df_to_regulons(pyscenic_result_df=res_df,
                                           result_folder=result_folder,
                                           fn_prefix=fn_prefix)
    # Calculate cellular regulon enrichment matrix
    # cell x TF, Enrichment of a regulon is measures as AUC of the recovery curve of the genes that define this regulon.
    auc_mtrx = regulons_to_aucell_matrix(expression_matrix=expression_mtrx,
                                         regulons=regulons,
                                         result_folder=result_folder,
                                         verbosity=verbosity,
                                         plot=plot,
                                         fn_prefix=fn_prefix)

    return pruned_grn


def check_tf_gene_set_intersection(tf_names: np.ndarray,
                                   gene_names: np.ndarray,
                                   verbosity: int = 0):

    intersection = np.intersect1d(tf_names, gene_names, return_indices=False)
    perc = intersection.shape[0] / tf_names.shape[0]
    if verbosity >= 1:
        print(f'# There are {gene_names.shape[0]} genes present in the dataset')
        print(f'# There are {tf_names.shape[0]} TFs')
        print(f'# {intersection.shape[0]} out of the {tf_names.shape[0]} TFs appear in the dataset,')
        print(f'# this corresponds to {round(perc, 3) * 100}%')


def infer_basic_grn(expression_matrix: pd.DataFrame,
                    tf_names: Union[str, list[str]],
                    method: str = 'grnboost2',
                    result_folder: Union[None, str] = None,
                    verbosity: int = 0,
                    plot: bool = False,
                    **kwargs) -> pd.DataFrame:

    if verbosity >= 1:
        if method == 'grnboost2':
            print('### GRNboost2 ... ###')
        elif method == 'genie3':
            print('### GENIE3 ... ###')

    if method == 'grnboost2':
        grn = grnboost2(expression_data=expression_matrix,
                        tf_names=tf_names,
                        verbose=False)  # out structure: TF target importance
    elif method == 'genie3':
        grn = genie3(expression_data=expression_matrix,
                     tf_names=tf_names,
                     verbose=False)
    else:
        grn = pd.DataFrame()

    if result_folder is not None:
        prefix = kwargs.get('fn_prefix')  # Get prefix for filename, returns None if no root is passed in kwargs
        if prefix is None:
            prefix = ''
        grn_p = os.path.join(result_folder, f'{prefix}basic_grn.csv')
        grn.to_csv(grn_p, index=False, sep='\t')
        # grn = pd.read_csv(grn_p, sep='\t')

    if verbosity >= 1:
        print(grn.head())
        genes = np.unique(grn[['TF', 'target']].to_numpy())
        print(f'# The inferred GRN has {genes.shape[0]} vertices and {grn.shape[0]} edges')

    if plot:
        plt.hist(np.log10(grn['importance']), bins=50)
        plt.show()

    return grn


def modules_from_grn(adjacencies: pd.DataFrame,
                     expression_matrix: pd.DataFrame,
                     result_folder: Union[None, str] = None,
                     **kwargs) -> list[pyscenic.utils.Regulon]:
    # modules = list(modules_from_adjacencies(adjacencies, expression_matrix))
    modules = list(modules_from_adjacencies(adjacencies, expression_matrix, rho_mask_dropouts=False))
    # Todo ...
    if result_folder is not None:
        prefix = kwargs.get('fn_prefix')  # Get prefix for filename, returns None if no root is passed in kwargs
        if prefix is None:
            prefix = ''
        modules_p = os.path.join(result_folder, f'{prefix}modules.pkl')
        with open(modules_p, 'wb') as f:
            pickle.dump(modules, f)
        # with open(modules_p, 'rb') as f:
        #     modules = pickle.load(f)

    return modules


def prune_grn(modules: list[pyscenic.utils.Regulon],
              database_path: str,
              motif_annotations_path: str,
              result_folder: Union[None, str] = None,
              verbosity: int = 0,
              **kwargs) -> pd.DataFrame:
    if verbosity >= 1:
        print('### Prune GRN ... ###')

    dbs = load_ranking_databases(database_path=database_path)

    # Prune GRN
    res_df = prune2df(rnkdbs=dbs,
                      modules=modules,
                      motif_annotations_fname=motif_annotations_path)

    if result_folder is not None:
        prefix = kwargs.get('fn_prefix')  # Get prefix for filename, returns None if no root is passed in kwargs
        if prefix is None:
            prefix = ''
        res_p = os.path.join(result_folder, f'{prefix}result_pyscenic.csv')
        res_df.to_csv(res_p)

    if verbosity >= 1:
        print(res_df.head())

    return res_df


def pyscenic_res_df_to_regulons(pyscenic_result_df: pd.DataFrame,
                                result_folder: Union[None, str] = None,
                                **kwargs) -> Sequence[pyscenic.utils.Regulon]:

    # pyscenic_result_df = correct_dtype_psc_res_df(pyscenic_result_df=pyscenic_result_df)
    regulons = df2regulons(pyscenic_result_df)

    if result_folder is not None:
        prefix = kwargs.get('fn_prefix')  # Get prefix for filename, returns None if no root is passed in kwargs
        if prefix is None:
            prefix = ''
        regulon_p = os.path.join(result_folder, f'{prefix}regulons.pkl')
        with open(regulon_p, 'wb') as f:
            pickle.dump(regulons, f)
        # with open(regulon_p, 'rb') as f:
        #    regulons = pickle.load(f)

    return regulons


def regulons_to_aucell_matrix(expression_matrix: pd.DataFrame,
                              regulons: Sequence[pyscenic.utils.Regulon],
                              result_folder: Union[None, str] = None,
                              verbosity: int = 0,
                              plot: bool = False,
                              **kwargs) -> pd.DataFrame:
    auc_mtx = aucell(expression_matrix, regulons)

    if result_folder is not None:
        prefix = kwargs.get('fn_prefix')
        if prefix is None:
            prefix = ''
        auc_p = os.path.join(result_folder, f'{prefix}auc_matrix.csv')
        auc_mtx.to_csv(auc_p)

    if verbosity >= 1:
        print(f'# The AUCell matrix contains {auc_mtx.shape[0]} cells and {auc_mtx.shape[1]} TFs')
        print(auc_mtx.head())

    if plot:
        try:
            sns.clustermap(auc_mtx, figsize=(12, 12))
            # plt.savefig('my_seaborn_plot.png')
            plt.show()
        except RecursionError:
            print('# Plotting not possible, '
                  'RecursionError: maximum recursion depth exceeded while getting the str of an object')

    top_n = 50
    top_tfs = auc_mtx.max(axis=0).sort_values(ascending=False).head(top_n)
    auc_mtx_top_n = auc_mtx[
        [c for c in auc_mtx.columns if c in top_tfs]
    ]
    print(f'# The top-{top_n} TFs are ...')
    print(top_tfs)
    if verbosity:
        print()
    if plot:
        sns.clustermap(
            auc_mtx_top_n,
            figsize=[15, 6.5],
            cmap="Blues",
            xticklabels=True,
            yticklabels=True,
        )
        # plt.savefig('my_seaborn_plot2.png')
        plt.show()

    return auc_mtx


def pyscenic_result_df_to_grn(pyscenic_result_df: pd.DataFrame,
                              result_folder: Union[None, str],
                              verbosity: int = 0,
                              **kwargs) -> pd.DataFrame:
    # Note: pyscenic_result_df has the following structure
    # Index columns: 'TF', 'MotifID'
    # Column name levels: 'Enrichment'; 'AUC', 'NES', 'MotifSimilarityQvalue', 'OrthologousIdentity', 'Annotation',
    # 'Context', 'TargetGenes', 'RankAtMax'

    # Get names of TFs that appear in the result
    tf_names = pyscenic_result_df.index.get_level_values('TF').to_numpy()
    tf_list = []
    tg_list = []
    weight_list = []  # ???
    for tf in tqdm(tf_names, total=tf_names.shape[0]):
        for target_list in pyscenic_result_df.loc[[tf]]['Enrichment']['TargetGenes']:
            if isinstance(target_list, str):
                # tl_str = target_list
                # tl_str = tl_str.replace('[', '')
                # tl_str = tl_str.replace(']', '')
                # tl_str = tl_str.replace('(', '')
                # tl_str = tl_str.replace(')', '')
                # tl_str = tl_str.replace(' ', '')
                # tl_str = tl_str.replace("'", '')
                # tl_str = tl_str.split(sep=',')
                # target_list = [(tl_str[i], float(tl_str[i + 1])) for i in range(0, len(tl_str), 2)]

                target_list = target_list_string_to_list(tl_str=target_list)

            for target, weight in target_list:
                tf_list.append(tf)
                tg_list.append(target)
                weight_list.append(weight)

    grn = pd.DataFrame()
    grn['TF'] = tf_list
    grn['target'] = tg_list
    grn['scenic_weight'] = weight_list

    # If multiple databases were used, duplicate edges might exist, remove them
    grn = grn.drop_duplicates(subset=['TF', 'target'])

    # Remove self loops
    grn = remove_self_loops(grn)

    # Reset index
    grn = grn.reset_index(drop=True)

    if result_folder is not None:
        prefix = kwargs.get('fn_prefix')  # Get prefix for filename, returns None if no root is passed in kwargs
        if prefix is None:
            prefix = ''
        grn_p = os.path.join(result_folder, f'{prefix}pruned_grn.csv')
        grn.to_csv(grn_p)

    if verbosity >= 1:
        print(grn.head())
        genes = np.unique(grn[['TF', 'target']].to_numpy())
        print(f'# The inferred GRN has {genes.shape[0]} vertices and {grn.shape[0]} edges')

    return grn


def combine_grns(grn_list: List[pd.DataFrame],
                 n_occurrence_thresh: int,
                 result_folder: Union[None, str],
                 tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                 verbosity: int = 0,
                 **kwargs) -> pd.DataFrame:

    """
    Combine multiple Gene Regulatory Networks (GRNs) into a single consensus GRN.

    This function merges multiple GRNs (as edge lists stored in DataFrames) by identifying edges
    (TF-target pairs) that appear in at least `n_occurrence_thresh` GRNs.
    The resulting combined GRN is saved to a CSV file if a result folder is provided.

    Args:
        grn_list (List[pd.DataFrame]): A list of GRN DataFrames to combine. Each DataFrame must have
            the same columns with transcription factors and corresponding targets specified by `tf_target_keys`.
        n_occurrence_thresh (int): Minimum number of occurrences of an edge across the GRNs to include it
            in the combined GRN.
        result_folder (Union[None, str]): Folder to save the resulting combined GRN as a CSV. If None,
            the result is not saved.
        tf_target_keys (Tuple[str, str], optional): Column names representing transcription factors (TF)
            and target genes in the GRNs. Defaults to ('TF', 'target').
        verbosity (int, optional): Level of logging for output messages. Defaults to 0.
        **kwargs: Additional arguments for customization, such as `fn_prefix` for the prefix of the result filename.

    Returns:
        pd.DataFrame: The combined GRN as a DataFrame containing TF-target pairs.
    """

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


# Auxiliary functions ##################################################################################################
def load_tf_names(path: str) -> list[str]:

    with open(path) as file:
        tfs_in_file = [line.strip() for line in file.readlines()]

    return tfs_in_file


def get_fname_wo_extension(fname: str) -> AnyStr:
    return os.path.splitext(os.path.basename(fname))[0]


def load_ranking_databases(database_path: str) -> list[ctxcore.rnkdb.FeatherRankingDatabase]:
    if '*' in database_path:  # Check if multiple database files are to be loaded
        database_files = glob.glob(database_path)
        dbs = [RankingDatabase(fname=fname, name=get_fname_wo_extension(fname)) for fname in database_files]
    else:
        dbs = [RankingDatabase(fname=database_path,
                               name=get_fname_wo_extension(database_path))]

    return dbs


def load_pyscenic_result_df(res_df_path: str,
                            verbosity: int = 0) -> pd.DataFrame:
    res_df = pd.read_csv(res_df_path, index_col=[0, 1], header=[0, 1])

    if verbosity >= 1:
        print(res_df)

    return res_df


def remove_self_loops(grn: pd.DataFrame,
                      tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> pd.DataFrame:
    tfs = grn[tf_target_keys[0]].to_numpy()
    targets = grn[tf_target_keys[1]].to_numpy()

    keep_bool = np.logical_not(np.equal(tfs, targets))

    grn = grn[keep_bool]

    return grn


def correct_dtype_psc_res_df(pyscenic_result_df: pd.DataFrame) -> pd.DataFrame:

    list_of_target_list_strings = pyscenic_result_df['Enrichment']['TargetGenes'].to_list()
    list_of_target_lists = [target_list_string_to_list(tl) for tl in list_of_target_list_strings]

    # pyscenic_result_df = pyscenic_result_df.drop(columns=[('Enrichment', 'TargetGenes')])
    # pyscenic_result_df = pyscenic_result_df.drop(columns='TargetGenes', axis=1, level=1)

    pyscenic_result_df[('Enrichment', 'TargetGenes')] = list_of_target_lists

    return pyscenic_result_df


def target_list_string_to_list(tl_str: str) -> list:
    tl_str = tl_str.replace('[', '')
    tl_str = tl_str.replace(']', '')
    tl_str = tl_str.replace('(', '')
    tl_str = tl_str.replace(')', '')
    tl_str = tl_str.replace(' ', '')
    tl_str = tl_str.replace("'", '')
    tl_str = tl_str.split(sep=',')

    target_list = [(tl_str[i], float(tl_str[i + 1])) for i in range(0, len(tl_str), 2)]

    return target_list

