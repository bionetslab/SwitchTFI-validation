
import scanpy as sc
import pandas as pd
from switchtfi.data import preendocrine_alpha, preendocrine_beta, erythrocytes
from switchtfi.fit import fit_model
from switchtfi.tf_ranking import rank_tfs


def main_endocrine():
    # ### Load the previously preprocessed scRNA-seq data stored as an AnnData object
    # (also available via the SwitchTFI functions)
    adata = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
    bdata = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
    # adata = preendocrine_alpha()
    # bdata = preendocrine_beta()

    # ### Load the previously inferred GRNs
    agrn = pd.read_csv('./results/01_grn_inf/endocrine/alpha/ngrnthresh9_alpha_pyscenic_combined_grn.csv',
                       index_col=0)
    bgrn = pd.read_csv('./results/01_grn_inf/endocrine/beta/ngrnthresh9_beta_pyscenic_combined_grn.csv',
                       index_col=0)

    # ### Perform SwitchTFI analyses
    # - Compute weights and empirical corrected p-values for each edge in the input GRN
    # - Prune the input GRN: remove edge if p-value > FWER-threshold  ==> transition GRN
    # - Rank transcription factors according to centrality in transition GRN
    agrn, aranked_tfs = fit_model(adata=adata,
                                  grn=agrn,
                                  result_folder='./results2/02_switchtfi/endocrine/alpha',
                                  verbosity=0,
                                  plot=True,
                                  save_intermediate=True)

    bgrn, branked_tfs = fit_model(adata=bdata,
                                  grn=bgrn,
                                  result_folder='./results/02_switchtfi/endocrine/beta',
                                  verbosity=0,
                                  plot=True,
                                  save_intermediate=True)


def main_hematopoiesis():
    # ### Load the previously preprocessed scRNA-seq data stored as an AnnData object
    # (also available via the SwitchTFI functions)
    adata = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')
    # adata = erythrocytes()

    # ### Load the previously inferred GRN
    grn = pd.read_csv('./results/01_grn_inf/hematopoiesis/ngrnthresh9_erythrocytes_pyscenic_combined_grn.csv',
                      index_col=0)

    # ### Perform SwitchTFI analyses
    # - Compute weights and empirical corrected p-values for each edge in the input GRN
    # - Prune the input GRN: remove edge if p-value > FWER-threshold  ==> transition GRN
    # - Rank transcription factors according to centrality in transition GRN
    grn, ranked_tfs = fit_model(adata=adata,
                                grn=grn,
                                result_folder='./results/02_switchtfi/hematopoiesis',
                                clustering_obs_key='prog_off',
                                verbosity=0,
                                plot=True,
                                save_intermediate=True)


def main_compute_outdegree_tf_rankings():
    # ### In the previous scripts the transcription factors were ranked according to their PageRank in the
    # transition GRN. Now we also generate the results for ranking according to outdegree.

    # ### Load the previously computed transition GRNs
    agrn = pd.read_csv('./results/02_switchtfi/endocrine/alpha/grn.csv', index_col=0)
    bgrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)
    erygrn = pd.read_csv('./results/02_switchtfi/hematopoiesis/grn.csv', index_col=0)

    # ### Compute the outdegree-based TF-rankings
    adf = rank_tfs(grn=agrn, centrality_measure='out_degree', reverse=False, weight_key='score',
                   result_folder='./results/02_switchtfi/endocrine/alpha', fn_prefix='outdeg_')
    bdf = rank_tfs(grn=bgrn, centrality_measure='out_degree', reverse=False, weight_key='score',
                   result_folder='./results/02_switchtfi/endocrine/beta', fn_prefix='outdeg_')
    erydf = rank_tfs(grn=erygrn, centrality_measure='out_degree', reverse=False, weight_key='score',
                     result_folder='./results/02_switchtfi/hematopoiesis', fn_prefix='outdeg_')


if __name__ == '__main__':

    main_endocrine()

    main_hematopoiesis()

    main_compute_outdegree_tf_rankings()

    print('done')

