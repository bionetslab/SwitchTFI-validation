
import os
import glob
import math

import pandas as pd
import scanpy as sc


from grn_inf.grn_inference import pyscenic_pipeline, combine_grns


def main_endocrine():
    # ### Define paths to files where TFs are stored
    tf_file = './data/tf/mus_musculus/allTFs_mm.txt'

    # ### Define paths to auxiliary annotation files needed for GRN inference with Scenic

    # ## Old (deprecated): 'mm9_mc9nr'
    # db_file = './data/scenic_aux_data/databases/mouse/mm9/' \
    #           "mm9-*.mc9nr.genes_vs_motifs.rankings.feather"
    # anno_file = './data/scenic_aux_data/motif2tf_annotations/' \
    #             'motifs-v9-nr.mgi-m0.001-o0.0.tbl'

    # New (recent): 'mm10_mc_v10_clust'
    db_file = './data/scenic_aux_data/databases/mouse/mm10/' \
              'mc_v10_clust/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'
    anno_file = './data/scenic_aux_data/motif2tf_annotations/' \
                'motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl'

    # ### Set parameters and perform GRN inference with Scenic
    base_res_p = './results/01_grn_inf/endocrine'
    n_grns = 18
    cell_types = ['alpha', 'beta']
    for cell_type in cell_types:
        print(f'###### Starting Pyscenic pipeline for {cell_type} ######')
        adata = sc.read_h5ad(f'./data/anndata/pre-endocrine_{cell_type}.h5ad')

        for i in range(n_grns):
            # Run GRN inference on all genes
            res_dir = os.path.join(base_res_p, cell_type)
            pyscenic_pipeline(
                adata=adata.copy(),
                layer_key='scaled_log1p_norm',
                tf_file=tf_file,
                result_folder=res_dir,
                database_path=db_file,
                motif_annotations_path=anno_file,
                grn_inf_method='grnboost2',
                fn_prefix=f'{i:02d}_{cell_type}_',
                verbosity=1,
                plot=False
            )

    # ### Combine the 18 individual Scenic GRNs into one
    # Edges that occur in >= n_occurrence_threshold individual GRNs are retained
    n_occurrence_threshold = 9

    print('### Combining Pyscenic GRNs')
    for cell_type in cell_types:
        grn_list = []
        res_dir = os.path.join(base_res_p, cell_type)
        # Get list of paths to csv files
        csv_files = glob.glob(res_dir + '/*_pruned_grn.csv')
        for csv_file in csv_files:
            grn_list.append(pd.read_csv(csv_file, index_col=[0]))

        combine_grns(
            grn_list=grn_list,
            n_occurrence_thresh=n_occurrence_threshold,
            result_folder=res_dir,
            verbosity=1,
            fn_prefix=f'ngrnthresh{n_occurrence_threshold}_{cell_type}_pyscenic_'
        )

    # ### Combine GrnBoost2 GRNs into one (not needed afterwards)
    print('### Combining Grnboost2 GRNS')
    for cell_type in cell_types:
        grn_list = []
        res_dir = os.path.join(base_res_p, cell_type)
        # Get list of paths to csv files
        csv_files = glob.glob(res_dir + '/*_basic_grn.csv')
        for csv_file in csv_files:
            grn = pd.read_csv(csv_file, sep='\t')
            # Extract top 1% of important edges
            top_1_perc = math.ceil(grn.shape[0] * 0.01)
            grn_list.append(grn[0:top_1_perc])

        combine_grns(
            grn_list=grn_list,
            n_occurrence_thresh=n_occurrence_threshold,
            result_folder=res_dir,
            verbosity=1,
            fn_prefix=f'ngrnthresh{n_occurrence_threshold}_{cell_type}_grnboost_'
        )


def main_hematopoiesis():

    # ### Define paths to files where TFs are stored
    tf_file = './data/tf/mus_musculus/allTFs_mm.txt'

    # ### Define paths to auxiliary annotation files needed for GRN inference with Scenic

    # ## Old (deprecated): 'mm9_mc9nr'
    # db_file = ./data/scenic_aux_data/databases/mouse/mm9/' \
    #           "mm9-*.mc9nr.genes_vs_motifs.rankings.feather"
    # anno_file = ./data/scenic_aux_data/motif2tf_annotations/' \
    #             'motifs-v9-nr.mgi-m0.001-o0.0.tbl'

    # New (recent): 'mm10_mc_v10_clust'
    db_file = './data/scenic_aux_data/databases/mouse/mm10/' \
              'mc_v10_clust/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'
    anno_file = './data/scenic_aux_data/motif2tf_annotations/' \
                'motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl'

    # ### Load AnnData wist scRNA-seq data
    adata = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')

    # ### Set parameters and perform GRN inference with Scenic
    base_res_p = './results/01_grn_inf/hematopoiesis'
    n_grns = 18

    for i in range(n_grns):
        pyscenic_pipeline(
            adata=adata.copy(),
            layer_key='scaled_log1p_norm',
            tf_file=tf_file,
            result_folder=base_res_p,
            database_path=db_file,
            motif_annotations_path=anno_file,
            grn_inf_method='grnboost2',
            fn_prefix=f'{i:02d}_erythrocytes_',
            verbosity=1,
            plot=False
        )

    # ### Combine the 18 individual Scenic GRNs into one
    # Edges that occur in >= n_occurrence_threshold individual GRNs are retained
    print('### Combining Pyscenic GRNs')
    n_occurrence_threshold = 9
    grn_list = []
    # Get list of paths to csv files
    csv_files = glob.glob(base_res_p + '/*_erythrocytes_pruned_grn.csv')
    for csv_file in csv_files:
        grn_list.append(pd.read_csv(csv_file, index_col=[0]))

    combine_grns(
        grn_list=grn_list,
        n_occurrence_thresh=n_occurrence_threshold,
        result_folder=base_res_p,
        verbosity=1,
        fn_prefix=f'ngrnthresh{n_occurrence_threshold}_erythrocytes_pyscenic_'
    )

    # ### Combine GrnBoost2 GRNs into one (not needed afterwards)
    print('### Combining Pyscenic GRNs')
    grn_list = []
    csv_files = glob.glob(base_res_p + '/*_erythrocytes_basic_grn.csv')

    for csv_file in csv_files:
        grn = pd.read_csv(csv_file, sep='\t')
        # Extract top 1% of important edges
        top_1_perc = math.ceil(grn.shape[0] * 0.01)
        grn_list.append(grn[0:top_1_perc])

    combine_grns(
        grn_list=grn_list,
        n_occurrence_thresh=n_occurrence_threshold,
        result_folder=base_res_p,
        verbosity=1,
        fn_prefix=f'ngrnthresh{n_occurrence_threshold}_erythrocytes_grnboost_'
    )


if __name__ == '__main__':

    # main_endocrine()

    # main_hematopoiesis()

    print('done')
