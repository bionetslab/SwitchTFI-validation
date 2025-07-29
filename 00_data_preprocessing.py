
import os
import numpy as np
import scanpy as sc
import scvelo as scv

from preprocessing.data_pipeline import data_pipeline, subset_obs


def main_endocrine():

    # ### Load anndata object as provided by scvelo
    adata = scv.datasets.pancreas('./data/anndata/endocrinogenesis_day15.h5ad')

    # ### Set parameters
    verbosity = 1
    plot = False
    progenitor_name = 'Pre-endocrine'
    offspring_names = ['Alpha', 'Beta']
    res_dir = './data/anndata/'
    os.makedirs(res_dir, exist_ok=True)

    # ### Run preprocessing for Pre-endocrine-beta-cell data and Pre-endocrine-alpha-cell data
    for offspring in offspring_names:
        print(f'###### Preprocessing: {progenitor_name}-{offspring} ######')

        res_fn = f'{progenitor_name.lower()}_{offspring.lower()}.h5ad'
        res_p = os.path.join(res_dir, res_fn)

        data_pipeline(
            adata=adata.copy(),
            res_path=res_p,
            obs_subset_keys=('clusters', [progenitor_name, offspring]),
            species='mus musculus',
            pct_counts_mt_threshold=8.0,
            cor_amb_rna=True,
            gene_expr_threshold=(10, 'n_cells'),
            top_k_deviant=6000,
            diffusion_time=1,
            verbosity=verbosity,
            plot=plot
        )


def main_hematopoiesis():

    # ### Define helper function
    def rename_mus_musculus_hematopoiesis_cells(
            data: sc.AnnData,
            cluster_obs_key: str = 'paul15_clusters'
    ) -> sc.AnnData:

        # Get cell type annotations
        cell_types = data.obs[cluster_obs_key].to_numpy()

        # Rename fine-grained annotations
        name_dict = {
            '19Lymph': 'lymphocytes_19',
            '18Eos': 'eosinophils_18',
            '17Neu': 'neutrophils_17',
            '16Neu': 'neutrophils_16',
            '15Mo': 'monocytes_15',
            '14Mo': 'monocytes_14',
            '13Baso': 'basophils_13',
            '12Baso': 'basophils_12',
            '11DC': 'dendritic_cells_11',
            '10GMP': 'granulocyte/macrophage_progenitors_10',
            '9GMP': 'granulocyte/macrophage_progenitors_9',
            '8Mk': 'megakaryocytes_8',
            '7MEP': 'megakaryocyte/erythrocyte_progenitors_7',
            '6Ery': 'erythrocytes_6',
            '5Ery': 'erythrocytes_5',
            '4Ery': 'erythrocytes_4',
            '3Ery': 'erythrocytes_3',
            '2Ery': 'erythrocytes_2',
            '1Ery': 'erythrocytes_1',
        }

        dummy = np.empty(data.n_obs, dtype=object)
        for key, item in name_dict.items():
            dummy[cell_types == key] = item
        data.obs[f'{cluster_obs_key}_renamed'] = dummy

        # Add coarse-grained annotations
        dummy2 = np.empty(data.n_obs, dtype=object)
        cluster_list_dict = {
            'erythrocytes': ['1Ery', '2Ery', '3Ery', '4Ery', '5Ery', '6Ery'],
            'megakaryocyte/erythrocyte_progenitors': ['7MEP'],
            'megakaryocytes': ['8Mk'],
            'granulocyte/macrophage_progenitors': ['9GMP', '10GMP'],
            'dendritic_cells': ['11DC'],
            'basophils': ['12Baso', '13Baso'],
            'monocytes': ['14Mo', '15Mo'],
            'neutrophils': ['16Neu', '17Neu'],
            'eosinophils': ['18Eos'],
            'lymphocytes': ['19Lymph']
        }

        for key, val in cluster_list_dict.items():
            dummy2[np.isin(cell_types, val)] = key

        data.obs[f'{cluster_obs_key}_coarse'] = dummy2

        return data

   # ### Load data set as provided by scanpy
    adata = sc.datasets.paul15()

    # ### Rename clusters to more readable names
    adata = rename_mus_musculus_hematopoiesis_cells(data=adata, cluster_obs_key='paul15_clusters')

    # ### Subset adata to erythrocyte lineage
    adata = subset_obs(
        adata=adata,
        obs_key='paul15_clusters',
        keep_vals=['1Ery', '2Ery', '3Ery', '4Ery', '5Ery', '6Ery', '7MEP']
    )

    # ### Define and annotate progenitor and offspring clusters
    prog_off_dict = {'prog': ['7MEP', '6Ery', '5Ery'], 'off': ['1Ery', '2Ery', '3Ery', '4Ery']}
    cell_types = adata.obs['paul15_clusters'].to_numpy()
    prog_off_anno = np.empty(adata.n_obs, dtype=object)
    for key, val in prog_off_dict.items():
        prog_off_anno[np.isin(cell_types, val)] = key
    adata.obs['prog_off'] = prog_off_anno

    # ### Set parameters for preprocessing and preprocess
    verbosity = 1
    plot = False
    res_p = './data/anndata/erythrocytes.h5ad'

    data_pipeline(
        adata=adata,
        res_path=res_p,
        obs_subset_keys=None,
        species='mus musculus',
        pct_counts_mt_threshold=8.0,
        cor_amb_rna=False,
        gene_expr_threshold=(10, 'n_cells'),
        verbosity=verbosity,
        plot=plot
    )


def main_tox():

    # 'sample':  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    # 'sample name':  {'ko_liver_1', 'ko_spleen_2', 'wt_spleen_2', 'ko_liver_2', 'ko_liver_3', 'wt_spleen_1', 'wt_liver_1', 'wt_liver_3', 'ko_spleen_1', 'wt_spleen_3', 'wt_liver_2', 'ko_spleen_3'}
    # 'subject': {0, 1, 2}, which mouse
    # 'cluster': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, see table in slides
    # Tissue: {0: 'spleen', 1: 'liver'}
    # Time: {0: 'd10', 1: 'd20'}
    # Genotype: {0: 'ko', 1: 'wt'}
    # Info also stored in .obs_names (e.g. wt_spleen_1_d10_id)

    data_dir = './data/anndata'

    subset_info = {
        'tissue': {'spleen': 0, 'liver': 1},
        'time': {'d10': 0, 'd20': 1},
        'genotype': {'wt': 1, 'ko': 0},
        'cluster': {1: 1, 2: 2}
    }

    # Define subsets ([tissue], [time], [genotype], [cluster])
    subsets = [
        {'tissue': ['spleen'], 'time': ['d10'], 'genotype': ['wt'], 'cluster': [1, 2]},
        {'tissue': ['spleen'], 'time': ['d10'], 'genotype': ['ko'], 'cluster': [1, 2]}
    ]

    load_full_dataset = False

    # ### Load full dataset and subset (tissue spleen, day 10)
    if load_full_dataset:
        toxdata_full = sc.read(os.path.join(data_dir, 'tox_ko_wt.h5ad'))

        toxdata_full.raw = None  # delete the raw to avoid error when saving

        # Iteratively subset the data
        for subset in subsets:
            mask = np.ones(len(toxdata_full), dtype=bool)
            parts = []

            for key, val in subset.items():
                og_vals = [subset_info[key][e] for e in val]
                mask &= toxdata_full.obs[key].isin(og_vals)
                parts.append(f'{key}_' + ''.join(map(str, val)))

            # Subset only once
            toxdata = toxdata_full[mask, :].copy()
            filename = 'toxdata_' + '_'.join(parts) + '.h5ad'

            toxdata.write_h5ad(os.path.join(data_dir, filename))

            del toxdata

    for subset in subsets:
        filename = 'toxdata_' + '_'.join(f'{key}_{"".join(map(str, val))}' for key, val in subset.items()) + '.h5ad'

        print('# ### Preprocessing: ', filename, '### #')

        toxdata = sc.read(os.path.join(data_dir, filename))

        # Add progenitor-offspring annotations based on cluster membership
        prog_off_dict = {'prog': [1], 'off': [2]}
        cluster_labels = toxdata.obs['cluster'].to_numpy()
        prog_off_anno = np.empty(toxdata.n_obs, dtype=object)
        for key, val in prog_off_dict.items():
            prog_off_anno[np.isin(cluster_labels, val)] = key
        toxdata.obs['prog_off'] = prog_off_anno

        # Preprocess
        data_pipeline(
            adata=toxdata,
            res_path=os.path.join(data_dir, filename[:-5] + '_processed.h5ad'),
            obs_subset_keys=None,
            species='mus musculus',
            pct_counts_mt_threshold=8.0,
            cor_amb_rna=False,
            gene_expr_threshold=(10, 'n_cells'),
            top_k_deviant=6000,
            diffusion_time=1,
            verbosity=1,
            plot=True
        )


if __name__ == '__main__':

    # main_endocrine()

    # main_hematopoiesis()

    # main_tox()

    print('done')






