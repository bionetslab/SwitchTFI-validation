
import scanpy as sc
import scvelo as scv
import numpy as np
import os

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


if __name__ == '__main__':

    main_endocrine()

    main_hematopoiesis()

    print('done')






