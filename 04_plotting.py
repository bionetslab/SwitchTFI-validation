
import networkx as nx
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import os
from scipy.stats import pearsonr

from switchtfi.utils import load_grn_json
from validation.val_utils import combine_p_vals_fisher


def plot_step_fct_and_trends():
    # ### Script for plotting Figure 2

    from validation.plotting import plot_step_function, plot_gam_gene_trend, plot_gam_gene_trend_heatmap
    from switchtfi.utils import get_regulons

    # Load AnnData
    bdata = sc.read_h5ad('./results/03_validation/anndata/trend_pre-endocrine_beta.h5ad')

    # Load unpruned and pruned GRNs
    grn = load_grn_json('./results/02_switchtfi/endocrine/beta/grn.json')
    pgrn = load_grn_json('./results/02_switchtfi/endocrine/beta/sigpvals_wy0.05_grn.json')

    grn = grn.sort_values(by=['weight'], axis=0, ascending=False)
    grn = grn.reset_index(drop=True)

    print(grn.head(20))
    print(pgrn.head(20))

    # Load results dataframe
    res_tfs = pd.read_csv('./results/02_switchtfi/endocrine/beta/ranked_tfs.csv', index_col=[0])

    fig = plt.figure(figsize=(12, 13), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AABB
        AABB
        CDEF
        GGGG
        HHII
        HHII
        """
    )

    layer_key = 'magic_imputed'
    title_fs = 20
    ax_fs = 18
    legend_fs = 18
    gene_trend_anno_fs = 14
    letter_fs = 20

    e0 = tuple(grn[['TF', 'target']].iloc[1])
    e1 = tuple(grn[['TF', 'target']].iloc[grn.shape[0] - 1])
    print(e0, e1)
    
    plot_step_function(adata=bdata,
                       grn=grn,
                       which=e0,
                       layer_key=layer_key,
                       plot_threshold=True,
                       ax_label_fontsize=ax_fs,
                       title_fontsize=title_fs,
                       legend_fontsize=legend_fs,
                       legend_loc='upper right',
                       show=False,
                       axs=axd['A'])

    plot_step_function(adata=bdata,
                       grn=grn,
                       which=e1,
                       layer_key=layer_key,
                       plot_threshold=True,
                       ax_label_fontsize=ax_fs,
                       title_fontsize=title_fs,
                       legend_fontsize=legend_fs,
                       legend_loc='upper center',
                       show=False,
                       axs=axd['B'])

    plot_gam_gene_trend(adata=bdata,
                        gene_names=[e0[0]],
                        plot_cells=True,
                        ax_label_fontsize=ax_fs,
                        show=False,
                        axs=axd['C'],
                        layer_key='magic_imputed')

    plot_gam_gene_trend(adata=bdata,
                        gene_names=[e0[1]],
                        plot_cells=True,
                        ax_label_fontsize=ax_fs,
                        show=False,
                        axs=axd['D'],
                        layer_key='magic_imputed')

    plot_gam_gene_trend(adata=bdata,
                        gene_names=[e1[0]],
                        plot_cells=True,
                        ax_label_fontsize=ax_fs,
                        show=False,
                        axs=axd['E'],
                        layer_key='magic_imputed')

    plot_gam_gene_trend(adata=bdata,
                        gene_names=[e1[1]],
                        plot_cells=True,
                        ax_label_fontsize=ax_fs,
                        show=False,
                        axs=axd['F'],
                        layer_key='magic_imputed')

    tf_names = res_tfs['gene'].tolist()[0:10]

    plot_gam_gene_trend_heatmap(adata=bdata,
                                gene_names=tf_names,
                                use_trend=True,
                                gene_names_fontsize=gene_trend_anno_fs,
                                ax_label_fontsize=ax_fs,
                                show=False,
                                axs=axd['G'],
                                colorbar_pad=0.0)

    regulons = get_regulons(grn=pgrn, gene_names=['Pdx1', 'Pax4'])
    targets0 = regulons['Pdx1']['targets']
    targets1 = regulons['Pax4']['targets']

    plot_gam_gene_trend_heatmap(adata=bdata,
                                gene_names=targets0,
                                use_trend=True,
                                annotate_gene_names=False,
                                ax_label_fontsize=ax_fs,
                                show=False,
                                axs=axd['H'],
                                plot_colorbar=False,
                                title='Targets of TF Pdx1',
                                title_fontsize=title_fs)

    plot_gam_gene_trend_heatmap(adata=bdata,
                                gene_names=targets1,
                                use_trend=True,
                                annotate_gene_names=False,
                                ax_label_fontsize=ax_fs,
                                show=False,
                                axs=axd['I'],
                                colorbar_pad=0.0,
                                title='Targets of TF Pax4',
                                title_fontsize=title_fs)

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        # ax = fig.add_subplot(axd[label])
        # ax.annotate(label, xy=(0.1, 1.1), xycoords='axes fraction', ha='center', fontsize=16)
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 0.95, label, transform=ax.transAxes + trans,
                fontsize=letter_fs, va='bottom', fontfamily='sans-serif', fontweight='bold')

    # plt.show()
    plt.savefig('./results/04_plots/stepfct_trends_beta.png', dpi=fig.dpi)


def plot_quantitative_analyses():
    # ### Script for plotting Figure 3

    from validation.plotting import plot_defrac_lineplot, plot_sdeqvals_vs_weight, plot_cc_score_hist
    from validation.val_utils import compare_grn_vs_rand_background
    from switchtfi.tf_ranking import grn_to_nx

    # Load AnnData
    bdata = sc.read_h5ad('./results/03_validation/anndata/switchde_magic_nozeroinflated_pre-endocrine_beta.h5ad')
    erydata = sc.read_h5ad('./results/03_validation/anndata/switchde_magic_nozeroinflated_erythrocytes.h5ad')
    # bdata = sc.read_h5ad('./results/03_validation/anndata/switchde_log1p_norm_zeroinflated_pre-endocrine_beta.h5ad')

    # Load result dataframes
    bgrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)
    bbase_grn = load_grn_json('./results/02_switchtfi/endocrine/beta/grn.json')
    erygrn = pd.read_csv('./results/02_switchtfi/hematopoiesis/grn.csv', index_col=0)
    erybase_grn = load_grn_json('./results/02_switchtfi/hematopoiesis/grn.json')

    # Combine ptDE q-values using Fishers method for each edge (TF, target) => edgewise ptDE q-values

    # Add minimal eps to 0 q-values for numeric stability, then combine q-values => joint q-value per edge
    bq = bdata.var['switchde_qval'].to_numpy()
    bq[bq == 0] = np.finfo(np.float64).eps
    bdata.var['switchde_qval'] = bq

    eryq = erydata.var['switchde_qval'].to_numpy()
    eryq[eryq == 0] = np.finfo(np.float64).eps
    erydata.var['switchde_qval'] = eryq

    bbase_grn = combine_p_vals_fisher(grn=bbase_grn, anndata=bdata)
    bminw = bgrn['weight'].to_numpy().min()

    erybase_grn = combine_p_vals_fisher(grn=erybase_grn, anndata=erydata)
    eryminw = erygrn['weight'].to_numpy().min()

    # Compute correlation
    bcorr = pearsonr(bbase_grn['weight'].to_numpy(), bbase_grn['switchde_qvals_combined_fisher'].to_numpy())
    erycorr = pearsonr(erybase_grn['weight'].to_numpy(), erybase_grn['switchde_qvals_combined_fisher'].to_numpy())

    print('###### Correlations(weights, combined q-vals): ')
    print(f'### Beta: {bcorr}')
    print(f'### Erythrocytes: {erycorr}')

    # Again for plotting add eps
    bqcomb = bbase_grn['switchde_qvals_combined_fisher'].to_numpy()
    bqcomb[bqcomb == 0] = np.finfo(np.float64).eps
    bbase_grn['switchde_qvals_combined_fisher'] = bqcomb

    eryqcomb = erybase_grn['switchde_qvals_combined_fisher'].to_numpy()
    eryqcomb[eryqcomb == 0] = np.finfo(np.float64).eps
    erybase_grn['switchde_qvals_combined_fisher'] = eryqcomb

    fig = plt.figure(figsize=(12, 8), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        ABC
        DEF
        """
    )

    # General
    title_fs = 16
    ax_fs = 16
    legend_fs = 14
    letter_fs = 20

    # q-vals vs weight
    dot_size0 = 6
    # defrac lineplot
    dot_size1 = 14

    plot_sdeqvals_vs_weight(
        grn=bbase_grn, comb_qval_alpha=0.01, weight_threshold=bminw, title=r'$\beta$-cell transition',
        legend_pos='upper center', size=dot_size0, ax_label_fontsize=ax_fs, title_fontsize=title_fs,
        legend_fontsize=legend_fs, axs=axd['A'])
    plot_sdeqvals_vs_weight(
        grn=erybase_grn, comb_qval_alpha=0.01, weight_threshold=eryminw, title='Erythrocyte differentiation',
        legend_pos='upper center', size=dot_size0, ax_label_fontsize=ax_fs, title_fontsize=title_fs,
        legend_fontsize=legend_fs, axs=axd['D'])

    plot_defrac_lineplot(
        adata=bdata, base_grn=bbase_grn, pruned_grn=bgrn, switchde_alpha=0.01, title=r'$\beta$-cell transition',
        size=dot_size1, ax_label_fontsize=ax_fs, title_fontsize=title_fs, legend_fontsize=legend_fs,
        legend_pos=(0.5, 0.65), axs=axd['B'], verbosity=1)
    plot_defrac_lineplot(
        adata=erydata, base_grn=erybase_grn, pruned_grn=erygrn, switchde_alpha=0.01,
        title='Erythrocyte differentiation', size=dot_size1, ax_label_fontsize=ax_fs, title_fontsize=title_fs,
        legend_fontsize=legend_fs, legend_pos=(0.5, 0.65), axs=axd['E'], verbosity=1)

    # ### Plot histograms of percentage of Lcc nodes in randomly sampled GRNs of same size as transition GRN
    n = 10000
    bccs_list, bn_vert_list = compare_grn_vs_rand_background(base_grn=bbase_grn, transition_grn=bgrn, n=n)
    eryccs_list, eryn_vert_list = compare_grn_vs_rand_background(base_grn=erybase_grn, transition_grn=erygrn, n=n)

    btgrn = grn_to_nx(grn=bgrn).to_undirected()
    erytgrn = grn_to_nx(grn=erygrn).to_undirected()
    bccs = list(nx.connected_components(btgrn))
    eryccs = list(nx.connected_components(erytgrn))

    plot_cc_score_hist(
        ccs=bccs, n_vert=btgrn.number_of_nodes(), ccs_list=bccs_list, n_vert_list=bn_vert_list,
        titel=r'$\beta$-cell transition', ax_label_fontsize=ax_fs, title_fontsize=title_fs,
        legend_fontsize=legend_fs, axs=axd['C'])
    plot_cc_score_hist(
        ccs=eryccs, n_vert=erytgrn.number_of_nodes(), ccs_list=eryccs_list, n_vert_list=eryn_vert_list,
        titel='Erythrocyte differentiation', ax_label_fontsize=ax_fs, title_fontsize=title_fs,
        legend_fontsize=legend_fs, axs=axd['F'])

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=letter_fs, va='bottom', fontfamily='sans-serif', fontweight='bold')

    # plt.show()
    plt.savefig('./results/04_plots/quantitative_beta_ery.png', dpi=fig.dpi)


def plot_qualitative_analysis():
    # ### Script for plotting Figure 4

    from validation.plotting import plot_circled_tfs, plot_enrichr_results, plot_gam_gene_trend_heatmap

    # ### Load ranked TFs for alpha- and beta-cell transition
    atfs = pd.read_csv('./results/02_switchtfi/endocrine/alpha/ranked_tfs.csv', index_col=[0])
    btfs = pd.read_csv('./results/02_switchtfi/endocrine/beta/ranked_tfs.csv', index_col=[0])

    p = False
    if p:
        top_k = 10
        print(f'### Top {top_k} TFs, alpha:')
        for tf in atfs['gene'][0:top_k].tolist():
            print(tf)
        print(f'### Top {top_k} TFs, beta:')
        for tf in btfs['gene'][0:top_k].tolist():
            print(tf)

    # ### Load AnnData with gene trends for alpha-cell transition
    adata = sc.read_h5ad('./results/03_validation/anndata/trend_pre-endocrine_alpha.h5ad')

    # ### Load ENRICHR GSEA results for beta-cell transition driver TFs and targets of Ybx1
    b_gobp = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/beta_pr_GO_Biological_Process_2023_table.txt'),
        delimiter='\t'
    )
    b_reactome = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/beta_pr_Reactome_2022_table.txt'),
        delimiter='\t'
    )

    # ### Plot results for Beta dataset ### #
    fig = plt.figure(figsize=(13, 9), constrained_layout=True, dpi=300)
    mosaic = [
        ['A', 'A', '.'],
        ['A', 'A', 'B'],
        ['A', 'A', 'B'],
        ['A', 'A', '.'],
        ['C', 'C', 'C'],
    ]

    axd = fig.subplot_mosaic(
        mosaic=mosaic,
        gridspec_kw={"height_ratios": [1/4, 1/4, 1/4, 1/4, 3], }
                     # "width_ratios": [3, 1]},
    )

    # General
    title_fs = 20
    ax_fs = 16
    letter_fs = 20
    # Top 10 driver TFs plot
    gene_name_fs = 16
    # Gene trends plot
    gene_trend_anno_fs = 14
    # Enrichr results plots
    term_fs = 16
    legend_fs = 16

    # ### Plot Top 10 driver TFs
    plot_circled_tfs(
        res_df=atfs, topk=10, fontsize=gene_name_fs, res_df2=btfs, title=None, title_fontsize=None, plottwoinone=True,
        y_ticks=(r'$\alpha$:', r'$\beta$:'), ax_label_fontsize=ax_fs, axs=axd['A'])

    # ### Plot gene trends of top 10 alpha-cell transition driver TFs
    plot_gam_gene_trend_heatmap(
        adata=adata,
        gene_names=atfs['gene'][0:10].tolist(),
        use_trend=True,
        annotate_gene_names=True,
        gene_names_fontsize=gene_trend_anno_fs,
        ax_label_fontsize=ax_fs,
        show=False,
        axs=axd['B'],
        plot_colorbar=False,
        title=r'$\alpha$-cell transition',
        title_fontsize=title_fs)

    # ### Plot GSEA results for top 10 beta-cell transition driver TFs
    plot_enrichr_results(
        res_dfs=[b_gobp, b_reactome],
        x='Adjusted P-value',
        top_k=[6, 6],
        reference_db_names=['GO_Biological_Process_2023', 'Reactome_2022'],
        title=r'$\beta$-cell transition',
        title_fontsize=title_fs,
        term_fontsize=term_fs,
        ax_label_fontsize=ax_fs,
        legend_fontsize=legend_fs,
        axs=axd['C']
    )

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=letter_fs, va='bottom', fontfamily='sans-serif', fontweight='bold')

    # plt.show()
    plt.savefig('./results/04_plots/qualitative_beta.png', dpi=fig.dpi)


def plot_hypotheses_generation_ybx1():
    # ### Script for plotting Figure 5

    from validation.plotting import plot_enrichr_results
    from switchtfi.plotting import plot_regulon

    # ### Load transition GRN of beta-cell transition
    bgrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)

    # ### Load ENRICHR GSEA results for beta-cell transition driver TFs and targets of Ybx1
    ybx1_gobp = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/beta_ybx1_targets_GO_Biological_Process_2023_table.txt'),
        delimiter='\t'
    )
    ybx1_gocc = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/beta_ybx1_targets_GO_Cellular_Component_2023_table.txt'),
        delimiter='\t'
    )
    ybx1_reactome = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/beta_ybx1_targets_Reactome_2022_table.txt'),
        delimiter='\t'
    )

    # ### Plot results for Beta dataset ### #
    fig = plt.figure(figsize=(13, 8), constrained_layout=True, dpi=300)
    mosaic = [
        ['A', 'A', 'A', 'A'],
        ['B', 'B', 'B', 'B']
    ]

    axd = fig.subplot_mosaic(
        mosaic=mosaic,
        gridspec_kw={'height_ratios': [5, 6], }
                     # "width_ratios": [3, 1]},
    )

    # General
    ax_fs = 16
    letter_fs = 20
    # Enrichr results plots
    term_fs = 16
    legend_fs = 16
    # Regulon
    nodes_size = 1000
    gene_fs = 16

    # ### Plot the top 20 targets of Ybx1
    plot_regulon(
        grn=bgrn, tf='Ybx1', sort_by='score', top_k=20, title=None, title_fontsize=None,
        font_size=gene_fs, node_size=nodes_size, show=False, dpi=300, axs=axd['A'])

    # ### Plot the GSEA results for the top 20 targets of Ybx1
    plot_enrichr_results(
        res_dfs=[ybx1_gobp, ybx1_reactome, ybx1_gocc],
        x='Adjusted P-value',
        top_k=[6, 3, 3],
        reference_db_names=['GO_Biological_Process_2023', 'Reactome_2022', 'GO_Cellular_Component_2023'],
        title=None,
        title_fontsize=None,
        term_fontsize=term_fs,
        truncate_term_k=80,
        ax_label_fontsize=ax_fs,
        legend_fontsize=legend_fs,
        axs=axd['B']
    )

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=letter_fs, va='bottom', fontfamily='sans-serif', fontweight='bold')

    # plt.show()
    plt.savefig('./results/04_plots/hypothesis_gen_ybx1.png', dpi=fig.dpi)


def plot_method_comparison():
    # ### Script for plotting Figure 6

    from validation.plotting import plot_upset_plot
    from validation.plotting import plot_defrac_in_top_k_lineplot, plot_digest_results

    # Load AnnData
    bdata = sc.read_h5ad('./results/03_validation/anndata/switchde_log1p_norm_zeroinflated_pre-endocrine_beta.h5ad')
    # bdata = sc.read_h5ad('./results/03_validation/anndata/switchde_magic_nozeroinflated_pre-endocrine_beta.h5ad')

    # sc.pl.umap(bdata, color='palantir_pseudotime', show=False)
    # plt.savefig('./results/04_plots/bumap.png')

    # ### Load result dataframes
    btfs = pd.read_csv('./results/02_switchtfi/endocrine/beta/ranked_tfs.csv', index_col=[0])
    btfs_outdeg = pd.read_csv('./results/02_switchtfi/endocrine/beta/outdeg_ranked_tfs.csv', index_col=[0])

    res_base_p = './results/03_validation/driver_genes/'
    bcr_genes = pd.read_csv(os.path.join(res_base_p, 'beta_cellrank_driver_genes.csv'), index_col=[0])
    bspjc_genes = pd.read_csv(os.path.join(res_base_p, 'beta_splicejac_driver_genes.csv'), index_col=[0])
    bdrivaer_genes = pd.read_csv(os.path.join(res_base_p, 'beta_drivaer_driver_genes.csv'), index_col=[0])

    b_res_list = [bcr_genes, bspjc_genes, bdrivaer_genes, btfs_outdeg, btfs]

    # Define names of method and color to be used in plots
    name_list = ['CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI PageRank']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#f52891cc', '#d62728']


    # ### Define paths to DIGEST results
    digest_base_p = './results/03_validation/digest_results'
    bp1 = os.path.join(digest_base_p, 'top20_beta_cellrank/542a7e3c-f24d-4856-9a7f-1e2b7c777e8e_result.json')
    bp2 = os.path.join(digest_base_p, 'top20_beta_splicejac/9706eeaa-ba4d-4189-aa71-a6248b60a0f6_result.json')
    bp3 = os.path.join(digest_base_p, 'top20_beta_drivaer/59f0306c-3da6-49bb-8391-55a8c59f1e70_result.json')
    bp4 = os.path.join(digest_base_p, 'top20_beta_switchtfi_outdeg/6adfd6ec-b32d-418e-8451-0e81f1c89914_result.json')
    bp5 = os.path.join(digest_base_p, 'top20_beta_switchtfi/c0932ef3-f281-46d5-acdd-358551495dbe_result.json')
    bpl = [bp1, bp2, bp3, bp4, bp5]

    # Same for erythrocytes, except for no splicejac
    erydata = sc.read_h5ad('./results/03_validation/anndata/switchde_log1p_norm_zeroinflated_erythrocytes.h5ad')
    # erydata = sc.read_h5ad('./results/03_validation/anndata/switchde_magic_nozeroinflated_erythrocytes.h5ad')
    erytfs = pd.read_csv('./results/02_switchtfi/hematopoiesis/ranked_tfs.csv', index_col=[0])
    erytfs_outdeg = pd.read_csv('./results/02_switchtfi/hematopoiesis/outdeg_ranked_tfs.csv', index_col=[0])

    erycr_genes = pd.read_csv(os.path.join(res_base_p, 'ery_cellrank_driver_genes.csv'), index_col=[0])
    erydrivaer_genes = pd.read_csv(os.path.join(res_base_p, 'ery_drivaer_driver_genes.csv'), index_col=[0])

    ery_res_list = [erycr_genes, erydrivaer_genes, erytfs_outdeg, erytfs]

    eryp1 = os.path.join(digest_base_p, 'top20_ery_cellrank/a4077393-4050-42ed-bf1a-a9ab3ea3726b_result.json')
    eryp2 = os.path.join(digest_base_p, 'top20_ery_drivaer/0d0d89e4-ea15-4ee7-8cc2-da8fe8d39a4f_result.json')
    eryp3 = os.path.join(digest_base_p, 'top20_ery_switchtfi_outdeg/70becca4-53b6-4ede-a7f5-419b0e3da377_result.json')
    eryp4 = os.path.join(digest_base_p, 'top20_ery_switchtfi/4dfe96bb-48ff-4d23-b9c3-7e94f8bf3da0_result.json')
    erypl = [eryp1, eryp2, eryp3, eryp4]

    # Define names of method and color to be used in plots
    ery_name_list = ['CellRank', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI PageRank']
    ery_colors = ['#1f77b4', '#2ca02c', '#f52891cc', '#d62728']

    # ### Plot results for Beta dataset ###
    fig = plt.figure(figsize=(14, 8), constrained_layout=True, dpi=100)
    axd = fig.subplot_mosaic(
        [
            ['A', 'C', 'E'],  # ['A.β', 'B.β', 'C.β'],
            ['B', 'D', 'F'],  # ['A.ery', 'B.ery', 'C.ery'],
        ],
    )

    # General
    title_fs = 20
    ax_fs = 20
    legend_fs = 16
    letter_fs = 14
    # Digest res
    dataset_fs = 18
    plt_xlabel = False
    dotsize = 9

    # ### Plot Upsetplot
    plot_upset_plot(
        res_list=b_res_list,
        names=tuple(name_list),
        title=r'$\beta$-cell transition data',
        title_fontsize=title_fs,
        axs=axd['A'],
        plot_folder='./results/04_plots',
        fn_prefix='beta')
    plot_upset_plot(
        res_list=ery_res_list,
        names=tuple(ery_name_list),
        title='Erythrocyte differentiation data',
        title_fontsize=title_fs,
        axs=axd['B'],
        plot_folder='./results/04_plots',
        fn_prefix='ery')

    plot_defrac_in_top_k_lineplot(
        res_df_list=b_res_list,
        adata=bdata,
        switchde_alpha=0.01,
        max_top_k=20,
        interval=5,
        axs=axd['C'],
        title=r'$\beta$-cell transition data',
        title_fontsize=title_fs,
        ax_label_fontsize=ax_fs,
        legend_fontsize=legend_fs,
        show=False,
        names=name_list,
        markers=['s', 'd', '^', 'o', 'o'],
        palette=colors,
        hue_order=('CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI PageRank'),
        legend_loc='lower right')

    plot_defrac_in_top_k_lineplot(
        res_df_list=ery_res_list,
        adata=erydata,
        switchde_alpha=0.01,
        max_top_k=20,
        interval=5,
        axs=axd['D'],
        title='Erythrocyte differentiation data',
        title_fontsize=title_fs,
        ax_label_fontsize=ax_fs,
        legend_fontsize=legend_fs,
        show=False,
        names=ery_name_list,
        markers=['s', '^', 'o', 'o'],
        palette=ery_colors,
        hue_order=('CellRank', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI PageRank'),
        legend_loc='center right')

    plot_digest_results(
        digest_res_path_list=bpl,
        name_list=name_list,
        color_list=colors,
        title=r'$\beta$-cell transition data',
        size=dotsize,
        title_fontsize=title_fs,
        ax_label_fontsize=ax_fs,
        legend_fontsize=legend_fs,
        x_ticks_fontsize=dataset_fs,
        plot_xlabel=plt_xlabel,
        axs=axd['E'],
        verbosity=1)
    plot_digest_results(
        digest_res_path_list=erypl,
        name_list=ery_name_list,
        color_list=ery_colors,
        title='Erythrocyte differentiation data',
        size=dotsize,
        title_fontsize=title_fs,
        ax_label_fontsize=ax_fs,
        legend_fontsize=legend_fs,
        x_ticks_fontsize=dataset_fs,
        plot_xlabel=plt_xlabel,
        axs=axd['F'],
        verbosity=1)

    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=letter_fs, va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/method_comparison.png', dpi=fig.dpi)
    # plt.savefig('./results/04_plots/method_comparison_imputed.png', dpi=fig.dpi)


def plot_alpha_res_appendix():
    # ### Script for plotting Supplementary Figure 1

    from validation.plotting import (
        plot_defrac_lineplot, plot_sdeqvals_vs_weight, plot_cc_score_hist, plot_enrichr_results,
        plot_defrac_in_top_k_lineplot, plot_digest_results, plot_upset_plot
    )
    from validation.val_utils import compare_grn_vs_rand_background
    from switchtfi.tf_ranking import grn_to_nx

    # ### Load AnnData with switchde p-values
    adata = sc.read_h5ad('./results/03_validation/anndata/switchde_magic_nozeroinflated_pre-endocrine_alpha.h5ad')

    # ### Load input and transition GRN
    agrn = pd.read_csv('./results/02_switchtfi/endocrine/alpha/grn.csv', index_col=0)
    abase_grn = load_grn_json('./results/02_switchtfi/endocrine/alpha/grn.json')

    # ### Compute edg-wise combined ptDE q-values
    # Add minimal eps to 0 q-values for numeric stability, then combine q-values => joint q-value per edge
    aq = adata.var['switchde_qval'].to_numpy()
    aq[aq == 0] = np.finfo(np.float64).eps
    adata.var['switchde_qval'] = aq

    abase_grn = combine_p_vals_fisher(grn=abase_grn, anndata=adata)
    aminw = agrn['weight'].to_numpy().min()

    # Compute correlation between edge-weights (from SwitchTFI) and ptDE edge-wise combined q-values
    acorr = pearsonr(abase_grn['weight'].to_numpy(), abase_grn['switchde_qvals_combined_fisher'].to_numpy())
    print('###### Correlations(weights, combined q-vals): ')
    print(f'### Alpha: {acorr}')

    # Again for plotting add eps to the combined q-values
    aqcomb = abase_grn['switchde_qvals_combined_fisher'].to_numpy()
    aqcomb[aqcomb == 0] = np.finfo(np.float64).eps
    abase_grn['switchde_qvals_combined_fisher'] = aqcomb

    # ### Load ENRICHR GSEA results
    a_gobp = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/alpha_pr_GO_Biological_Process_2023_table.txt'),
        delimiter='\t'
    )
    a_reactome = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/alpha_pr_Reactome_2022_table.txt'),
        delimiter='\t'
    )

    # ### Load AnnData with ptDE p-values
    adata = sc.read_h5ad('./results/03_validation/anndata/switchde_log1p_norm_zeroinflated_pre-endocrine_alpha.h5ad')

    # ### Load result dataframes with predicted driver TFs from SwitchTFI and competitor methods
    atfs = pd.read_csv('./results/02_switchtfi/endocrine/alpha/ranked_tfs.csv', index_col=[0])
    atfs_outdeg = pd.read_csv('./results/02_switchtfi/endocrine/alpha/outdeg_ranked_tfs.csv', index_col=[0])

    res_base_p = './results/03_validation/driver_genes/'
    acr_genes = pd.read_csv(os.path.join(res_base_p, 'alpha_cellrank_driver_genes.csv'), index_col=[0])

    aspjc_genes = pd.read_csv(os.path.join(res_base_p, 'alpha_splicejac_driver_genes.csv'), index_col=[0])

    adrivaer_genes = pd.read_csv(os.path.join(res_base_p, 'alpha_drivaer_driver_genes.csv'), index_col=[0])

    a_res_list = [acr_genes, aspjc_genes, adrivaer_genes, atfs_outdeg, atfs]

    # ### Define paths to DIGEST results
    digest_base_p = './results/03_validation/digest_results'
    ap1 = os.path.join(digest_base_p, 'top20_alpha_cellrank/b8d14a52-62ad-49ca-871e-9b8456086669_result.json')
    ap2 = os.path.join(digest_base_p, 'top20_alpha_splicejac/694d78ec-0fcd-4055-84fc-d29c53ffbde5_result.json')
    ap3 = os.path.join(digest_base_p, 'top20_alpha_drivaer/c1c20b4f-4c58-4973-9cae-4fd9ced7f7d3_result.json')
    ap4 = os.path.join(digest_base_p, 'top20_alpha_switchtfi_outdeg/2a70a4b4-a78e-4ade-a2a6-e0a9c2d93901_result.json')
    ap5 = os.path.join(digest_base_p, 'top20_alpha_switchtfi/0abd6509-83f7-4952-bde6-717d9b6b0648_result.json')
    apl = [ap1, ap2, ap3, ap4, ap5]

    # ###  Define names of method and color to be used in method comparison plots
    name_list = ['CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI PageRank']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#f52891cc', '#d62728']

    fig = plt.figure(figsize=(12, 12), constrained_layout=True, dpi=100)
    axd = fig.subplot_mosaic(
        """
        ABC
        DDD
        EFG
        """
    )

    # General
    ax_fs = 20
    legend_fs = 16
    letter_fs = 14
    # q-vals vs weight
    dot_size0 = 6
    # defrac lineplot
    dot_size1 = 12
    # Enrichr
    term_fs = 14
    # Digest res
    dot_size2 = 9
    dataset_fs = 18
    plt_xlabel = False


    # ### Plot quantitative analysis
    plot_sdeqvals_vs_weight(grn=abase_grn, comb_qval_alpha=0.01, weight_threshold=aminw, title=None,
                            legend_fontsize=legend_fs, ax_label_fontsize=ax_fs, legend_pos='upper center',
                            size=dot_size0, axs=axd['A'])
    plot_defrac_lineplot(adata=adata, base_grn=abase_grn, pruned_grn=agrn, switchde_alpha=0.01, title=None,
                         legend_fontsize=legend_fs, ax_label_fontsize=ax_fs, size=dot_size1, axs=axd['B'], verbosity=1)

    # ### Plot histograms of percentage of Lcc nodes in randomly sampled GRNs of same size as transition GRN
    # First run comparison against random background model ...
    # n = 10000
    n = 100
    accs_list, an_vert_list = compare_grn_vs_rand_background(base_grn=abase_grn, transition_grn=agrn, n=n)
    atgrn = grn_to_nx(grn=agrn).to_undirected()
    bccs = list(nx.connected_components(atgrn))
    # ... then plot the results
    plot_cc_score_hist(ccs=bccs, n_vert=atgrn.number_of_nodes(), ccs_list=accs_list, n_vert_list=an_vert_list,
                       titel=None, ax_label_fontsize=ax_fs, legend_fontsize=legend_fs, axs=axd['C'])

    # ### Plot qualitative analysis
    # Plot GSEA results for top 10 beta-cell transition driver TFs
    plot_enrichr_results(
        res_dfs=[a_gobp, a_reactome],
        x='Adjusted P-value',
        top_k=[6, 6],
        reference_db_names=['GO_Biological_Process_2023', 'Reactome_2022'],
        term_fontsize=term_fs,
        ax_label_fontsize=ax_fs,
        legend_fontsize=legend_fs,
        axs=axd['D']
    )

    # ### Plot method comparison
    plot_defrac_in_top_k_lineplot(
        res_df_list=a_res_list,
        adata=adata,
        switchde_alpha=0.01,
        max_top_k=20,
        interval=5,
        axs=axd['E'],
        title=None,
        ax_label_fontsize=ax_fs,
        legend_fontsize=legend_fs,
        show=False,
        names=name_list,
        markers=['s', 'd', '^', 'o', 'o'],
        palette=colors,
        hue_order=('CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI PageRank'))

    plot_upset_plot(res_list=a_res_list, names=tuple(name_list), axs=axd['F'], plot_folder='./results/04_plots',
                    fn_prefix='alpha_')

    plot_digest_results(digest_res_path_list=apl, name_list=name_list, color_list=colors, size=dot_size2,
                        ax_label_fontsize=ax_fs, plot_xlabel=plt_xlabel, x_ticks_fontsize=dataset_fs,
                        legend_fontsize=legend_fs, axs=axd['G'], verbosity=1)

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=letter_fs, va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/supplementary_alpha.png', dpi=fig.dpi)


def plot_ery_res_appendix():
    # ### Script for plotting Supplementary Figure 2

    from validation.plotting import plot_gam_gene_trend_heatmap, plot_circled_tfs, plot_enrichr_results

    # ### Load AnnData with trand and with ptDE p-values
    adata_trend = sc.read_h5ad('./results/03_validation/anndata/trend_erythrocytes.h5ad')
    adata_sde = sc.read_h5ad('./results/03_validation/anndata/switchde_magic_nozeroinflated_erythrocytes.h5ad')

    # ### Load input and transition GRN
    grn = load_grn_json('./results/02_switchtfi/hematopoiesis/grn.json')
    transition_grn = pd.read_csv('./results/02_switchtfi/hematopoiesis/grn.csv', index_col=0)

    # ### Load ranked TFs
    ranked_tfs = pd.read_csv('./results/02_switchtfi/hematopoiesis/outdeg_ranked_tfs.csv', index_col=0)
    top_tfs = ranked_tfs['gene'].tolist()[0:10]

    # ### Load the GSEA results
    ery_gobp = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/ery_outdeg_GO_Biological_Process_2023_table.txt'),
        delimiter='\t'
    )
    ery_reactome = pd.read_csv(
        os.path.join(
            os.getcwd(), 'results/03_validation/gsea_results/ery_outdeg_Reactome_2022_table.txt'),
        delimiter='\t'
    )
    ery_mgi = pd.read_csv(
        os.path.join(
            os.getcwd(),
            'results/03_validation/gsea_results/ery_outdeg_MGI_Mammalian_Phenotype_Level_4_2021_table.txt'),
        delimiter='\t'
    )

    # ### Combine the switchde q-values using Fishers method for each edge (TF, target) of the base GRN
    # Add minimal eps to 0 q-values for numeric stability, then combine q-values => joint q-value per edge
    q = adata_sde.var['switchde_qval'].to_numpy()
    q[q == 0] = np.finfo(np.float64).eps
    adata_sde.var['switchde_qval'] = q

    grn = combine_p_vals_fisher(grn=grn, anndata=adata_sde)

    # ### Compute correlation (just for printing, not for plotting)
    corr = pearsonr(grn['weight'].to_numpy(), grn['switchde_qvals_combined_fisher'].to_numpy())
    print('###### Correlations(weights, combined q-vals): ')
    print(f'### q-vals: {corr}')

    # Again for plotting add eps
    qcomb = grn['switchde_qvals_combined_fisher'].to_numpy()
    qcomb[qcomb == 0] = np.finfo(np.float64).eps
    grn['switchde_qvals_combined_fisher'] = qcomb

    corr = pearsonr(grn['weight'].to_numpy(), -np.log(grn['switchde_qvals_combined_fisher'].to_numpy()))
    print(f'### -log(q-vals): {corr}')

    # Plot results
    fig = plt.figure(figsize=(12, 10), constrained_layout=True, dpi=100)
    axd = fig.subplot_mosaic(
        """
        A
        B
        C
        """
    )

    # General
    ax_fs = 20
    letter_fs = 14
    # Top 10 driver TFs plot
    gene_name_fs = 11
    # Gene trends plot
    gene_trend_anno_fs = 12
    # Enrichr results plots
    term_fs = 20
    legend_fs = 16
    # Regulon
    nodes_size = 1000
    gene_fs = 12

    plot_gam_gene_trend_heatmap(
        adata=adata_trend,
        gene_names=top_tfs,
        gene_names_fontsize=gene_trend_anno_fs,
        ax_label_fontsize=ax_fs,
        use_trend=True,
        annotate_gene_names=True,
        show=False,
        axs=axd['A'],
        colorbar_pad=0.05)
    plot_circled_tfs(res_df=ranked_tfs, topk=10, fontsize=gene_name_fs, ax_label_fontsize=ax_fs, axs=axd['B'])

    plot_enrichr_results(
        res_dfs=[ery_gobp, ery_reactome, ery_mgi],
        x='Adjusted P-value',
        top_k=[6, 6, 6],
        reference_db_names=['GO_Biological_Process_2023', 'Reactome_2022', 'MGI_Mammalian_Phenotype_Level_4_2021'],
        term_fontsize=term_fs,
        legend_fontsize=legend_fs,
        axs=axd['C']
    )

    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='x-large', va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/supplementary_ery.png', dpi=fig.dpi)


def plot_grns():
    # ### Script for plotting Supplementary Figure 3
    from switchtfi.plotting import plot_grn

    # Load GRNs
    # abasegrn = load_grn_json('./results/02_switchtfi/endocrine/alpha/grn.json')
    # agrn = pd.read_csv('./results/02_switchtfi/endocrine/alpha/grn.csv', index_col=0)
    # bbasegrn = load_grn_json('./results/02_switchtfi/endocrine/beta/grn.json')
    # bgrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)
    erybasegrn = load_grn_json('./results/02_switchtfi/hematopoiesis/grn.json')
    erygrn = pd.read_csv('./results/02_switchtfi/hematopoiesis/grn.csv', index_col=0)

    fig = plt.figure(figsize=(6, 12), constrained_layout=True, dpi=600)
    axd = fig.subplot_mosaic(
        """
        A
        B
        """
    )

    letter_fs = 26

    plot_grn(grn=erybasegrn, plot_folder='./results/04_plots', fn_prefix='base_', axs=axd['A'])
    plot_grn(grn=erygrn, plot_folder='./results/04_plots', fn_prefix='', axs=axd['B'])

    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=letter_fs, va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/grns.png')


if __name__ == '__main__':

    # plot_step_fct_and_trends()

    # plot_quantitative_analyses()

    plot_qualitative_analysis()

    plot_hypotheses_generation_ybx1()

    # plot_method_comparison()

    # plot_alpha_res_appendix()

    # plot_ery_res_appendix()

    # plot_grns()

    print('done')













