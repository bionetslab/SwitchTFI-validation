
import networkx as nx
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import os
from scipy.stats import combine_pvalues, pearsonr

from switchtfi.utils import load_grn_json


def plot_step_fct_and_trends():
    # ### Script for plotting Figure 2

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

    fig = plt.figure(figsize=(12, 12), constrained_layout=True, dpi=100)
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

    e0 = tuple(grn[['TF', 'target']].iloc[1])
    e1 = tuple(grn[['TF', 'target']].iloc[grn.shape[0] - 1])

    print(e0, e1)

    layer_key = 'magic_imputed'
    ax_label_fs = 12

    from validation.plotting import plot_step_function, plot_gam_gene_trend, plot_gam_gene_trend_heatmap
    plot_step_function(adata=bdata,
                       grn=grn,
                       which=e0,
                       layer_key=layer_key,
                       plot_threshold=True,
                       ax_label_fontsize=ax_label_fs,
                       legend_loc='none',
                       show=False,
                       axs=axd['A'])

    plot_step_function(adata=bdata,
                       grn=grn,
                       which=e1,
                       layer_key=layer_key,
                       plot_threshold=True,
                       ax_label_fontsize=ax_label_fs,
                       legend_loc='custom right',
                       show=False,
                       axs=axd['B'])

    plot_gam_gene_trend(adata=bdata,
                        gene_names=[e0[0]],
                        plot_cells=True,
                        ax_label_fontsize=ax_label_fs,
                        show=False,
                        axs=axd['C'],
                        layer_key='magic_imputed')

    plot_gam_gene_trend(adata=bdata,
                        gene_names=[e0[1]],
                        plot_cells=True,
                        ax_label_fontsize=ax_label_fs,
                        show=False,
                        axs=axd['D'],
                        layer_key='magic_imputed')

    plot_gam_gene_trend(adata=bdata,
                        gene_names=[e1[0]],
                        plot_cells=True,
                        ax_label_fontsize=ax_label_fs,
                        show=False,
                        axs=axd['E'],
                        layer_key='magic_imputed')

    plot_gam_gene_trend(adata=bdata,
                        gene_names=[e1[1]],
                        plot_cells=True,
                        ax_label_fontsize=ax_label_fs,
                        show=False,
                        axs=axd['F'],
                        layer_key='magic_imputed')

    tf_names = res_tfs['gene'].tolist()[0:10]

    plot_gam_gene_trend_heatmap(adata=bdata,
                                gene_names=tf_names,
                                use_trend=True,
                                gene_names_fontsize=11,
                                ax_label_fontsize=ax_label_fs,
                                show=False,
                                axs=axd['G'],
                                colorbar_pad=-0.125)

    from switchtfi.utils import get_regulons
    regulons = get_regulons(grn=pgrn, gene_names=['Pdx1', 'Pax4'])
    targets0 = regulons['Pdx1']['targets']
    targets1 = regulons['Pax4']['targets']

    plot_gam_gene_trend_heatmap(adata=bdata,
                                gene_names=targets0,
                                use_trend=True,
                                annotate_gene_names=False,
                                ax_label_fontsize=ax_label_fs,
                                show=False,
                                axs=axd['H'],
                                plot_colorbar=False,
                                title='Targets of TF Pdx1')

    plot_gam_gene_trend_heatmap(adata=bdata,
                                gene_names=targets1,
                                use_trend=True,
                                annotate_gene_names=False,
                                ax_label_fontsize=ax_label_fs,
                                show=False,
                                axs=axd['I'],
                                colorbar_pad=-0.3,
                                title='Targets of TF Pax4')

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        # ax = fig.add_subplot(axd[label])
        # ax.annotate(label, xy=(0.1, 1.1), xycoords='axes fraction', ha='center', fontsize=16)
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=14, va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/stepfct_trends.png', dpi=fig.dpi)


def plot_quantitative_analyses():
    # ### Script for plotting Figure 3

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
    def combine_p_vals_fisher(grn: pd.DataFrame,
                              anndata: sc.AnnData) -> pd.DataFrame:

        grn['switchde_qvals_combined_fisher'] = np.ones(grn.shape[0])

        for i in range(grn.shape[0]):
            tf = grn['TF'].loc[i]
            target = grn['target'].loc[i]
            tf_qval = anndata.var['switchde_qval'][tf]
            target_qval = anndata.var['switchde_qval'][target]

            grn.at[i, 'switchde_qvals_combined_fisher'] = combine_pvalues([tf_qval, target_qval], method='fisher')[1]

        return grn

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

    from validation.plotting import plot_defrac_lineplot, plot_sdeqvals_vs_weight, plot_cc_score_hist

    fig = plt.figure(figsize=(12, 8), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        ABC
        DEF
        """
    )

    plot_sdeqvals_vs_weight(grn=bbase_grn, comb_qval_alpha=0.01, weight_threshold=bminw,
                            title=r'$\beta$-cell transition data', legend_pos='upper center', size=6, axs=axd['A'])
    plot_sdeqvals_vs_weight(grn=erybase_grn, comb_qval_alpha=0.01, weight_threshold=eryminw,
                            title='Erythrocyte differentiation data', legend_pos='upper center', size=6, axs=axd['D'])

    plot_defrac_lineplot(adata=bdata, base_grn=bbase_grn, pruned_grn=bgrn, switchde_alpha=0.01,
                         title=r'$\beta$-cell transition data', size=12, legend_pos='center', axs=axd['B'], verbosity=1)
    plot_defrac_lineplot(adata=erydata, base_grn=erybase_grn, pruned_grn=erygrn, switchde_alpha=0.01,
                         title='Erythrocyte differentiation data', size=12, axs=axd['E'], verbosity=1)

    # ### Plot histograms of percentage of Lcc nodes in randomly sampled GRNs of same size as transition GRN
    from validation.val_utils import compare_grn_vs_rand_background
    from switchtfi.tf_ranking import grn_to_nx
    n = 10000
    bccs_list, bn_vert_list = compare_grn_vs_rand_background(base_grn=bbase_grn, transition_grn=bgrn, n=n)
    eryccs_list, eryn_vert_list = compare_grn_vs_rand_background(base_grn=erybase_grn, transition_grn=erygrn, n=n)

    btgrn = grn_to_nx(grn=bgrn).to_undirected()
    erytgrn = grn_to_nx(grn=erygrn).to_undirected()
    bccs = list(nx.connected_components(btgrn))
    eryccs = list(nx.connected_components(erytgrn))

    plot_cc_score_hist(ccs=bccs, n_vert=btgrn.number_of_nodes(), ccs_list=bccs_list, n_vert_list=bn_vert_list,
                       titel=r'$\beta$-cell transition data', axs=axd['C'])
    plot_cc_score_hist(ccs=eryccs, n_vert=erytgrn.number_of_nodes(), ccs_list=eryccs_list, n_vert_list=eryn_vert_list,
                       titel='Erythrocyte differentiation data', axs=axd['F'])

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=14.0, va='bottom', fontfamily='sans-serif', fontweight='bold')

    # plt.show()
    plt.savefig('./results/04_plots/quantitative_beta.png', dpi=fig.dpi)


def plot_qualitative_analysis():
    # ### Script for plotting Figure 4
    # Load ranked TFs
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

    # Load ENRICHR GSEA results
    benrichr = pd.read_csv('./results/03_validation/gsea_results/beta_Enrichr-KG.csv')

    # Load AnnData
    adata = sc.read_h5ad('./results/03_validation/anndata/trend_pre-endocrine_alpha.h5ad')

    # ### Plot results for Beta dataset ###
    fig = plt.figure(figsize=(12, 7), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AAB
        CCC
        CCC
        """
    )

    title_fontsize = 'x-large'
    from validation.plotting import plot_circled_tfs, plot_enrichr_results, plot_gam_gene_trend_heatmap
    plot_circled_tfs(res_df=atfs, topk=10, fontsize=11, res_df2=btfs, title=None, title_fontsize=None,
                     plottwoinone=True, y_ticks=(r'$\alpha$:', r'$\beta$:'), axs=axd['A'])

    plot_gam_gene_trend_heatmap(adata=adata, gene_names=atfs['gene'][0:10].tolist(),
                                use_trend=True,
                                annotate_gene_names=True,
                                gene_names_fontsize=12,
                                ax_label_fontsize=12,
                                show=False,
                                axs=axd['B'],
                                plot_colorbar=False,
                                title=r'$\alpha$-cell transition data',
                                title_fontsize=title_fontsize)

    # plot_enrichr_results(res_df=benrichr, x='combined score', color='q-value', title='Beta',
    #                      title_fontsize=title_fontsize, term_fontsize=12, axs=axd['A'])
    plot_enrichr_results(res_df=benrichr, x='combined score', title=r'$\beta$-cell transition data',
                         title_fontsize=title_fontsize, term_fontsize=13, axs=axd['C'])

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=14.0, va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/qualitative_beta.png', dpi=fig.dpi)


def plot_regulon():
    # ### Script for plotting Figure 5
    from switchtfi.plotting import plot_regulon
    # Load transition GRN of Pre-endocrine beta-cell dataset
    bgrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)

    plot_regulon(grn=bgrn, tf='Ybx1', sort_by='score', top_k=20, title=None, font_size=12, node_size=1200,
                 show=False, dpi=300)

    plt.show()
    # plt.savefig('./results/04_plots/regulon_ybx1.png', dpi=300)


def plot_method_comparison():
    # ### Script for plotting Figure 6
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
    name_list = ['CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI']
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
    ery_name_list = ['CellRank', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI']
    ery_colors = ['#1f77b4', '#2ca02c', '#f52891cc', '#d62728']

    # ### Plot results for Beta dataset ###
    fig = plt.figure(figsize=(14, 8), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        [
            ['A', 'C', 'E'],  # ['A.β', 'B.β', 'C.β'],
            ['B', 'D', 'F'],  # ['A.ery', 'B.ery', 'C.ery'],
        ],
    )

    # Plot Upsetplot
    from validation.plotting import plot_upset_plot
    plot_upset_plot(res_list=b_res_list, names=tuple(name_list), title=r'$\beta$-cell transition data', axs=axd['A'],
                    plot_folder='./results/04_plots', fn_prefix='beta')
    plot_upset_plot(res_list=ery_res_list, names=tuple(ery_name_list), title='Erythrocyte differentiation data',
                    axs=axd['B'], plot_folder='./results/04_plots', fn_prefix='ery')

    from validation.plotting import plot_defrac_in_top_k_lineplot, plot_digest_results

    plot_defrac_in_top_k_lineplot(res_df_list=b_res_list,
                                  adata=bdata,
                                  switchde_alpha=0.01,
                                  max_top_k=20,
                                  interval=5,
                                  axs=axd['C'],
                                  title=r'$\beta$-cell transition data',
                                  show=False,
                                  names=name_list,
                                  markers=['s', 'd', '^', 'o', 'o'],
                                  palette=colors,
                                  hue_order=('CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI'),
                                  legend_loc='lower right')

    plot_defrac_in_top_k_lineplot(res_df_list=ery_res_list,
                                  adata=erydata,
                                  switchde_alpha=0.01,
                                  max_top_k=20,
                                  interval=5,
                                  axs=axd['D'],
                                  title='Erythrocyte differentiation data',
                                  show=False,
                                  names=ery_name_list,
                                  markers=['s', '^', 'o', 'o'],
                                  palette=ery_colors,
                                  hue_order=('CellRank', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI'),
                                  legend_loc='center right')

    plot_digest_results(digest_res_path_list=bpl, name_list=name_list, color_list=colors,
                        title=r'$\beta$-cell transition data', size=9, axs=axd['E'], verbosity=1)
    plot_digest_results(digest_res_path_list=erypl, name_list=ery_name_list, color_list=ery_colors,
                        title='Erythrocyte differentiation data', size=9, axs=axd['F'], verbosity=1)

    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='x-large', va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/method_comparison.png', dpi=fig.dpi)
    # plt.savefig('./results/04_plots/method_comparison_imputed.png', dpi=fig.dpi)


def plot_alpha_res_appendix():
    # ### Script for plotting Supplementary Figure 1

    fig = plt.figure(figsize=(12, 12), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        ABC
        DDD
        EFG
        """
    )

    # ### Plot quantitative analysis
    # Load AnnData
    adata = sc.read_h5ad('./results/03_validation/anndata/switchde_magic_nozeroinflated_pre-endocrine_alpha.h5ad')

    # Load result dataframes
    agrn = pd.read_csv('./results/02_switchtfi/endocrine/alpha/grn.csv', index_col=0)
    abase_grn = load_grn_json('./results/02_switchtfi/endocrine/alpha/grn.json')

    # Combine ptDE q-values using Fishers method for each edge (TF, target) => edgewise ptDE q-values
    def combine_p_vals_fisher(grn: pd.DataFrame,
                              anndata: sc.AnnData) -> pd.DataFrame:

        grn['switchde_qvals_combined_fisher'] = np.ones(grn.shape[0])

        for i in range(grn.shape[0]):
            tf = grn['TF'].loc[i]
            target = grn['target'].loc[i]
            tf_qval = anndata.var['switchde_qval'][tf]
            target_qval = anndata.var['switchde_qval'][target]

            grn.at[i, 'switchde_qvals_combined_fisher'] = combine_pvalues([tf_qval, target_qval], method='fisher')[1]

        return grn

    # Add minimal eps to 0 q-values for numeric stability, then combine q-values => joint q-value per edge
    aq = adata.var['switchde_qval'].to_numpy()
    aq[aq == 0] = np.finfo(np.float64).eps
    adata.var['switchde_qval'] = aq

    abase_grn = combine_p_vals_fisher(grn=abase_grn, anndata=adata)
    aminw = agrn['weight'].to_numpy().min()

    # Compute correlation
    acorr = pearsonr(abase_grn['weight'].to_numpy(), abase_grn['switchde_qvals_combined_fisher'].to_numpy())

    print('###### Correlations(weights, combined q-vals): ')
    print(f'### Alpha: {acorr}')

    # Again for plotting add eps
    aqcomb = abase_grn['switchde_qvals_combined_fisher'].to_numpy()
    aqcomb[aqcomb == 0] = np.finfo(np.float64).eps
    abase_grn['switchde_qvals_combined_fisher'] = aqcomb
    from validation.plotting import plot_defrac_lineplot, plot_sdeqvals_vs_weight, plot_cc_score_hist

    plot_sdeqvals_vs_weight(grn=abase_grn, comb_qval_alpha=0.01, weight_threshold=aminw, title=None,
                            legend_pos='upper center', size=6, axs=axd['A'])
    plot_defrac_lineplot(adata=adata, base_grn=abase_grn, pruned_grn=agrn, switchde_alpha=0.01,
                         title=None, size=12, axs=axd['B'], verbosity=1)

    # ### Plot histograms of percentage of Lcc nodes in randomly sampled GRNs of same size as transition GRN
    from validation.val_utils import compare_grn_vs_rand_background
    from switchtfi.tf_ranking import grn_to_nx
    n = 10000
    accs_list, an_vert_list = compare_grn_vs_rand_background(base_grn=abase_grn, transition_grn=agrn, n=n)
    atgrn = grn_to_nx(grn=agrn).to_undirected()
    bccs = list(nx.connected_components(atgrn))

    plot_cc_score_hist(ccs=bccs, n_vert=atgrn.number_of_nodes(), ccs_list=accs_list, n_vert_list=an_vert_list,
                       titel=None, axs=axd['C'])

    # ### Plot qualitative analysis
    # Load ENRICHR GSEA results
    aenrichr = pd.read_csv('./results/03_validation/gsea_results/alpha_Enrichr-KG.csv')

    from validation.plotting import plot_enrichr_results

    # plot_enrichr_results(res_df=aenrichr, x='combined score', color='q-value', axs=axd['C'])
    plot_enrichr_results(res_df=aenrichr, x='combined score', term_fontsize=12, axs=axd['D'])

    # ### Method comparison
    # Load AnnData with switchtfi p-values
    adata = sc.read_h5ad('./results/03_validation/anndata/switchde_log1p_norm_zeroinflated_pre-endocrine_alpha.h5ad')

    # sc.pl.umap(adata, color='palantir_pseudotime', show=False)
    # plt.savefig('./results/04_plots/aumap.png')

    # ### Load result dataframes
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

    # Define names of method and color to be used in plots
    name_list = ['CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#f52891cc', '#d62728']

    from validation.plotting import plot_defrac_in_top_k_lineplot, plot_digest_results, plot_upset_plot

    plot_defrac_in_top_k_lineplot(res_df_list=a_res_list,
                                  adata=adata,
                                  switchde_alpha=0.01,
                                  max_top_k=20,
                                  interval=5,
                                  axs=axd['E'],
                                  title=None,
                                  show=False,
                                  names=name_list,
                                  markers=['s', 'd', '^', 'o', 'o'],
                                  palette=colors,
                                  hue_order=('CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI'))

    plot_upset_plot(res_list=a_res_list, names=tuple(name_list), axs=axd['F'], plot_folder='./results/04_plots',
                    fn_prefix='alpha_')

    plot_digest_results(digest_res_path_list=apl, name_list=name_list, color_list=colors, size=9, axs=axd['G'],
                        verbosity=1)

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='x-large', va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/supplementary_alpha.png', dpi=fig.dpi)


def plot_ery_res_appendix():
    # ### Script for plotting Supplementary Figure 2

    # Load AnnData
    adata_trend = sc.read_h5ad('./results/03_validation/anndata/trend_erythrocytes.h5ad')
    adata_sde = sc.read_h5ad('./results/03_validation/anndata/switchde_magic_nozeroinflated_erythrocytes.h5ad')

    # Load GRNs
    grn = load_grn_json('./results/02_switchtfi/hematopoiesis/grn.json')
    transition_grn = pd.read_csv('./results/02_switchtfi/hematopoiesis/grn.csv', index_col=0)

    # Load ranked TFs
    ranked_tfs = pd.read_csv('./results/02_switchtfi/hematopoiesis/outdeg_ranked_tfs.csv', index_col=0)
    top_tfs = ranked_tfs['gene'].tolist()[0:10]

    # Load the gesea results
    enrichr_res = pd.read_csv('./results/03_validation/gsea_results/ery_Enrichr-KG.csv')

    # Combine the switchde q-values using Fishers method for each edge (TF, target) of the base GRN
    def combine_p_vals_fisher(grn: pd.DataFrame,
                              anndata: sc.AnnData) -> pd.DataFrame:

        grn['switchde_qvals_combined_fisher'] = np.ones(grn.shape[0])

        for i in range(grn.shape[0]):
            tf = grn['TF'].loc[i]
            target = grn['target'].loc[i]
            tf_qval = anndata.var['switchde_qval'][tf]
            target_qval = anndata.var['switchde_qval'][target]

            grn.at[i, 'switchde_qvals_combined_fisher'] = combine_pvalues([tf_qval, target_qval], method='fisher')[1]

        return grn

    # Add minimal eps to 0 q-values for numeric stability, then combine q-values => joint q-value per edge
    q = adata_sde.var['switchde_qval'].to_numpy()
    q[q == 0] = np.finfo(np.float64).eps
    adata_sde.var['switchde_qval'] = q

    grn = combine_p_vals_fisher(grn=grn, anndata=adata_sde)
    minw = transition_grn['weight'].to_numpy().min()

    # Compute correlation
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
    fig = plt.figure(figsize=(12, 10), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        A
        B
        C
        """
    )

    # Plot UMAP
    # sc.pl.umap(adata_trend, color='prog_off', palette={'prog': '#ffac00', 'off': '#95f527'}, ax=axd['A'], title='',
    #            show=False)
    # sc.pl.umap(adata_trend, color='palantir_pseudotime', ax=axd['B'], title='', color_map='magma', show=False)

    from validation.plotting import plot_gam_gene_trend_heatmap, plot_circled_tfs, plot_enrichr_results

    plot_gam_gene_trend_heatmap(adata=adata_trend,
                                gene_names=top_tfs,
                                gene_names_fontsize=12,
                                use_trend=True,
                                annotate_gene_names=True,
                                show=False,
                                axs=axd['A'],
                                colorbar_pad=0.05)
    plot_circled_tfs(res_df=ranked_tfs, topk=10, fontsize=14, axs=axd['B'])
    # plot_enrichr_results(res_df=enrichr_res, x='combined score', color='q-value', axs=axd['G'])
    plot_enrichr_results(res_df=enrichr_res, x='combined score', log_trafo=True, term_fontsize=12, axs=axd['C'])

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

    plot_grn(grn=erybasegrn, plot_folder='./results/04_plots', fn_prefix='base_', axs=axd['A'])
    plot_grn(grn=erygrn, plot_folder='./results/04_plots', fn_prefix='', axs=axd['B'])

    import matplotlib.transforms as mtransforms
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=26, va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.show()
    # plt.savefig('./results/04_plots/grns.png')

    fig = plt.figure(figsize=(6, 5), constrained_layout=True, dpi=600)
    axd = fig.subplot_mosaic(
        """
        A
        """
    )
    plot_grn(grn=erygrn, plot_folder='./results/04_plots', fn_prefix='zzz_', axs=axd['A'])
    # plt.savefig('./results/04_plots/grnery.png')
    plt.show()


if __name__ == '__main__':

    # plot_step_fct_and_trends()

    # plot_quantitative_analyses()

    # plot_qualitative_analysis()

    # plot_regulon()

    # plot_method_comparison()

    # plot_alpha_res_appendix()

    # plot_ery_res_appendix()

    # plot_grns()

    print('done')













