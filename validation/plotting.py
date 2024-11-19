
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import scanpy as sc
import matplotlib.image as mpimg
import math
import json
import os
from typing import *

from scipy.stats import pearsonr
from itertools import combinations
from upsetplot import UpSet


from switchtfi.utils import csr_to_numpy, anndata_to_numpy


def plot_step_function(adata: sc.AnnData,
                       grn: pd.DataFrame,
                       which: Union[Tuple[str, str], list[int]],
                       layer_key: Union[None, str] = None,
                       palette: Union[dict, str, None] = None,
                       plot_threshold: bool = False,
                       ax_label_fontsize: Union[float, None] = None,
                       clustering_key: str = 'clusters',
                       weight_key: str = 'weight',
                       tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                       cell_bool_key: str = 'cell_bool',
                       threshold_key: str = 'threshold',
                       pred_l_key: str = 'pred_l',
                       pred_r_key: str = 'pred_r',
                       show: bool = True,
                       axs: Union[plt.Axes, None] = None,
                       legend_loc: Union[str, None] = 'right margin'):

    if isinstance(which, tuple):  # Tuple of strings ('tf', 'target') as input
        tf = which[0]
        target = which[1]
        i = find_i(grn, tf, target, tf_target_keys)
        ax = plot_step_function_aux(adata=adata,
                                    grn=grn,
                                    i=i,
                                    layer_key=layer_key,
                                    palette=palette,
                                    plot_threshold=plot_threshold,
                                    ax_label_fontsize=ax_label_fontsize,
                                    clustering_key=clustering_key,
                                    weight_key=weight_key,
                                    tf_target_keys=tf_target_keys,
                                    cell_bool_key=cell_bool_key,
                                    threshold_key=threshold_key,
                                    pred_l_key=pred_l_key,
                                    pred_r_key=pred_r_key,
                                    show=show,
                                    axs=axs,
                                    legend_loc=legend_loc)
        return ax

    else:  # List of indices as input
        ax_list = []
        for i in which:
            ax = plot_step_function_aux(adata=adata,
                                        grn=grn,
                                        i=i,
                                        layer_key=layer_key,
                                        palette=palette,
                                        plot_threshold=plot_threshold,
                                        ax_label_fontsize=ax_label_fontsize,
                                        clustering_key=clustering_key,
                                        weight_key=weight_key,
                                        tf_target_keys=tf_target_keys,
                                        cell_bool_key=cell_bool_key,
                                        threshold_key=threshold_key,
                                        pred_l_key=pred_l_key,
                                        pred_r_key=pred_r_key,
                                        show=show,
                                        axs=axs,
                                        legend_loc=legend_loc)
            ax_list.append(ax)
        if ax_list:
            return ax_list


def plot_step_function_aux(adata: sc.AnnData,
                           grn: pd.DataFrame,
                           i: Union[int],
                           layer_key: Union[None, str] = None,
                           palette: Union[dict, str, None] = None,
                           plot_threshold: bool = False,
                           ax_label_fontsize: Union[float, None] = None,
                           clustering_key: str = 'clusters',
                           weight_key: str = 'weight',
                           tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                           cell_bool_key: str = 'cell_bool',
                           threshold_key: str = 'threshold',
                           pred_l_key: str = 'pred_l',
                           pred_r_key: str = 'pred_r',
                           show: bool = True,
                           axs: Union[plt.Axes, None] = None,
                           legend_loc: Union[str, None] = 'right margin'):
    # Get names of TF and target
    tf_key = tf_target_keys[0]
    target_key = tf_target_keys[1]
    tf = grn[tf_key].iloc[i]
    target = grn[target_key].iloc[i]
    # Subset adata to cells used for fitting weight
    cell_bool = grn[cell_bool_key].iloc[i]
    adata = adata[cell_bool, :].copy()

    custom_legend = False
    if legend_loc == 'custom right':
        legend_loc = 'none'
        custom_legend = True

    try:
        ax = sc.pl.scatter(adata,
                           x=tf,
                           y=target,
                           color=clustering_key,
                           layers=layer_key,
                           legend_loc=legend_loc,
                           palette=palette,
                           title=f"{weight_key.capitalize()}: {grn[weight_key].iloc[i]}",
                           show=False,
                           use_raw=False,
                           ax=axs)
    except KeyError:
        ax = sc.pl.scatter(adata,
                           x=tf,
                           y=target,
                           color=clustering_key,
                           layers=layer_key,
                           legend_loc=legend_loc,
                           palette=palette,
                           show=False,
                           use_raw=False,
                           ax=axs)

    if layer_key is None:
        x = csr_to_numpy(adata[:, tf].X).flatten()
    else:
        x = csr_to_numpy(adata[:, tf].layers[layer_key]).flatten()
    min_x = x.min()
    max_x = x.max()

    threshold = grn[threshold_key].iloc[i]
    pred_l = grn[pred_l_key].iloc[i]
    pred_r = grn[pred_r_key].iloc[i]

    ax.plot([min_x, threshold], [pred_l, pred_l], color='red', zorder=2)
    ax.scatter([threshold], [pred_l], color='red', marker='o', zorder=3)
    ax.plot([threshold, max_x], [pred_r, pred_r], color='red', zorder=2)
    ax.scatter([threshold], [pred_r], color='red', marker='o', facecolor='white', zorder=3)

    if plot_threshold:
        ax.axvline(x=threshold, color='red', linestyle='--', label='Dec. bound.', zorder=1)

    if custom_legend:
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='center left', borderaxespad=0.)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # adjust fontsize
    if ax_label_fontsize is not None:
        ax.xaxis.label.set_size(ax_label_fontsize)
        ax.yaxis.label.set_size(ax_label_fontsize)

    if show:
        plt.show()
    else:
        return ax


def plot_n_cells_vs_weight(grn: pd.DataFrame,
                           plt_title: Union[str, None] = None,
                           weight_key: str = 'weight',
                           cell_bool_key: str = 'cell_bool',
                           verbosity: int = 0) -> Tuple[np.ndarray, np.ndarray]:

    weights = grn[weight_key].to_numpy()
    n_cells = np.array([a.sum() for a in grn[cell_bool_key].to_numpy()])

    res = pearsonr(n_cells, weights)
    if verbosity >= 0:
        print(f"# The Pearson_CC(weights, number of cells used for fitting) = {res.statistic}, \n"
              f"# The corresponding p-value (two-sided test, H0: correlation != 0) is {res.pvalue}")

    median_weight = np.median(weights)
    median_n_cells = np.median(n_cells)

    plt.scatter(x=n_cells, y=weights, marker='.')
    plt.axvline(x=median_n_cells, color='green', label=f'median n_cells: {round(median_n_cells, 2)}')
    plt.axhline(y=median_weight, color='red', label=f'median weights: {round(median_weight, 2)}')
    plt.xlabel('n_cells')
    plt.ylabel('weight')
    plt.legend(loc='best')
    if plt_title is not None:
        plt.title(f'{plt_title}, pc: {round(res.statistic, 4)}, pval: {res.pvalue}')
    plt.show()

    return weights, n_cells


def plot_pvals_vs_weight(grn: pd.DataFrame,
                         weight_key: str = 'weight',
                         pval_key: str = 'pvals_wy',
                         sig_thresh: float = 0.05,
                         show: bool = True):

    weights = grn[weight_key].to_numpy()
    pvals = grn[pval_key].to_numpy()
    plt.scatter(weights, pvals, color='green', marker='o', alpha=0.8)
    plt.xlabel('weight')
    plt.ylabel('p-value')
    plt.axhline(y=sig_thresh, color='red', label=f'alpha: {sig_thresh}')

    if show:
        plt.show()


def plot_gam_gene_trend(adata: sc.AnnData,
                        gene_names: List[str],
                        gene_trend_varm_key: str = 'gam_gene_trends',
                        pseudotime_obs_key: str = 'palantir_pseudotime',
                        plot_cells: bool = False,
                        ci_uns_key: Union[str, None] = 'gam_confidence_intervals',
                        ax_label_fontsize: Union[float, None] = None,
                        show: bool = True,
                        axs: Union[plt.Axes, None] = None,
                        **kwargs):

    pt_vec = adata.obs[pseudotime_obs_key].to_numpy()
    pt_grid = np.linspace(pt_vec.min(), pt_vec.max(), adata.varm[gene_trend_varm_key].shape[1])

    for gene in gene_names:
        if axs is None:
            if plot_cells:
                layer_key = kwargs.get('layer_key')
                expression_vec = anndata_to_numpy(adata=adata[:, gene], layer_key=layer_key)
                plt.scatter(pt_vec,
                            expression_vec,
                            c='gray',  # expression_vec,
                            cmap='gray',
                            alpha=0.6,
                            edgecolors='none',
                            linewidths=0,
                            s=6.0)

            plt.plot(pt_grid, adata[:, gene].varm[gene_trend_varm_key].flatten(), label=f'{gene}', color='red')
            plt.xlabel('Pseudotime')
            plt.ylabel('Gene expression')
            plt.title('Gene trend in pseudotime')

            if ci_uns_key is not None:
                ci_dict = adata.uns[ci_uns_key]
                cis = ci_dict[gene]  # 2 x trend_resolution
                plt.fill_between(x=pt_grid, y1=cis[0], y2=cis[1],
                                 alpha=0.2, color='red', linestyle='--', edgecolor='red')

        else:
            if plot_cells:
                layer_key = kwargs.get('layer_key')
                expression_vec = anndata_to_numpy(adata=adata[:, gene], layer_key=layer_key)
                axs.scatter(pt_vec,
                            expression_vec,
                            c='gray',  # expression_vec,
                            cmap='gray',
                            alpha=0.6,
                            edgecolors='none',
                            linewidths=0,
                            s=6.0)  # label='expression data')

            axs.plot(pt_grid, adata[:, gene].varm[gene_trend_varm_key].flatten(), label=f'{gene}', color='red')
            axs.set_xlabel('Pseudotime')
            axs.set_ylabel(f'{gene}')

            if ci_uns_key is not None:
                ci_dict = adata.uns[ci_uns_key]
                cis = ci_dict[gene]  # 2 x trend_resolution
                axs.fill_between(x=pt_grid, y1=cis[0], y2=cis[1],
                                 alpha=0.2, color='red', linestyle='--', edgecolor='red')

            # adjust fontsize
            if ax_label_fontsize is not None:
                axs.xaxis.label.set_size(ax_label_fontsize)
                axs.yaxis.label.set_size(ax_label_fontsize)

        if show:
            plt.legend()
            plt.tight_layout()
            plt.show()


def plot_gam_gene_trend_heatmap(adata: sc.AnnData,
                                gene_names: List[str],
                                use_trend: bool = False,
                                annotate_gene_names: bool = True,
                                gene_names_fontsize: Union[str, float, None] = None,
                                ax_label_fontsize: Union[float, None] = None,
                                gene_trend_varm_key: Union[str, None] = 'gam_gene_trends',
                                layer_key: Union[str, None] = 'magic_imputed',
                                pseudotime_obs_key: Union[str, None] = 'palantir_pseudotime',
                                show: bool = True,
                                axs: Union[plt.Axes, None] = None,
                                plot_colorbar: bool = True,
                                colorbar_pad: float = 0.05,
                                title: Union[str, None] = None,
                                title_fontsize: Union[str, float, None] = None):
    # Get pt-vector
    pt_vec = adata.obs[pseudotime_obs_key].to_numpy()

    if not use_trend:
        # Get expression data
        data = adata.to_df(layer=layer_key)
        data = data[gene_names]
        # Sort cells w.r.t. pseudotime
        sorting_idx = np.argsort(pt_vec)
        data = data.iloc[sorting_idx].to_numpy()

        dummy = data.copy()
        window_size = 6
        dummy = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size) / window_size, mode='valid'),
                                    axis=0, arr=dummy)

    else:
        data = adata[:, gene_names].varm[gene_trend_varm_key].T
        dummy = adata[:, gene_names].varm[gene_trend_varm_key].T

    # Scale gene expressions to [0, 1] for comparability
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    data = (data - mins) / (maxs - mins)

    # Sort genes such that the ones with the earliest peak come first
    argmaxs = np.argmax(dummy, axis=0)
    gene_sorting_idx = np.argsort(argmaxs)
    data = data[:, gene_sorting_idx]
    gene_names = np.array(gene_names)[gene_sorting_idx].tolist()

    if axs is None:
        heatmap = plt.imshow(data.T, cmap='viridis', aspect='auto', interpolation='none')

        # Set labels and title
        if ax_label_fontsize is not None:
            plt.xlabel('Pseudotime', fontsize=ax_label_fontsize)
        else:
            plt.xlabel('Pseudotime')
        if annotate_gene_names:
            # Annotate gene names on the y-axis
            if gene_names_fontsize is not None:
                plt.yticks(ticks=np.arange(len(gene_names)), labels=gene_names, fontsize=gene_names_fontsize)
            else:
                plt.yticks(ticks=np.arange(len(gene_names)), labels=gene_names)
        else:
            # Remove ticks and tick labels from the y-axis
            plt.yticks(ticks=[], labels=[])  # Remove ticks

        # Annotate pseudotime on the x-axis
        if use_trend:
            # Define x-tick values
            x_tick_values = np.round(np.linspace(pt_vec.min(), pt_vec.max(), num=5), decimals=2)
            x_tick_positions = np.linspace(0, data.shape[0] - 1, num=5).astype(int)
        else:
            x_tick_values = np.round(np.linspace(pt_vec.min(), pt_vec.max(), num=2), decimals=2)
            x_tick_positions = np.linspace(0, data.shape[0] - 1, num=2).astype(int)
        plt.xticks(x_tick_positions, x_tick_values)

        if plot_colorbar:
            # Add colorbar
            plt.colorbar(heatmap, label='Scaled expression')

        if title is not None:
            if title_fontsize is not None:
                plt.title(title, fontsize=title_fontsize)
            else:
                plt.title(title)

        # Show plot
        if show:
            plt.tight_layout()
            plt.show()
    else:
        heatmap = axs.imshow(data.T, cmap='viridis', aspect='auto', interpolation='none')
        axs.set_frame_on(False)
        if ax_label_fontsize is None:
            axs.set_xlabel('Pseudotime')
        else:
            axs.set_xlabel('Pseudotime', fontsize=ax_label_fontsize)
        if annotate_gene_names:
            if gene_names_fontsize is not None:
                axs.set_yticks(ticks=np.arange(len(gene_names)), labels=gene_names, fontsize=gene_names_fontsize)
            else:
                axs.set_yticks(ticks=np.arange(len(gene_names)), labels=gene_names)

        else:
            axs.set_yticks(ticks=[], labels=[])
        if use_trend:
            x_tick_values = np.round(np.linspace(pt_vec.min(), pt_vec.max(), num=5), decimals=2)
            x_tick_positions = np.linspace(0, data.shape[0] - 1, num=5).astype(int)
        else:
            x_tick_values = np.round(np.linspace(pt_vec.min(), pt_vec.max(), num=2), decimals=2)
            x_tick_positions = np.linspace(0, data.shape[0] - 1, num=2).astype(int)
        axs.set_xticks(x_tick_positions, x_tick_values)
        if plot_colorbar:
            plt.colorbar(heatmap, label='Scaled expression', ax=axs, location='right', pad=colorbar_pad)
        if title is not None:
            if title_fontsize is not None:
                axs.set_title(title, fontsize=title_fontsize)
            else:
                axs.set_title(title)


def plot_defrac_lineplot(adata: sc.AnnData,
                         base_grn: pd.DataFrame,
                         pruned_grn: pd.DataFrame,
                         switchde_alpha: float = 0.01,
                         title: Union[str, None] = None,
                         size: float = 10.,
                         plot_legend: bool = True,
                         legend_pos: str = 'upper center',
                         axs: Union[plt.Axes, None] = None,
                         show: bool = False,
                         tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                         switchde_qvals_var_key: str = 'switchde_qval',
                         verbosity: int = 0):

    base_genes = np.unique(base_grn[list(tf_target_keys)].to_numpy())
    base_tfs = np.unique(base_grn[tf_target_keys[0]].to_numpy())
    base_targets = np.unique(base_grn[tf_target_keys[1]].to_numpy())

    # Subset Anndata to genes that appear in the base GRN
    gn = adata.var_names.to_numpy()
    base_grn_gene_bool = np.isin(gn, base_genes)
    adata = adata[:, base_grn_gene_bool]
    gn = adata.var_names.to_numpy()

    pruned_genes = np.unique(pruned_grn[list(tf_target_keys)].to_numpy())
    pruned_tfs = np.unique(pruned_grn[tf_target_keys[0]].to_numpy())
    pruned_targets = np.unique(pruned_grn[tf_target_keys[1]].to_numpy())

    switchde_qvals = adata.var[switchde_qvals_var_key].to_numpy()
    sig = (switchde_qvals <= switchde_alpha)

    base_frac_degenes = sig.mean()
    pruned_frac_degenes = sig[np.isin(gn, pruned_genes)].mean()
    base_frac_detfs = sig[np.isin(gn, base_tfs)].mean()
    pruned_frac_detfs = sig[np.isin(gn, pruned_tfs)].mean()
    base_frac_detargets = sig[np.isin(gn, base_targets)].mean()
    pruned_frac_detargets = sig[np.isin(gn, pruned_targets)].mean()

    if verbosity >= 1:
        print(f'### genes: base: {base_frac_degenes}, pruned: {pruned_frac_degenes}, '
              f'delta: {pruned_frac_degenes - base_frac_degenes}')
        print(f'### tfs: base: {base_frac_detfs}, pruned: {pruned_frac_detfs}, '
              f'delta: {pruned_frac_detfs - base_frac_detfs}')
        print(f'### targets: base: {base_frac_detargets}, pruned: {pruned_frac_detargets}, '
              f'delta: {pruned_frac_detargets - base_frac_detargets}')

    # pos = [1, 2]
    # data = [-np.log(switchde_qvals[sig]), -np.log(switchde_qvals[sig * np.isin(vn, pruned_genes)])]
    # axd['A'].violinplot(data, pos, showmedians=True)

    data = pd.DataFrame()
    data['DE fraction'] = [base_frac_degenes, pruned_frac_degenes, base_frac_detfs, pruned_frac_detfs,
                           base_frac_detargets, pruned_frac_detargets]
    data['GRN'] = ['Input GRN (Scenic)', 'Transition GRN (SwitchTFI)'] * 3
    data['subset'] = ['All genes', 'All genes', 'TFs', 'TFs', 'Targets', 'Targets']

    if axs is None:
        ax = sns.stripplot(data=data, x='DE fraction', y='subset', hue='GRN', size=size, linewidth=1, orient='h',
                           jitter=False)

    else:
        ax = sns.stripplot(data=data, x='DE fraction', y='subset', hue='GRN', size=size, linewidth=1, orient='h',
                           jitter=False, ax=axs)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    # sns.despine(left=True, bottom=True)
    # Remove the lines (spines) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set(ylabel=None)

    sns.move_legend(ax, legend_pos)

    x_ticks = np.round(np.linspace(data['DE fraction'].to_numpy().min(), data['DE fraction'].to_numpy().max(), num=4),
                       decimals=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    if not plot_legend:
        ax.legend_.remove()

    if title is not None:
        ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()


def plot_sdeqvals_vs_weight(grn: pd.DataFrame,
                            comb_qval_alpha: float = 0.01,
                            weight_threshold: Union[float, None] = None,
                            title: Union[str, None] = None,
                            legend_pos: str = 'best',
                            size: Union[float, None] = None,
                            weight_key: str = 'weight',
                            qval_key: str = 'switchde_qvals_combined_fisher',
                            axs: Union[plt.Axes, None] = None,
                            show: bool = False):

    weights = grn[weight_key].to_numpy()
    qvals = grn[qval_key].to_numpy()
    qvals = - np.log10(qvals)
    comb_qval_alpha = - np.log10(comb_qval_alpha)

    if weight_threshold is not None:
        w_bool = (weights >= weight_threshold).astype(int)
    else:
        w_bool = np.ones(grn.shape[0])

    # Define red and orange colors
    colors = np.array(['#1f77b4', '#ff7f0e'])
    if axs is None:
        plt.scatter(weights, qvals, c=colors[w_bool], s=size, marker='o', alpha=0.8)
        plt.xlabel('Weight')
        plt.ylabel('-log10(combined q-value)')
        plt.axhline(y=comb_qval_alpha, color='red', label=f'q-val threshold: {comb_qval_alpha}')
        if weight_threshold is not None:
            plt.axvline(x=weight_threshold, color='gold', label=f'Weight threshold: {weight_threshold}')
        plt.legend(loc=legend_pos)

        x_ticks = np.round(
            np.linspace(weights.min(), weights.max(), num=4),
            decimals=2)
        plt.xticks(x_ticks, labels=x_ticks)

        if title is not None:
            plt.title(title)

    else:
        axs.scatter(weights, qvals, c=colors[w_bool], s=size, marker='o', alpha=0.8)
        axs.set_xlabel('Weight')
        axs.set_ylabel('-log10(combined q-value)')
        axs.axhline(y=comb_qval_alpha, color='red', label=f'q-value threshold: {np.round(comb_qval_alpha, decimals=3)}')

        if weight_threshold is not None:
            axs.axvline(x=weight_threshold, color='gold',
                        label=f'Weight threshold: {np.round(weight_threshold, decimals=3)}')
        axs.legend(loc=legend_pos)

        x_ticks = np.round(
            np.linspace(weights.min(), weights.max(), num=4),
            decimals=2)
        axs.set_xticks(x_ticks)
        axs.set_xticklabels(x_ticks)

        if title is not None:
            axs.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()


def plot_base_v_pruned_grn_comparison(adata: sc.AnnData,
                                      base_grn: pd.DataFrame,
                                      pruned_grn: pd.DataFrame,
                                      switchde_alpha: float = 0.01,
                                      title: Union[str, None] = None,
                                      plot_legend: bool = True,
                                      axs: Union[plt.Axes, None] = None,
                                      show: bool = False,
                                      tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                                      switchde_qvals_var_key: str = 'switchde_qval',
                                      verbosity: int = 0):

    base_genes = np.unique(base_grn[list(tf_target_keys)].to_numpy())
    base_tfs = np.unique(base_grn[tf_target_keys[0]].to_numpy())
    base_targets = np.unique(base_grn[tf_target_keys[1]].to_numpy())

    # Subset Anndata to genes that appear in the base GRN
    gn = adata.var_names.to_numpy()
    base_grn_gene_bool = np.isin(gn, base_genes)
    adata = adata[:, base_grn_gene_bool]
    gn = adata.var_names.to_numpy()

    pruned_genes = np.unique(pruned_grn[list(tf_target_keys)].to_numpy())
    pruned_tfs = np.unique(pruned_grn[tf_target_keys[0]].to_numpy())
    pruned_targets = np.unique(pruned_grn[tf_target_keys[1]].to_numpy())

    switchde_qvals = adata.var[switchde_qvals_var_key].to_numpy()
    sig = (switchde_qvals <= switchde_alpha)

    base_frac_degenes = sig.mean()
    pruned_frac_degenes = sig[np.isin(gn, pruned_genes)].mean()
    base_frac_detfs = sig[np.isin(gn, base_tfs)].mean()
    pruned_frac_detfs = sig[np.isin(gn, pruned_tfs)].mean()
    base_frac_detargets = sig[np.isin(gn, base_targets)].mean()
    pruned_frac_detargets = sig[np.isin(gn, pruned_targets)].mean()

    if verbosity >= 1:
        print(f'### genes: base: {base_frac_degenes}, pruned: {pruned_frac_degenes}')
        print(f'### tfs: base: {base_frac_detfs}, pruned: {pruned_frac_detfs}')
        print(f'### targets: base: {base_frac_detargets}, pruned: {pruned_frac_detargets}')

    # pos = [1, 2]
    # data = [-np.log(switchde_qvals[sig]), -np.log(switchde_qvals[sig * np.isin(vn, pruned_genes)])]
    # axd['A'].violinplot(data, pos, showmedians=True)

    data = pd.DataFrame()
    data['DE fraction'] = [base_frac_degenes, pruned_frac_degenes, base_frac_detfs, pruned_frac_detfs,
                           base_frac_detargets, pruned_frac_detargets]
    data['GRN'] = ['base', 'pruned'] * 3
    data['subset'] = ['genes', 'genes', 'TFs', 'TFs', 'targets', 'targets']

    if axs is None:
        ax = sns.stripplot(data=data, x='DE fraction', y='subset', hue='GRN', size=10, linewidth=1, orient='h',
                           jitter=False)

    else:
        ax = sns.stripplot(data=data, x='DE fraction', y='subset', hue='GRN', size=10, linewidth=1, orient='h',
                           jitter=False, ax=axs)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    sns.despine(left=True, bottom=True)

    ax.set(ylabel=None)

    sns.move_legend(ax, 'center left', bbox_to_anchor=(1, 0.5))

    x_ticks = np.round(np.linspace(data['DE fraction'].to_numpy().min(), data['DE fraction'].to_numpy().max(), num=4),
                       decimals=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    if not plot_legend:
        ax.legend_.remove()

    if title is not None:
        ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()


def plot_defrac_in_top_k_lineplot(res_df_list: List[pd.DataFrame],
                                  adata: sc.AnnData,
                                  switchde_alpha: float = 0.01,
                                  max_top_k: int = 20,
                                  interval: int = 1,
                                  axs: Union[plt.Axes, None] = None,
                                  title: Union[str, None] = None,
                                  show: bool = False,
                                  names: Union[List[str], None] = None,
                                  markers: Union[List[str], None] = ('s', 'd', '^', 'o'),
                                  palette: Union[List[str], None] = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'),
                                  hue_order: Union[Tuple[str, ...], None] =
                                  ('CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI'),
                                  legend_loc: str = 'best',
                                  gene_key: str = 'gene',
                                  sdeqval_var_key: str = 'switchde_qval'):

    # Create dataframe for plotting
    df = create_switchdefrac_df(res_df_list=res_df_list,
                                adata=adata,
                                switchde_alpha=switchde_alpha,
                                max_top_k=max_top_k,
                                names=names,
                                gene_key=gene_key,
                                sdeqval_var_key=sdeqval_var_key)

    df = df[df['Top k'] % interval == 0].copy()

    # Transform from wide into long format for seaborn
    df_melted = pd.melt(df, id_vars=['Top k'], var_name='Method', value_name='ptDE fraction')

    if markers is not None:
        markers = list(markers)
    if palette is not None:
        palette = list(palette)

    ax = sns.lineplot(data=df_melted, x='Top k', y='ptDE fraction', hue='Method', style='Method', markers=markers,
                      palette=palette, dashes=False, hue_order=list(hue_order), ax=axs, zorder=2)

    # Add horizontal line at fraction of de genes in dataset
    de_frac = (adata.var[sdeqval_var_key] <= switchde_alpha).mean()
    ax.axhline(y=de_frac, color='darkred', zorder=1, label='Dataset ptDE frac')
    hue_order = ('Dataset ptDE frac', ) + hue_order

    # Manually reorder legend
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(s) for s in list(hue_order)[::-1]]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=legend_loc)

    # Set x-ticks at integers
    ax.set_xticks(range(interval, max_top_k + 1, interval))
    ax.set_xticklabels(range(interval, max_top_k + 1, interval))

    # Add vertical lines at multiples of 5
    # Find the smallest multiple of 5 that is greater than or equal to interval
    start_multiple_of_5 = math.ceil(interval / 5) * 5
    for i in range(start_multiple_of_5, max_top_k + 1, 5):
        ax.axvline(x=i, color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)

    if title is not None:
        ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()


def plot_defrac_in_top_k_lineplot2(res_df_list: List[pd.DataFrame],
                                  adata: sc.AnnData,
                                  switchde_alpha: float = 0.01,
                                  max_top_k: int = 20,
                                  axs: Union[plt.Axes, None] = None,
                                  title: Union[str, None] = None,
                                  show: bool = False,
                                  names: Union[List[str], None] = None,
                                  markers: Union[List[str], None] = ('s', 'd', '^', 'o'),
                                  palette: Union[List[str], None] = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'),
                                  hue_order: Union[Tuple[str, ...], None] =
                                  ('CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI'),
                                  legend_loc: str = 'best',
                                  gene_key: str = 'gene',
                                  sdeqval_var_key: str = 'switchde_qval'):

    # Create dataframe for plotting
    df = create_switchdefrac_df(res_df_list=res_df_list,
                                adata=adata,
                                switchde_alpha=switchde_alpha,
                                max_top_k=max_top_k,
                                names=names,
                                gene_key=gene_key,
                                sdeqval_var_key=sdeqval_var_key)
    # Transform from wide into long format for seaborn
    df_melted = pd.melt(df, id_vars=['Top k'], var_name='Method', value_name='ptDE fraction')

    if markers is not None:
        markers = list(markers)
    if palette is not None:
        palette = list(palette)

    ax = sns.lineplot(data=df_melted, x='Top k', y='ptDE fraction', hue='Method', style='Method', markers=markers,
                      palette=palette, dashes=False, hue_order=list(hue_order), ax=axs, zorder=2)

    # Add horizontal line at fraction of de genes in dataset
    de_frac = (adata.var[sdeqval_var_key] <= switchde_alpha).mean()
    ax.axhline(y=de_frac, color='darkred', zorder=1, label='Dataset ptDE frac')
    hue_order = ('Dataset ptDE frac', ) + hue_order

    # Manually reorder legend
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(s) for s in list(hue_order)[::-1]]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=legend_loc)

    # Set x-ticks at integers
    ax.set_xticks(range(max_top_k + 1))

    # Add vertical lines at multiples of 5
    for i in range(0, max_top_k + 1, 5):
        ax.axvline(x=i, color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)

    if title is not None:
        ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()


def plot_digest_results(digest_res_path_list: List[str],
                        name_list: List[str],
                        color_list: List[str],
                        title: Union[str, None] = None,
                        size: float = 5,
                        axs: Union[plt.Axes, None] = None,
                        show: bool = False,
                        verbosity: int = 0):

    # Create dictionary mapping names to colors
    color_dict = {name: col for name, col in zip(name_list, color_list)}
    databases = ['GO.BP', 'GO.CC', 'GO.MF', 'KEGG']
    df = pd.DataFrame()
    name_col = []
    color_col = []
    database_col = []
    digest_scores_col = []
    digest_pvals_col = []
    for i, res_p in enumerate(digest_res_path_list):
        with open(res_p, 'r') as f:
            digest_res = json.load(f)
        for database in databases:
            name_col.append(name_list[i])
            color_col.append(color_list[i])
            database_col.append(database)
            digest_scores_col.append(digest_res['input_values']['values']['JI-based'][database])
            digest_pvals_col.append(digest_res['p_values']['values']['JI-based'][database])

    df['method'] = name_col
    df['color'] = color_col
    df['Database'] = database_col
    df['DIGEST score'] = digest_scores_col
    df['digest_pval'] = digest_pvals_col

    if verbosity >= 1:
        print(df)

    if axs is None:
        ax = sns.stripplot(data=df, x='Database', y='DIGEST score', hue='method', jitter=True, palette=color_dict,
                           size=size)
    else:
        ax = sns.stripplot(data=df, x='Database', y='DIGEST score', hue='method', jitter=True, palette=color_dict,
                           size=size, ax=axs)

    # add grid in background
    ax.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5)

    # remove the legend title
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=labels, title='')

    # Manually reorder legend
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(s) for s in name_list[::-1]]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title='')

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()


def plot_enrichr_results2(res_df: pd.DataFrame,
                         x: str = 'combined score',  # 'z-score', 'p-value', 'q-value', 'combined score'
                         color: str = 'q-value',
                         title: Union[str, None] = None,
                         title_fontsize: Union[str, float, None] = None,
                         term_fontsize: Union[str, float, None] = None,
                         axs: Union[plt.Axes, None] = None,
                         transp: float = 0.8,
                         fig_size: Tuple[float, float] = (16., 10.),
                         show: bool = False):

    # Add column to dataframe with values to be plotted, set label of x-axis
    plot_vals = res_df[x].to_numpy()
    if x in {'p-value', 'q-value'}:
        plot_vals = -np.log(plot_vals)
        x_label = f'-log({x})'
    else:
        # plot_vals = np.log10(1 + plot_vals)
        x_label = 'Combined score'
    res_df['plot_val'] = plot_vals

    # Sort dataframe w.r.t. plot vals
    res_df.sort_values(by='plot_val', ascending=True, inplace=True)

    # Create a color map based on chosen column
    norm = mcolors.Normalize(vmin=res_df[color].min(), vmax=res_df[color].max())
    cmap = plt.get_cmap('coolwarm_r')

    # Create a new figure and axis if not provided
    if axs is None:
        fig, axs = plt.subplots(figsize=fig_size)

    # Create a bar plot
    bars = axs.barh(res_df['Term'], res_df['plot_val'], color=cmap(norm(res_df[color])), alpha=transp)

    # Add color bar
    # Adjust the transparency of the colormap for the colorbar
    cmap_transp = cmap(np.arange(cmap.N))
    cmap_transp[:, -1] = transp  # Set the alpha channel to your desired transparency level (0.7 in this case)
    cmap_transp = mcolors.ListedColormap(cmap_transp)

    sm = plt.cm.ScalarMappable(cmap=cmap_transp, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axs)
    # cbar.set_alpha(transp)
    # cbar._draw_all()  # Todo
    # Force the figure to redraw to apply the transparency setting
    axs.figure.canvas.draw_idle()
    cbar.set_label(color)

    # Annotate bars with the term at the base, with a small space
    space = res_df['plot_val'].max() * 0.02
    for bar, term in zip(bars, res_df['Term']):
        axs.text(bar.get_x() + space, bar.get_y() + bar.get_height() / 2, term[0].upper() + term[1:], va='center',
                 ha='left', fontsize=term_fontsize, color='black')

    # Set library as y-tick labels
    axs.set_yticks(np.arange(len(res_df)))
    axs.set_yticklabels(res_df['Library'], fontsize=8, color='grey')

    # Add labels and title
    axs.set_xlabel(x_label, fontsize=12)

    if title is not None:
        if title_fontsize is not None:
            axs.set_title(title, fontsize=title_fontsize)
        else:
            axs.set_title(title)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # if title is not None:
    #     axs.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()


def plot_enrichrkg_results(res_df: pd.DataFrame,
                           x: str = 'combined score',  # 'z-score', 'p-value', 'q-value', 'combined score'
                           log_trafo: bool = False,
                           title: Union[str, None] = None,
                           title_fontsize: Union[str, float, None] = None,
                           term_fontsize: Union[str, float, None] = None,
                           axs: Union[plt.Axes, None] = None,
                           transp: float = 0.8,
                           fig_size: Tuple[float, float] = (16., 10.),
                           show: bool = False):

    # Add column to dataframe with values to be plotted, set label of x-axis
    plot_vals = res_df[x].to_numpy()
    if x in {'p-value', 'q-value'}:
        plot_vals = -np.log(plot_vals)
        x_label = f'-log({x})'
    else:
        # plot_vals = np.log10(1 + plot_vals)
        x_label = 'Combined score'
        if log_trafo:
            plot_vals = np.log10(plot_vals)
            x_label = 'log10(combined score)'
    res_df['plot_val'] = plot_vals

    # Sort dataframe w.r.t. plot vals
    res_df.sort_values(by='plot_val', ascending=True, inplace=True)

    # Extract unique categories
    unique_categories = res_df['Library'].unique()

    # Create a discrete colormap
    colors = plt.get_cmap('Set3', len(unique_categories))

    # Map unique categories to colors
    color_mapping = {category: colors(i) for i, category in enumerate(unique_categories)}

    # Apply the colormap to the DataFrame
    res_df['Color'] = res_df['Library'].map(color_mapping)

    # Create a new figure and axis if not provided
    if axs is None:
        fig, axs = plt.subplots(figsize=fig_size)

    # Create a bar plot
    bars = axs.barh(res_df['Term'], res_df['plot_val'], color=res_df['Color'], alpha=transp)

    # Annotate bars with the term at the base, with a small space
    space = res_df['plot_val'].max() * 0.02
    for bar, term in zip(bars, res_df['Term']):
        axs.text(bar.get_x() + space, bar.get_y() + bar.get_height() / 2, term[0].upper() + term[1:], va='center',
                 ha='left', fontsize=term_fontsize, color='black')

    # Set library as y-tick labels
    # axs.set_yticks(np.arange(len(res_df)))
    # axs.set_yticklabels(res_df['Library'], fontsize=8, color='grey')
    # Remove y ticks
    axs.set_yticks([])

    # Add labels and title
    axs.set_xlabel(x_label, fontsize=12)

    if title is not None:
        if title_fontsize is not None:
            axs.set_title(title, fontsize=title_fontsize)
        else:
            axs.set_title(title)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Add legend with dots
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[category], markersize=12) for
               category in unique_categories]
    labels = unique_categories
    legend = axs.legend(handles, labels, loc='lower right', fontsize=12)
    legend.get_frame().set_linewidth(0)  # Remove box around the legend

    if show:
        plt.tight_layout()
        plt.show()


def plot_enrichr_results(res_dfs: Sequence[pd.DataFrame],
                         x: str = Literal['P-value', 'Adjusted P-value', 'Combined Score'],
                         top_k: Union[Sequence[int], None] = None,
                         reference_db_names: Union[Sequence[str], None] = None,
                         log_trafo: bool = False,
                         title: Union[str, None] = None,
                         title_fontsize: Union[str, float, None] = None,
                         term_fontsize: Union[str, float, None] = None,
                         axs: Union[plt.Axes, None] = None,
                         transp: float = 0.8,
                         fig_size: Tuple[float, float] = (16., 10.),
                         show: bool = False):

    # ### Columns in the GSEA dataframe:
    # Term	Overlap	P-value	Adjusted P-value	Old P-value	Old Adjusted P-value	Odds Ratio	Combined Score	Genes

    if top_k is None:
        top_k = [3] * len(res_dfs)

    plot_df = pd.DataFrame()

    for i, res_df in enumerate(res_dfs):
        # Add column to dataframe with library/reference database name
        if reference_db_names is not None:
            res_df['Library'] = reference_db_names[i]

        # Add column to dataframe with values to be plotted, set label of x-axis
        plot_vals = res_df[x].to_numpy()
        if x in {'P-value', 'Adjusted P-value'}:
            plot_vals = -np.log(plot_vals)
        else:
            # plot_vals = np.log10(1 + plot_vals)
            if log_trafo:
                plot_vals = np.log10(plot_vals)
        res_df['plot_val'] = plot_vals

        # Sort dataframe w.r.t. plot vals
        res_df.sort_values(by='plot_val', ascending=False, inplace=True)

        first_k_rows = res_df.head(top_k[i])
        if plot_df.empty:
            plot_df = first_k_rows.copy()
        else:
            plot_df = pd.concat([plot_df, first_k_rows], ignore_index=True)

    plot_df.sort_values(by='plot_val', ascending=True, inplace=True)

    if reference_db_names is not None:
        # Extract unique categories
        unique_categories = plot_df['Library'].unique()
        # Create a discrete colormap
        colors = plt.get_cmap('Set3', len(unique_categories))
        # Map unique categories to colors
        color_mapping = {category: colors(i) for i, category in enumerate(unique_categories)}
        # Apply the colormap to the DataFrame
        plot_df['Color'] = plot_df['Library'].map(color_mapping)

    if x in {'P-value', 'Adjusted P-value'}:
        x_label = f'-log10({x})'
    else:
        x_label = 'Combined score'
        if log_trafo:
            x_label = 'log10(combined score)'

    # Create a new figure and axis if not provided
    if axs is None:
        fig, axs = plt.subplots(figsize=fig_size)

    # Create a bar plot
    if reference_db_names is not None:
        bars = axs.barh(plot_df['Term'], plot_df['plot_val'], color=plot_df['Color'], alpha=transp)
    else:
        bars = axs.barh(plot_df['Term'], plot_df['plot_val'], color='lightseagreen', alpha=transp)

    # Annotate bars with the term at the base, with a small space
    space = plot_df['plot_val'].max() * 0.02
    for bar, term in zip(bars, plot_df['Term']):
        axs.text(bar.get_x() + space, bar.get_y() + bar.get_height() / 2, term[0].upper() + term[1:], va='center',
                 ha='left', fontsize=term_fontsize, color='black')

    # Set library as y-tick labels
    # axs.set_yticks(np.arange(len(res_df)))
    # axs.set_yticklabels(res_df['Library'], fontsize=8, color='grey')
    # Remove y ticks
    axs.set_yticks([])

    # Add labels and title
    axs.set_xlabel(x_label, fontsize=12)

    if title is not None:
        if title_fontsize is not None:
            axs.set_title(title, fontsize=title_fontsize)
        else:
            axs.set_title(title)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if reference_db_names is not None:
        # Add legend with dots
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[category], markersize=12)
                   for category in unique_categories]
        labels = unique_categories
        legend = axs.legend(handles, labels, loc='lower right', fontsize=12)
        legend.get_frame().set_linewidth(0)  # Remove box around the legend

    if show:
        plt.tight_layout()
        plt.show()


def plot_circled_tfs(res_df: pd.DataFrame,
                     topk: int = 10,
                     fontsize: int = 6,
                     title: Union[str, None] = None,
                     title_fontsize: Union[str, float, None] = None,
                     res_df2: Union[pd.DataFrame, None] = None,
                     plottwoinone: bool = False,
                     y_ticks: Union[Tuple[str, str], None] = None,
                     axs: Union[plt.Axes, None] = None):

    tfs = res_df['gene'][0:topk].tolist()
    if res_df2 is not None:
        tfs2 = res_df2['gene'][0:topk].tolist()
        inters = set(tfs).intersection(set(tfs2))
        intersec_bool = [False] * topk
        for i, tf in enumerate(tfs):
            if tf in inters:
                intersec_bool[i] = 1
    else:
        intersec_bool = None

    tf_rgba0 = [43, 165, 255, 0.5]
    tf_rgba1 = [43, 165, 255, 0.5]
    tf_cols = get_rgba_color_gradient(rgba1=tf_rgba0,
                                      rgba2=tf_rgba1,
                                      values=len(tfs))

    def plot_strings_with_colors(strings, colors, highlight_bool, fontsize, ypos, ax):
        ax.set_aspect('equal')
        if highlight_bool is None:
            highlight_bool = [False] * len(strings)

        for i, (text, color) in enumerate(zip(strings, colors)):
            ring_col = 'gold' if highlight_bool[i] else 'black'
            circle = plt.Circle((i, ypos), 0.5, color=color, ec=ring_col, linewidth=2)
            ax.add_artist(circle)
            ax.text(i, ypos, text, color='black', ha='center', va='center', fontsize=fontsize)

        ax.set_xlim(-1, len(strings))
        ax.set_ylim(-1, 1)
        # ax.axis('off')
        ax.yaxis.set_visible(False)
        ax.set_xticks(range(0, len(strings)), labels=range(1, len(strings) + 1))
        ax.set_xlabel('Rank', fontsize=12)

    if axs is None:
        fig, axs = plt.subplots()

    plot_strings_with_colors(strings=tfs, colors=tf_cols, highlight_bool=intersec_bool, fontsize=fontsize, ypos=0.0,
                             ax=axs)

    if plottwoinone:
        tfs2 = res_df2['gene'][0:topk].tolist()
        inters = set(tfs).intersection(set(tfs2))
        intersec_bool2 = [False] * topk
        for i, tf in enumerate(tfs2):
            if tf in inters:
                intersec_bool2[i] = 1
        plot_strings_with_colors(strings=tfs2, colors=tf_cols, highlight_bool=intersec_bool2, fontsize=fontsize,
                                 ypos=1.25, ax=axs)
        axs.set_ylim(-1, 2)

        if y_ticks is not None:
            axs.yaxis.set_visible(True)
            axs.set_yticks([0, 1.25], list(y_ticks), fontsize=14)
            axs.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)

    if title is not None:
        if title_fontsize is not None:
            axs.set_title(title, fontsize=title_fontsize)
        else:
            axs.set_title(title)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)


def plot_upset_plot(res_list: List[pd.DataFrame],
                    top_k: int = 20,
                    names: Tuple[str] = ('CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI outdeg', 'SwitchTFI'),
                    title: Union[str, None] = None,
                    gene_key: str = 'gene',
                    plot_folder: Union[str, None] = None,
                    fn_prefix: Union[str, None] = None,
                    axs: Union[plt.Axes, None] = None):

    index_tuples = []
    values = []

    # Iterate over length of combinations
    for i in range(len(res_list)):
        # Iterate over combinations
        for j in combinations(range(len(res_list)), i + 1):
            dummy = [False] * len(res_list)
            inter = set(res_list[j[0]][gene_key][0:top_k].tolist())
            # Iterate over elements of current combination
            for n, k in enumerate(j):
                dummy[k] = True
                inter = inter.intersection(set(res_list[j[n]][gene_key][0:top_k].tolist()))

            index_tuples.append(tuple(dummy))
            values.append(len(inter))

    index = pd.MultiIndex.from_tuples(index_tuples, names=list(names))
    series = pd.Series(values, index=index)
    # upset = UpSet(series)
    upset = UpSet(series, sort_by='cardinality', min_subset_size=1, min_degree=2, totals_plot_elements=0, show_counts="{:,}")
    upset.plot()

    if plot_folder is not None:
        if fn_prefix is None:
            fn_prefix = ''
        plt.savefig(os.path.join(plot_folder, f'{fn_prefix}upsetplot.png'))

    if axs is not None:
        if plot_folder is None:
            plot_folder = './'
        if fn_prefix is None:
            fn_prefix = ''
        plt.savefig(os.path.join(plot_folder, f'{fn_prefix}upsetplot.png'), bbox_inches='tight')
        plt.close(plt.gcf())

        # Load the image
        image = mpimg.imread(os.path.join(plot_folder, f'{fn_prefix}upsetplot.png'))
        # Display the image
        axs.imshow(image)
        # Hide the axes
        axs.axis('off')

        if title is not None:
            axs.set_title(title)

        # Set aspect ratio and limits
        # axs.set_aspect('auto')
        # axs.set_xlim([0, image.shape[1]])
        # axs.set_ylim([image.shape[0], 0])


def plot_cc_score_hist(ccs: list,
                       n_vert: int,
                       ccs_list: list,
                       n_vert_list: list,
                       titel: Union[str, None],
                       axs: plt.Axes):

    # Calculate cc-scores for random subnetworks
    scores = [0] * len(ccs_list)
    for i, cc in enumerate(ccs_list):  # iterate over iterators of components
        scores[i] = sum((len(c) / n_vert_list[i]) ** 2 for c in cc)
    scores = np.array(scores)

    # Calculate cc-score for transition GRN
    score_tgrn = sum((len(c) / n_vert) ** 2 for c in ccs)

    axs.hist(scores, bins=15, edgecolor='black', color='yellow')
    axs.axvline(score_tgrn, color='red', linewidth=2,
                label=f'Score transition GRN: {round(score_tgrn, 4)}\n'
                      f'Empirical P-value: {round(calc_emp_pval(val=score_tgrn, val_vec=scores, geq=True), 4)}')
    axs.legend()

    axs.set_xlabel('Score')
    axs.set_ylabel(f'Count (total: n = {scores.shape[0]})')

    if titel is not None:
        axs.set_title(titel)


# Auxiliary ############################################################################################################
def find_i(grn: pd.DataFrame,
           tf: str,
           target: str,
           tf_target_keys: Tuple[str, str] = ('TF', 'target'),) -> int:
    tf_key = tf_target_keys[0]
    target_key = tf_target_keys[1]
    tf_bool = (grn[tf_key].to_numpy() == tf)
    target_bool = (grn[target_key].to_numpy() == target)

    tf_target_combination_bool = tf_bool * target_bool

    i = int(np.where(tf_target_combination_bool)[0][0])  # Get 1st entry of array corresponding to 1st dimension

    return i


def get_rgba_color_gradient(rgba1: List[float],
                            rgba2: List[float],
                            values: Union[int, np.ndarray]) -> np.ndarray:

    if isinstance(values, int):
        points = np.linspace(0, 1, values)

        out = interpolation_function(rgba1=rgba1, rgba2=rgba2, points=points)
    else:
        out = interpolation_function(rgba1=rgba1, rgba2=rgba2, points=values)

    return out


def interpolation_function(rgba1: List[float],
                           rgba2: List[float],
                           points: np.ndarray) -> np.ndarray:
    # Scale to [0,1] range
    rgba1 = np.array([a / 255 if i < 3 else a for i, a in enumerate(rgba1)])
    rgba2 = np.array([a / 255 if i < 3 else a for i, a in enumerate(rgba2)])

    # Calculate linear interpolation for each value in array -> n-values x 4 matrix
    # out = np.minimum(rgba1, rgba2) + np.expand_dims(points, 1) * (np.maximum(rgba1, rgba2) - np.minimum(rgba1, rgba2))

    out = rgba1 + np.expand_dims(points, 1) * (rgba2 - rgba1)

    return out


def create_switchdefrac_df(res_df_list: List[pd.DataFrame],
                           adata: sc.AnnData,
                           switchde_alpha: float = 0.01,
                           max_top_k: int = 20,
                           names: Union[List[str], None] = None,
                           gene_key: str = 'gene',
                           sdeqval_var_key: str = 'switchde_qval') -> pd.DataFrame:

    frac_df = pd.DataFrame()
    frac_df['Top k'] = np.array(range(1, max_top_k + 1))

    for i, res_df in enumerate(res_df_list):
        frac_dummy = np.zeros(max_top_k)
        for j in range(max_top_k):

            frac_dummy[j] = get_sde_frac(res_df=res_df, adata=adata, top_k=j + 1, sde_alpha=switchde_alpha,
                                         gene_key=gene_key, sdeqval_var_key=sdeqval_var_key)

        if names is None:
            cname = i
        else:
            cname = names[i]
        frac_df[cname] = frac_dummy

    return frac_df


def get_sde_frac(res_df: pd.DataFrame,
                 adata: sc.AnnData,
                 top_k: int,
                 sde_alpha: float,
                 gene_key: str,
                 sdeqval_var_key: str) -> float:

    top_k_genes = res_df[gene_key].tolist()[0:top_k]
    top_k_sdeqvals = adata.var.loc[top_k_genes, sdeqval_var_key].to_numpy()

    return (top_k_sdeqvals <= sde_alpha).mean()


def calc_emp_pval(val: float,
                  val_vec: np.ndarray,
                  geq: bool = True) -> float:
    if geq:
        p_val = 1 + (val <= val_vec).sum()
    else:
        p_val = 1 + (val >= val_vec).sum()

    p_val /= (1 + val_vec.shape[0])

    return p_val


