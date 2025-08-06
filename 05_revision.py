

def main():

    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scanpy as sc

    from scipy.stats import pearsonr
    from switchtfi.utils import align_anndata_grn
    from switchtfi.weight_fitting import fit_regression_stump_model, prune_special_cases, prune_wrt_n_cells
    from switchtfi import compute_corrected_pvalues, rank_tfs, remove_insignificant_edges

    np.random.seed(42)

    res_path = './results/05_revision/no_imputation/preliminary'
    os.makedirs(res_path, exist_ok=True)

    datasets = ['ery', 'beta', 'alpha']

    layers = ['log1p_norm', 'magic_imputed']

    n_cell_pruning_params = [
        (None, None), ('percent', 0.01), ('percent', 0.05),  ('percent', 0.1), ('percent', 0.15), ('percent', 0.2),
        ('percent', 0.25), ('percent', 0.3), ('percent', 0.35), ('percent', 0.4), ('percent', 0.45), ('percent', 0.5)
    ]

    for dataset in datasets:

        # ### Compare the weights
        res_subdir = os.path.join(res_path, dataset)
        os.makedirs(res_subdir, exist_ok=True)

        # Load the anndata object with the data and the precomputed GRN
        if dataset == 'ery':
            data = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')
            grn = pd.read_csv(
                './results/01_grn_inf/hematopoiesis/ngrnthresh9_erythrocytes_pyscenic_combined_grn.csv',
                index_col=0
            )
            cell_anno_key = 'prog_off'

        elif dataset == 'beta':
            data = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
            grn = pd.read_csv(
                './results/01_grn_inf/endocrine/alpha/ngrnthresh9_alpha_pyscenic_combined_grn.csv',
                index_col=0
            )
            cell_anno_key = 'clusters'

        else:
            data = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
            grn = pd.read_csv(
                './results/01_grn_inf/endocrine/beta/ngrnthresh9_beta_pyscenic_combined_grn.csv',
                index_col=0
            )
            cell_anno_key = 'clusters'

        # Align the data and GRN
        data, grn = align_anndata_grn(
            adata=data,
            grn=grn,
        )

        # Compute edge-wise weights using each layer
        layer_to_grn_weighted = dict()

        for layer in layers:

            grn_weighted = fit_regression_stump_model(
                adata=data,
                grn=grn.copy(),
                layer_key=layer,
                result_folder=None,
                clustering_obs_key=cell_anno_key,
            )

            grn_weighted, _ = prune_special_cases(
                grn=grn_weighted,
                result_folder=res_subdir,
                verbosity=1,
                fn_prefix=layer + '_',
            )

            layer_to_grn_weighted[layer] = grn_weighted.copy()

            # ### Compare the empirical p-values at different n-cell thresholding levels
            for mode, thresh in n_cell_pruning_params:

                # Define dir where to save results
                subdir = f'thresh_{str(thresh).replace(".", "_")}' if mode is not None else 'thresh_0'
                res_subsubdir = os.path.join(res_subdir, 'p_values', subdir)
                os.makedirs(res_subsubdir, exist_ok=True)

                # Prune thr GRN w.r.t. the number of cells used for fitting the regression stump
                if mode is not None:

                    grn_weighted_pruned = prune_wrt_n_cells(
                        grn=grn_weighted.copy(),
                        mode=mode,
                        threshold=thresh,
                    )

                else:

                    grn_weighted_pruned = grn_weighted.copy()


                grn_pval = compute_corrected_pvalues(
                    adata=data,
                    grn=grn_weighted_pruned,
                    result_folder=res_subsubdir,
                    clustering_obs_key=cell_anno_key,
                    fn_prefix=layer + '_'
                )

                grn_pval[['TF', 'target', 'weight', 'pvals_wy']].to_csv(
                    os.path.join(res_subsubdir, f'{layer}_grn_all_edges.csv')
                )

                transition_grn = remove_insignificant_edges(
                    grn=grn_pval.copy(),
                    alpha=0.05,
                    result_folder=res_subsubdir,
                    fn_prefix=layer + '_',
                )

                transition_grn[['TF', 'target', 'weight', 'pvals_wy']].to_csv(
                    os.path.join(res_subsubdir, f'{layer}_transition_grn.csv')
                )

                ranked_tfs = rank_tfs(
                    grn=transition_grn,
                    result_folder=res_subsubdir,
                    fn_prefix=layer + '_',
                )


        # Plot the number odf cells used during the fit against the weight
        fig, axs = plt.subplots(1, len(layers), figsize=(10, 4), dpi=300)

        for i, layer in enumerate(layers):
            grn = layer_to_grn_weighted[layer]
            ax = axs[i]

            weights = grn['weight'].to_numpy()
            n_cells = np.array([a.sum() for a in grn['cell_bool'].to_numpy()])

            correlation, p_val = pearsonr(n_cells, weights)

            median_weight = np.median(weights)
            median_n_cells = np.median(n_cells)

            ax.scatter(x=n_cells, y=weights, marker='.')

            ax.axvline(x=median_n_cells, color='green', label=f'median # cells per edge: {int(median_n_cells)}')
            ax.axhline(y=median_weight, color='red', label=f'median weight: {round(median_weight, 3)}')
            ax.set_xlabel('# cells per edge')
            ax.set_ylabel('weight')
            ax.legend()
            ax.set_title(f'{layer}, Pearson c.: {round(correlation, 3)}')

        fig.savefig(os.path.join(res_subdir, 'n_cells_vs_weight.png'))

        # Plot the n cell hist and thresholds
        fig, axs = plt.subplots(1, len(layers), figsize=(10, 4), dpi=300)

        for i, layer in enumerate(layers):
            grn = layer_to_grn_weighted[layer]
            ax = axs[i]

            cell_bool_array = np.vstack(grn['cell_bool'])
            n_cells = cell_bool_array.sum(axis=1)
            n_cells_max = np.max(n_cells)

            ax.hist(n_cells, bins=30, edgecolor='grey')

            ax.set_xlabel('# cells per edge')
            ax.set_ylabel('count')
            ax.set_title(layer)

            for mode, threshold in n_cell_pruning_params:

                if mode is None:
                    continue

                ax.axvline(x=threshold * n_cells_max, color='red', linestyle='-')

                # Add text near the top of the line
                ax.text(
                    threshold * n_cells_max,
                    ax.get_ylim()[1] * 0.98,
                    f'{threshold}',
                    rotation=90,
                    verticalalignment='top',
                    horizontalalignment='left',
                    color='red',
                    fontsize=8
                )

        fig.savefig(os.path.join(res_subdir, 'p_values', 'n_cells_and_thresholds.png'))


    for dataset in datasets:

        res_subdir = os.path.join(res_path, dataset)

        for mode, threshold in n_cell_pruning_params:

            # Define dir where to save results
            subdir = f'thresh_{str(threshold).replace(".", "_")}' if mode is not None else 'thresh_0'
            res_subsubdir = os.path.join(res_subdir, 'p_values', subdir)

            # Plot weights vs pvalues
            fig, axs = plt.subplots(1, len(layers), figsize=(10, 4), dpi=300)

            for i, layer in enumerate(layers):

                grn = pd.read_csv(os.path.join(res_subsubdir, f'{layer}_grn_all_edges.csv'), index_col=0)

                ax = axs[i]

                weights = grn['weight'].to_numpy()
                p_vals = grn['pvals_wy'].to_numpy()

                ax.scatter(
                    x=weights,
                    y=p_vals,
                    marker='o',
                    s=20,
                    alpha=0.9,
                    edgecolors='gray',
                    linewidth=0.5
                )

                ax.axhline(y=0.05, color='red', label='0.05')
                ax.axhline(y=0.01, color='orange', label='0.01')

                ax.set_xlabel('weight')
                ax.set_ylabel('p-value')
                ax.legend()
                ax.set_title(f'{layer}')

            fig.savefig(os.path.join(res_subsubdir, 'weights_vs_pvalues.png'))



        # ### Compare the similarity of the results
        res_dict = dict()
        for mode, threshold in n_cell_pruning_params:

            # Define dir where to save results
            subdir = f'thresh_{str(threshold).replace(".", "_")}' if mode is not None else 'thresh_0'
            res_subsubdir = os.path.join(res_subdir, 'p_values', subdir)
            if threshold is None:
                threshold = 0.0

            tgrn_imp = pd.read_csv(os.path.join(res_subsubdir, 'magic_imputed_transition_grn.csv'), index_col=0)
            tgrn_no_imp = pd.read_csv(os.path.join(res_subsubdir, 'log1p_norm_transition_grn.csv'), index_col=0)

            tfs_imp = pd.read_csv(os.path.join(res_subsubdir, 'magic_imputed_ranked_tfs.csv'), index_col=0)
            tfs_no_imp = pd.read_csv(os.path.join(res_subsubdir, 'log1p_norm_ranked_tfs.csv'), index_col=0)

            dummy = dict()

            dummy['n_edges_imp'] = tgrn_imp.shape[0]
            dummy['n_edges_no_imp'] = tgrn_no_imp.shape[0]

            dummy['n_nodes_imp'] = np.unique(tgrn_imp[['TF', 'target']].to_numpy()).shape[0]
            dummy['n_nodes_no_imp'] = np.unique(tgrn_no_imp[['TF', 'target']].to_numpy()).shape[0]

            dummy['n_tfs_imp'] = np.unique(tgrn_imp['TF'].to_numpy()).shape[0]
            dummy['n_tfs_no_imp'] = np.unique(tgrn_no_imp['TF'].to_numpy()).shape[0]

            dummy['n_targets_imp'] = np.unique(tgrn_imp['target'].to_numpy()).shape[0]
            dummy['n_targets_no_imp'] = np.unique(tgrn_no_imp['target'].to_numpy()).shape[0]

            edges_imp = set(tgrn_imp[['TF', 'target']].itertuples(index=False, name=None))
            edges_no_imp = set(tgrn_no_imp[['TF', 'target']].itertuples(index=False, name=None))
            common_edges = edges_imp & edges_no_imp
            dummy['common_edges'] = len(common_edges)

            tfs_imp_unique = set(tgrn_imp['TF'].to_numpy())
            tfs_no_imp_unique = set(tgrn_no_imp['TF'].to_numpy())
            common_tfs = tfs_imp_unique & tfs_no_imp_unique
            dummy['common_tfs'] = len(common_tfs)

            for i in [5, 10, 15, 20, 25]:
                if tfs_imp.shape[0] >= i and tfs_no_imp.shape[0] >= i:

                    tfs_imp_unique = set(tfs_imp['gene'].to_numpy()[0:i])
                    tfs_no_imp_unique = set(tfs_no_imp['gene'].to_numpy()[0:i])
                    common_tfs = tfs_imp_unique & tfs_no_imp_unique
                    dummy[f'common_tfs_top_{i}'] = len(common_tfs)

                else:
                    dummy[f'common_tfs_top_{i}'] = np.nan

            res_dict[threshold] = dummy.copy()

        res_df = pd.DataFrame.from_dict(res_dict, orient='index')

        res_df.to_csv(os.path.join(res_path, dataset, 'res_df.csv'))

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 0)
        pd.set_option('display.max_colwidth', None)
        print(res_df)


def main_visualize_results():

    import os
    import numpy as np

    import matplotlib.pyplot as plt
    import scanpy as sc

    from switchtfi.utils import csr_to_numpy, load_grn_json

    res_path = './results/05_revision/no_imputation/preliminary'
    save_path = './results/05_revision/no_imputation/preliminary/plots'
    os.makedirs(save_path, exist_ok=True)

    datasets = ['ery', 'beta', 'alpha']

    for dataset in datasets:
        if dataset == 'ery':
            data = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')
            cell_anno_key = 'prog_off'

        elif dataset == 'beta':
            data = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
            cell_anno_key = 'clusters'

        else:
            data = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
            cell_anno_key = 'clusters'

        grn_magic = load_grn_json(
            os.path.join(res_path, dataset, 'p_values', 'thresh_0_2', 'magic_imputed_sigpvals_wy0.05_grn.json')
        ).sort_values('weight', ascending=False).reset_index(drop=True)

        grn_log1p_norm = load_grn_json(
            os.path.join(res_path, dataset, 'p_values', 'thresh_0_2', 'log1p_norm_sigpvals_wy0.05_grn.json'),
        ).sort_values('weight', ascending=False).reset_index(drop=True)

        top_k = min(3, grn_log1p_norm.shape[0], grn_magic.shape[0])

        # ### Plot the top k weighted edges on MAGIC imputed data
        fig, axs = plt.subplots(top_k, 2, figsize=(12, 4 * top_k), dpi=300)

        for i in range(top_k):

            tf = grn_magic.loc[i, 'TF']
            target = grn_magic.loc[i, 'target']
            weight = grn_magic.loc[i, 'weight']
            threshold = grn_magic.loc[i, 'threshold']
            pred_l = grn_magic.loc[i, 'pred_l']
            pred_r = grn_magic.loc[i, 'pred_r']

            row = grn_log1p_norm[(grn_log1p_norm['TF'] == tf) & (grn_log1p_norm['target'] == target)]


            weight_no_magic = float(row['weight'].to_numpy()[0]) if not row.empty else -1
            threshold_no_magic = float(row['threshold'].to_numpy()[0]) if not row.empty else np.nan
            pred_l_no_magic = float(row['pred_l'].to_numpy()[0]) if not row.empty else np.nan
            pred_r_no_magic = float(row['pred_r'].to_numpy()[0]) if not row.empty else np.nan

            x_magic = csr_to_numpy(data[:, tf].layers['magic_imputed']).flatten()
            y_magic = csr_to_numpy(data[:, target].layers['magic_imputed']).flatten()
            x_no_magic = csr_to_numpy(data[:, tf].layers['log1p_norm']).flatten()
            y_no_magic = csr_to_numpy(data[:, target].layers['log1p_norm']).flatten()
            labels = data.obs[cell_anno_key].to_numpy()

            x_bool = (x_magic != 0)
            y_bool = (y_magic != 0)
            keep_bool = np.logical_and(x_bool, y_bool)

            x_magic = x_magic[keep_bool]
            y_magic = y_magic[keep_bool]
            x_no_magic = x_no_magic[keep_bool]
            y_no_magic = y_no_magic[keep_bool]
            labels = labels[keep_bool]

            unique_labels  =np.unique(labels)
            cmap = plt.get_cmap('Set2')
            label_to_color = {
                label: cmap(i / len(unique_labels)) for i, label in enumerate(unique_labels)
            }
            colors = [label_to_color[label] for label in labels]
            legend_handles = [
                plt.Line2D([], [], marker='o', color='w', label=label, markerfacecolor=color, markersize=6)
                for label, color in label_to_color.items()
            ]

            # Plot for imputed data
            ax = axs[i, 0]

            ax.scatter(
                x_magic,
                y_magic,
                c=colors,
                cmap='Set2',
                alpha=0.8,
                edgecolors='none',
                s=10,
            )

            min_x, max_x = x_magic.min(), x_magic.max()

            ax.plot([min_x, threshold], [pred_l, pred_l], color='red', zorder=2)
            ax.scatter([threshold], [pred_l], color='red', marker='o', zorder=3)
            ax.plot([threshold, max_x], [pred_r, pred_r], color='red', zorder=2)
            ax.scatter([threshold], [pred_r], color='red', marker='o', facecolor='white', zorder=3)
            ax.axvline(x=threshold, color='red', linestyle='--', zorder=1)
            ax.set_title(f'MAGIC imputed, weight: {round(weight, 4)}')
            ax.set_xlabel(tf)
            ax.set_ylabel(target)

            ax.legend(handles=legend_handles, title='Label')

            # Plot for non-imputed data
            ax = axs[i, 1]

            ax.scatter(
                x_no_magic,
                y_no_magic,
                c=colors,
                cmap='Set2',
                alpha=0.8,
                edgecolors='none',
                s=10,
            )

            min_x, max_x = x_no_magic.min(), x_no_magic.max()

            if weight_no_magic != -1:
                ax.plot([min_x, threshold_no_magic], [pred_l_no_magic, pred_l_no_magic], color='red', zorder=2)
                ax.scatter([threshold_no_magic], [pred_l_no_magic], color='red', marker='o', zorder=3)
                ax.plot([threshold_no_magic, max_x], [pred_r_no_magic, pred_r_no_magic], color='red', zorder=2)
                ax.scatter([threshold_no_magic], [pred_r_no_magic], color='red', marker='o', facecolor='white', zorder=3)
                ax.axvline(x=threshold_no_magic, color='red', linestyle='--', label='Dec. bound.', zorder=1)
            title = f'No imputation, weight: {round(weight_no_magic, 4)}' if weight_no_magic!= -1 else 'No imputation'
            ax.set_title(title)
            ax.set_xlabel(tf)
            ax.set_ylabel(target)

            ax.legend(handles=legend_handles, title='Label')

        plt.tight_layout()
        fig.savefig(os.path.join(save_path, f'{dataset}_topk_on_magic_scatter_plot.png'))

        # ### Plot the top k weighted edges on non-imputed data
        fig, axs = plt.subplots(top_k, 2, figsize=(12, 4 * top_k), dpi=300)

        for i in range(top_k):

            tf = grn_log1p_norm.loc[i, 'TF']
            target = grn_log1p_norm.loc[i, 'target']
            weight = grn_log1p_norm.loc[i, 'weight']
            threshold = grn_log1p_norm.loc[i, 'threshold']
            pred_l = grn_log1p_norm.loc[i, 'pred_l']
            pred_r = grn_log1p_norm.loc[i, 'pred_r']

            row = grn_magic[(grn_magic['TF'] == tf) & (grn_magic['target'] == target)]

            weight_magic = float(row['weight'].to_numpy()[0]) if not row.empty else -1
            threshold_magic = float(row['threshold'].to_numpy()[0]) if not row.empty else np.nan
            pred_l_magic = float(row['pred_l'].to_numpy()[0]) if not row.empty else np.nan
            pred_r_magic = float(row['pred_r'].to_numpy()[0]) if not row.empty else np.nan

            x_no_magic = csr_to_numpy(data[:, tf].layers['log1p_norm']).flatten()
            y_no_magic = csr_to_numpy(data[:, target].layers['log1p_norm']).flatten()
            x_magic = csr_to_numpy(data[:, tf].layers['magic_imputed']).flatten()
            y_magic = csr_to_numpy(data[:, target].layers['magic_imputed']).flatten()
            labels = data.obs[cell_anno_key].to_numpy()

            x_bool_no_magic = (x_magic != 0)
            y_bool_no_magic = (y_magic != 0)
            keep_bool_no_magic = np.logical_and(x_bool_no_magic, y_bool_no_magic)

            x_bool_magic = (x_magic != 0)
            y_bool_magic = (y_magic != 0)
            keep_bool_magic = np.logical_and(x_bool_magic, y_bool_magic)

            x_no_magic = x_no_magic[keep_bool_no_magic]
            y_no_magic = y_no_magic[keep_bool_no_magic]
            labels_no_magic = labels[keep_bool_no_magic]

            x_magic = x_magic[keep_bool_magic]
            y_magic = y_magic[keep_bool_magic]
            labels_magic = labels[keep_bool_magic]

            unique_labels = np.unique(labels)
            cmap = plt.get_cmap('Set2')
            label_to_color = {
                label: cmap(i / len(unique_labels)) for i, label in enumerate(unique_labels)
            }
            colors_no_magic = [label_to_color[label] for label in labels_no_magic]
            colors_magic = [label_to_color[label] for label in labels_magic]
            legend_handles = [
                plt.Line2D([], [], marker='o', color='w', label=label, markerfacecolor=color, markersize=6)
                for label, color in label_to_color.items()
            ]

            # Plot for imputed data
            ax = axs[i, 0]

            ax.scatter(
                x_no_magic,
                y_no_magic,
                c=colors_no_magic,
                cmap='Set2',
                alpha=0.8,
                edgecolors='none',
                s=10,
            )

            min_x, max_x = x_no_magic.min(), x_no_magic.max()

            ax.plot([min_x, threshold], [pred_l, pred_l], color='red', zorder=2)
            ax.scatter([threshold], [pred_l], color='red', marker='o', zorder=3)
            ax.plot([threshold, max_x], [pred_r, pred_r], color='red', zorder=2)
            ax.scatter([threshold], [pred_r], color='red', marker='o', facecolor='white', zorder=3)
            ax.axvline(x=threshold, color='red', linestyle='--', zorder=1)
            ax.set_title(f'No imputation, weight: {round(weight, 4)}')
            ax.set_xlabel(tf)
            ax.set_ylabel(target)

            ax.legend(handles=legend_handles, title='Label')

            # Plot for non-imputed data
            ax = axs[i, 1]

            ax.scatter(
                x_magic,
                y_magic,
                c=colors_magic,
                cmap='Set2',
                alpha=0.8,
                edgecolors='none',
                s=10,
            )

            min_x, max_x = x_magic.min(), x_magic.max()

            if weight_magic != -1:
                ax.plot([min_x, threshold_magic], [pred_l_magic, pred_l_magic], color='red', zorder=2)
                ax.scatter([threshold_magic], [pred_l_magic], color='red', marker='o', zorder=3)
                ax.plot([threshold_magic, max_x], [pred_r_magic, pred_r_magic], color='red', zorder=2)
                ax.scatter([threshold_magic], [pred_r_magic], color='red', marker='o', facecolor='white',
                           zorder=3)
                ax.axvline(x=threshold_magic, color='red', linestyle='--', label='Dec. bound.', zorder=1)
            ax.set_xlabel(tf)
            ax.set_ylabel(target)
            title = f'MAGIC imputed, weight: {round(weight_magic, 4)}' if weight_magic != -1 else 'MAGIC imputed'
            ax.set_title(title)

            ax.legend(handles=legend_handles, title='Label')

        plt.tight_layout()
        fig.savefig(os.path.join(save_path, f'{dataset}_topk_on_non_imputed_scatter_plot.png'))


def main_no_imputation_results():

    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    import seaborn as sns
    import scanpy as sc

    from switchtfi.utils import align_anndata_grn, labels_to_bool, solve_lsap
    from switchtfi.weight_fitting import fit_regression_stump_model, prune_special_cases

    np.random.seed(42)

    res_path = './results/05_revision/no_imputation'
    os.makedirs(res_path, exist_ok=True)

    datasets = ['ery', 'beta', 'alpha']

    layers = ['log1p_norm', 'magic_imputed']

    inference = False

    plot = True

    if inference:

        for dataset in datasets:

            # ### Compare the weights
            res_subdir = os.path.join(res_path, dataset)
            os.makedirs(res_subdir, exist_ok=True)

            # Load the anndata object with the data and the precomputed GRN
            if dataset == 'ery':
                data = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')
                grn = pd.read_csv(
                    './results/01_grn_inf/hematopoiesis/ngrnthresh9_erythrocytes_pyscenic_combined_grn.csv',
                    index_col=0
                )
                cell_anno_key = 'prog_off'

            elif dataset == 'beta':
                data = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
                grn = pd.read_csv(
                    './results/01_grn_inf/endocrine/alpha/ngrnthresh9_alpha_pyscenic_combined_grn.csv',
                    index_col=0
                )
                cell_anno_key = 'clusters'

            else:
                data = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
                grn = pd.read_csv(
                    './results/01_grn_inf/endocrine/beta/ngrnthresh9_beta_pyscenic_combined_grn.csv',
                    index_col=0
                )
                cell_anno_key = 'clusters'

            # Align the data and GRN
            data, grn = align_anndata_grn(
                adata=data,
                grn=grn,
            )

            for layer in layers:

                # ### Compute weighted GRN on non-permuted data
                grn_weighted = fit_regression_stump_model(
                    adata=data.copy(),
                    grn=grn.copy(),
                    layer_key=layer,
                    result_folder=None,
                    clustering_obs_key=cell_anno_key,
                )

                grn_weighted, _ = prune_special_cases(
                    grn=grn_weighted,
                    result_folder=res_subdir,
                    verbosity=1,
                    fn_prefix=layer + '_',
                )

                # Compute the number of cells used for fitting the regression stump
                n_cells = np.array([a.sum() for a in grn_weighted['cell_bool']])
                grn_weighted['num_cells'] = n_cells

                # ### Compute weighted GRNs on permuted data
                n_permutations = 10

                n_edges = grn_weighted.shape[0]

                # Get labels fom anndata, turn into bool vector
                labels = labels_to_bool(data.obs[cell_anno_key].to_numpy())

                # Initialize container to store the weights computed with permuted labels
                permutation_weights = np.zeros((n_edges, n_permutations))

                # Iterate over edges
                for i in range(n_edges):

                    # Get labels of the cells that were used for fitting the weight
                    cell_bool = grn_weighted['cell_bool'].iloc[i]
                    edge_labels = labels[cell_bool]

                    # Get clustering derived from the regressions stump during the weight calculation
                    clustering_dt_reg = grn_weighted['cluster_bool_dt'].iloc[i]

                    for j in range(n_permutations):
                        # Permute labels and compute weight
                        permutation_weights[i, j] = solve_lsap(
                            clust1=clustering_dt_reg,
                            clust2=np.random.permutation(edge_labels),
                        )

                permutation_weights_df = pd.DataFrame(
                    data=permutation_weights,
                    columns=[f'perm_w_{i}' for i in range(n_permutations)],
                )

                # Join the weighted GRN and computed permutation weights
                grn_joined = pd.concat(
                    [grn_weighted.reset_index(drop=True), permutation_weights_df.reset_index(drop=True)],
                    axis=1
                )

                keep_columns = ['TF', 'target', 'weight', 'num_cells'] + [f'perm_w_{i}' for i in range(n_permutations)]
                grn_joined[keep_columns].to_csv(os.path.join(res_subdir, f'{layer}_grn.csv'))

                print(grn_joined[keep_columns])

    if plot:

        trafo = 'log1p_norm'  # 'log1p_norm', 'magic_imputed'

        ds_to_ds_name = {'beta': 'Beta', 'alpha': 'Alpha', 'ery': 'Ery'}

        fig = plt.figure(figsize=(10, 8), constrained_layout=True, dpi=300)
        axd = fig.subplot_mosaic(
            # [
            #     list('A' * 5 + 'B' * 5 +  'C' * 5),
            #     list('D' * 3 + 'E' * 3 + 'F' * 3 + 'G' * 3 + 'H' * 3),
            # ],
            [
                list('A' * 2 + 'B' * 2 + 'C' * 2),
                list('DEFGHI'),
                list('JKLMNO'),
                list('PQRSTU')
            ],
            gridspec_kw={'height_ratios': [2, 1, 1, 1]}
        )

        for dataset, plot_label in zip(['beta', 'alpha', 'ery'], list('ABC')):

            grn_path = os.path.join(res_path, dataset, f'{trafo}_grn.csv')
            grn = pd.read_csv(grn_path, index_col=0)

            ax = axd[plot_label]

            sns.scatterplot(
                data=grn,
                x='num_cells',
                y='weight',
                s=3,
                edgecolor='darkgrey',
                linewidth=0.3,
                color='royalblue',
                ax=ax
            )

            median_num_cells = np.median(grn['num_cells'].to_numpy())

            ax.axvline(
                median_num_cells,
                color='darkred',
                linestyle='-',
                linewidth=1,
                label=f'# cells median: {int(median_num_cells)}'
            )

            ax.legend()

            ax.set_xlabel('# cells')
            ax.set_ylabel('Weight')

            ax.set_title(ds_to_ds_name[dataset])

        for dataset, plot_labels in zip(['beta', 'alpha', 'ery'], ['DEFGHI', 'JKLMNO', 'PQRSTU']):

            grn_path = os.path.join(res_path, dataset, f'{trafo}_grn.csv')
            grn = pd.read_csv(grn_path, index_col=0)

            for i, plot_label in enumerate(list(plot_labels)):

                ax = axd[plot_label]

                sns.scatterplot(
                    data=grn,
                    x='num_cells',
                    y=f'perm_w_{i}',
                    s=3,
                    edgecolor='darkgrey',
                    linewidth=0.3,
                    color='royalblue',
                    ax=ax
                )

                ax.set_xlabel('# cells')
                ax.set_ylabel('Weight')

                # ax.set_xlabel('')
                # ax.set_ylabel('')

                ax.set_title(ds_to_ds_name[dataset])

        for label, ax in axd.items():
            trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
            ax.text(
                0.0, 0.95, label, transform=ax.transAxes + trans,
                fontsize=12, va='bottom', fontfamily='sans-serif', fontweight='bold'
            )

        plt.savefig(os.path.join(res_path, f'{trafo}_weights_vs_num_cells.png'), dpi=300)






if __name__ == '__main__':

    # main_no_imputation_results()

    # main()

    # main_visualize_results()

    import scanpy as sc

    erydata = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')
    adata = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
    bdata = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')

    print(erydata.X)
    print(adata.X)
    print(bdata.X)

    print('done')

