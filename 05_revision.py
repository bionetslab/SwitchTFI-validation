

def main_no_imputation_deprecated():

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


def main_visualize_no_imputation_results_deprecated():

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


def main_targets_enrichment_analysis():

    import os

    import numpy as np
    import pandas as pd

    save_path = './results/05_revision/enrichment_analysis_targets'
    os.makedirs(save_path, exist_ok=True)

    res_path_base = './results/02_switchtfi'
    dataset_to_res_dir = {'Beta': 'endocrine/beta', 'Alpha': 'endocrine/alpha', 'Erythrocytes': 'hematopoiesis'}

    datasets = ['Beta', 'Alpha', 'Erythrocytes']
    centrality_measures = ['pagerank', 'outdeg']

    num_top_tfs_list = [10, 10000]
    num_top_targets_list = [1, 2, 5, 10, 10000]

    sort_by = 'weight'  # 'weight', 'score'

    get_targets = True
    if get_targets:
        def get_top_targets(
                g: pd.DataFrame,
                tfs: list[str],
                num_top_targets: int = 3,
                weight_key: str = 'weight'
        ) -> list[str]:

            targets = []
            for tf in tfs:
                g_sub = g[g['TF'] == tf].copy()
                g_sub_sorted = g_sub.sort_values(by=weight_key, ascending=False)
                top_targets = g_sub_sorted['target'].tolist()[:num_top_targets]
                targets.extend(top_targets)

            return targets

        save_path_targets_naive = os.path.join(save_path, 'top_targets_naive')
        os.makedirs(save_path_targets_naive, exist_ok=True)

        save_path_targets = os.path.join(save_path, 'top_targets')
        os.makedirs(save_path_targets, exist_ok=True)

        for dataset in datasets:

            results_path = os.path.join(res_path_base, dataset_to_res_dir[dataset])

            # Load the transition GRN
            grn = pd.read_csv(os.path.join(results_path, 'grn.csv'), index_col=0)

            # Compute the score
            weights = grn['weight'].to_numpy()
            pvals = grn['pvals_wy'].to_numpy()
            pvals += np.finfo(np.float64).eps
            grn['score'] = -np.log10(pvals) * weights

            grn = grn.sort_values(by=sort_by, ascending=False)

            # Extract top targets naively
            for top_k in [10, 20]:
                top_targets = grn['target'].tolist()[:top_k]

                fn = f'{dataset}_num_targets_{top_k}.txt'

                with open(os.path.join(save_path_targets_naive, fn), 'w') as f:
                    for t in top_targets:
                        f.write(t + '\n')

            # Extract top targets per TF
            for cm in centrality_measures:

                cm_str = (cm + '_') if cm == 'outdeg' else ''
                ranked_tfs = pd.read_csv(os.path.join(results_path, f'{cm_str}ranked_tfs.csv'), index_col=0)

                for n_tf in num_top_tfs_list:

                    top_tfs = ranked_tfs['gene'].tolist()[:n_tf]

                    for n_target in num_top_targets_list:

                        if (n_tf == 5 and n_target == 1) or (n_tf == 10 and n_target == 10) or (n_tf == 10000 and n_target != 10000):
                            continue

                        top_targets = get_top_targets(
                            g=grn,
                            tfs=top_tfs,
                            num_top_targets=n_target,
                            weight_key=sort_by
                        )

                        fn = f'{dataset}_{cm}_num_tfs_{n_tf}_num_targets_{n_target}.txt'

                        with open(os.path.join(save_path_targets, fn), 'w') as f:
                            for t in top_targets:
                                f.write(t + '\n')


def main_targets_enrichment_plots():

    # Results for top 10 TFs by PR and respectively the top 5 targets by weight

    import os

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    from validation.plotting import plot_enrichr_results

    res_dir = './results/05_revision/enrichment_analysis_targets'
    datasets = ['Beta', 'Alpha', 'Erythrocytes']

    dataset_to_dataset_shorthand = {'Beta': 'beta', 'Alpha': 'alpha', 'Erythrocytes': 'ery'}

    fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        '''
        A
        B
        C
        '''
    )

    for subplot_key, dataset in zip(list('ABC'), datasets):

        res_p_gobp = os.path.join(
            res_dir,
            'enrichr_results',
            f'{dataset_to_dataset_shorthand[dataset]}_top10_tfs_top5_targets_GO_Biological_Process_2023_table.txt'
        )
        res_p_reactome = os.path.join(
            res_dir,
            'enrichr_results',
            f'{dataset_to_dataset_shorthand[dataset]}_top10_tfs_top5_targets_Reactome_2022_table.txt'
        )
        res_gobp = pd.read_csv(res_p_gobp, delimiter='\t')
        res_reactome = pd.read_csv(res_p_reactome, delimiter='\t')

        plot_enrichr_results(
            res_dfs=[res_gobp, res_reactome],
            x='Adjusted P-value',
            top_k=[6, 6],
            reference_db_names=['GO_Biological_Process_2023', 'Reactome_2022'],
            term_fontsize=8,
            axs=axd[subplot_key],
        )

        axd[subplot_key].set_title(dataset)

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=None, va='bottom', fontfamily='sans-serif', fontweight='bold')

    fig.savefig(os.path.join(res_dir, 'enrichr_results_targets.png'), dpi=fig.dpi)



    # Todo


def main_tf_ranking_similarity():
    import os
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    import seaborn as sns

    from tqdm import tqdm
    from scipy.stats import kendalltau, spearmanr

    # ### Note:
    # - Kendall's Tau: Count dis-/concordant pairs (with -1, +1 => ranges in [-1, 1])
    # - Spearman's Rank Correlation: Pearson correlation between the rank rvs (ranges in [-1, 1])

    random.seed(42)

    save_p = './results/05_revision/tf_ranking_similarity'
    os.makedirs(save_p, exist_ok=True)

    datasets = ['Beta', 'Alpha', 'Erythrocytes']
    centrality_measures = ['pagerank', 'out_degree']
    ranking_scores = ['tau', 'rho']

    cm_to_cm_name = {'pagerank': 'PR', 'out_degree': 'OD'}

    ranking_score_to_symbol = {'tau': r'$\tau$', 'rho': r'$\rho$', 'ndcg': 'nDCG'}

    base_res_dir = './results/02_switchtfi'
    dataset_to_res_dir = {'Beta': 'endocrine/beta', 'Alpha': 'endocrine/alpha', 'Erythrocytes': 'hematopoiesis'}

    num_permutations = 10000

    def compute_tau_rho(
            ranking_1: list[str], ranking_2: list[str], verbosity: int = 0
    ) -> dict[str, tuple[float, float]]:

        # Map genes to ranks
        gene_to_rank_r1 = {gene: i for i, gene in enumerate(ranking_1)}
        gene_to_rank_r2 = {gene: i for i, gene in enumerate(ranking_2)}

        # Align items -> same order in array (pr as reference), rank with pr/od as entry
        all_tfs = list(sorted(ranking_1))
        rank_list_r1 = [gene_to_rank_r1[tf] for tf in all_tfs]
        rank_list_r2 = [gene_to_rank_r2[tf] for tf in all_tfs]

        tau, pval_tau = kendalltau(rank_list_r1, rank_list_r2)
        rho, pval_rho = spearmanr(rank_list_r1, rank_list_r2)

        if verbosity > 0:
            print(f"'# ### Kendall's Tau: {tau}, p-value: {pval_tau}")
            print(f"'# ### Spearman's Rank Correlation: {rho}, p-value: {pval_rho}")

        return {'tau': (tau, pval_tau), 'rho': (rho, pval_rho)}


    res_dfs_random = []
    res_dfs_actual = []
    for dataset in datasets:

        print(f'# ### {dataset} ### #')

        res_dir = os.path.join(base_res_dir, dataset_to_res_dir[dataset])
        ranked_tfs_pr = pd.read_csv(os.path.join(res_dir, 'ranked_tfs.csv'), index_col=0)['gene'].tolist()
        ranked_tfs_od = pd.read_csv(os.path.join(res_dir, 'outdeg_ranked_tfs.csv'), index_col=0)['gene'].tolist()
        tfs = list(sorted(ranked_tfs_pr))

        # Similarity between pr and od
        corrs_actual = compute_tau_rho(ranked_tfs_pr, ranked_tfs_od, verbosity=1)
        for metric, (val, pval) in corrs_actual.items():
            res_dfs_actual.append({
                "dataset": dataset,
                "ranking_metric": metric,
                "value": val,
                "pval": pval,
                "comparison": "PR_vs_OD"
            })

        # Similarity of pr and od to random
        rows = []
        for _ in tqdm(range(num_permutations), total=num_permutations):
            ranked_tfs_random = random.sample(tfs, k=len(tfs))

            for ranking, centrality_measure in [(ranked_tfs_pr, "pagerank"), (ranked_tfs_od, 'out_degree')]:
                ranking_similarities = compute_tau_rho(ranking, ranked_tfs_random)

                for metric, (val, pval) in ranking_similarities.items():

                    rows.append({
                        "dataset": dataset,
                        "centrality_measure": centrality_measure,
                        "ranking_metric": metric,
                        "value": val,
                        "pval": pval
                    })
        res_dfs_random.append(pd.DataFrame(rows))

    res_df_actual = pd.DataFrame(res_dfs_actual)
    res_df_random = pd.concat(res_dfs_random, ignore_index=True)

    for score in ranking_scores:

        fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=300)
        axd = fig.subplot_mosaic(
            '''
            ABC
            DEF
            '''
        )

        for cm, subplot_keys in zip(centrality_measures, [list('ABC'), list('DEF')]):

            colors = sns.color_palette('deep', 3)

            for dataset, subplot_key, color in zip(datasets, subplot_keys, colors):

                # Get the plot data for the histogram
                data_bool = (
                        (res_df_random['ranking_metric'] == score)
                        & (res_df_random['dataset'] == dataset)
                        & (res_df_random['centrality_measure'] == cm)
                )

                plot_df = res_df_random[data_bool].copy()

                rand_scores = plot_df['value'].to_numpy()

                pr_vs_od_score = (
                    res_df_actual
                    .query("dataset == @dataset and ranking_metric == @score")
                    ["value"].iloc[0]
                )

                empirical_p_value = ((pr_vs_od_score <= rand_scores).sum() + 1) / (rand_scores.shape[0] + 1)

                legend_label = (
                    f"PR–OD Similarity:"
                    f"\n{ranking_score_to_symbol[score]}: {np.round(pr_vs_od_score, 3)},"
                    f"\nEmp. P-value: {np.round(empirical_p_value, 4)}"
                )

                # Plot
                ax = axd[subplot_key]
                ax.hist(
                    rand_scores,
                    bins=50,
                    color=color,
                    edgecolor='grey',
                    linewidth=0.2,
                )

                ax.axvline(pr_vs_od_score, color='red', linestyle='-', linewidth=1.0, label=legend_label)

                ax.set_xlabel(ranking_score_to_symbol[score])
                ax.set_ylabel(fr'Count (total: $n$ = {num_permutations})')
                ax.set_title(f'{dataset} | {cm_to_cm_name[cm]}')

                ax.legend()

        # Annotate subplot mosaic tiles with labels
        for label, ax in axd.items():
            trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize=14, va='bottom', fontfamily='sans-serif', fontweight='bold')

        fig.savefig(os.path.join(save_p, f'ranking_similarities_{score}.png'), dpi=fig.dpi)


def main_additional_tf_target_scatter_plots():

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    import scanpy as sc

    from switchtfi.utils import load_grn_json, csr_to_numpy

    res_subdir = './results/05_revision/tf_target_scatter_plots'
    os.makedirs(res_subdir, exist_ok=True)

    datasets  = ['beta', 'alpha', 'ery']

    ds_to_ds_name = {'beta': 'Beta', 'alpha': 'Alpha', 'ery': 'Ery'}

    layer_key = 'magic_imputed'

    num_plot_per_row = 5
    num_plots_in_total = 30

    panel_layout = [
        [str(i) for i in range(start, min(start + num_plot_per_row, num_plots_in_total + 1))]
        for start in range(1, num_plots_in_total + 1, num_plot_per_row)
    ]

    fig = plt.figure(figsize=(11, 13), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        panel_layout,
    )

    i = 0
    for dataset in datasets:

        # Load the anndata object with the data and the precomputed GRN
        if dataset == 'ery':
            data = sc.read_h5ad('./data/anndata/erythrocytes.h5ad')
            grn = load_grn_json('./results/02_switchtfi/hematopoiesis/grn.json')
            cell_anno_key = 'prog_off'

        elif dataset == 'beta':
            data = sc.read_h5ad('./data/anndata/pre-endocrine_beta.h5ad')
            grn = load_grn_json('./results/02_switchtfi/endocrine/beta/grn.json')
            cell_anno_key = 'clusters'

        else:
            data = sc.read_h5ad('./data/anndata/pre-endocrine_alpha.h5ad')
            grn = load_grn_json('./results/02_switchtfi/endocrine/beta/grn.json')
            cell_anno_key = 'clusters'

        grn = grn.sort_values(by=['weight'], axis=0, ascending=False)
        grn = grn.reset_index(drop=True)

        labels = data.obs[cell_anno_key].to_numpy()

        if dataset == 'beta':
            # Keep the original muted orange and green
            label_to_color = {
                'Pre-endocrine': '#fdd49e',  # muted peach/orange
                'Beta': '#c7e9c0'  # muted light green
            }

        elif dataset == 'alpha':
            # Use muted coral and muted lavender — no overlap with beta
            label_to_color = {
                'Pre-endocrine': '#fcbba1',  # muted coral
                'Alpha': '#807dba'  # muted lavender-purple
            }

        else:  # ery
            # Use muted gold and soft plum — no overlap with beta or alpha
            label_to_color = {
                'prog': '#fddc84',  # muted gold
                'off': '#bcbddc'  # muted plum/lilac
            }

        if dataset == 'beta':
            # Keep original slightly muted orange and green
            label_to_color = {
                'Pre-endocrine': '#fdae6b',  # warmer orange-peach
                'Beta': '#a1d99b'  # fresher light green
            }

        elif dataset == 'alpha':
            label_to_color = {
                'Pre-endocrine': '#fb6a4a',  # more saturated coral
                'Alpha': '#6a51a3'  # deeper lavender-purple
            }

        else:  # ery
            label_to_color = {
                'prog': '#fddc00',  # saturated golden yellow
                'off': '#9e9ac8'  # soft but clear violet-gray
            }

        legend_handles = [
            plt.Line2D([], [], marker='o', color='w', label=label.capitalize(), markerfacecolor=color, markersize=6)
            for label, color in label_to_color.items()
        ]


        # Plot the highest weighted edges
        ax_keys_hi = panel_layout[i]
        for j, ax_key in enumerate(ax_keys_hi):

            tf = grn.loc[j, 'TF']
            target = grn.loc[j, 'target']
            weight = grn.loc[j, 'weight']
            threshold = grn.loc[j, 'threshold']
            pred_l = grn.loc[j, 'pred_l']
            pred_r = grn.loc[j, 'pred_r']

            x = csr_to_numpy(data[:, tf].layers[layer_key]).flatten()
            y = csr_to_numpy(data[:, target].layers[layer_key]).flatten()

            x_bool = (x != 0)
            y_bool = (y != 0)
            keep_bool = np.logical_and(x_bool, y_bool)

            x = x[keep_bool]
            y = y[keep_bool]
            labels_plot = labels[keep_bool]

            colors = [label_to_color[label] for label in labels_plot]

            ax = axd[ax_key]

            ax.scatter(
                x,
                y,
                c=colors,
                alpha=0.9,
                edgecolors='none',
                s=10,
            )

            min_x, max_x = x.min(), x.max()

            ax.plot([min_x, threshold], [pred_l, pred_l], color='red', zorder=2)
            ax.scatter([threshold], [pred_l], color='red', marker='o', zorder=3)
            ax.plot([threshold, max_x], [pred_r, pred_r], color='red', zorder=2)
            ax.scatter([threshold], [pred_r], color='red', marker='o', facecolor='white', zorder=3)
            ax.axvline(x=threshold, color='red', linestyle='--', zorder=1)
            ax.set_title(fr'{ds_to_ds_name[dataset]}, $w = {round(weight, 3)}$')
            ax.set_xlabel(tf)
            ax.set_ylabel(target)

            ax.legend(handles=legend_handles)

        i += 1

        # Plot the lowest weighted edges
        ax_keys_lo = panel_layout[i]
        for j, ax_key in enumerate(ax_keys_lo):

            tf = grn.loc[(grn.shape[0] - 1) - j, 'TF']
            target = grn.loc[(grn.shape[0] - 1) - j, 'target']
            weight = grn.loc[(grn.shape[0] - 1) - j, 'weight']
            threshold = grn.loc[(grn.shape[0] - 1) - j, 'threshold']
            pred_l = grn.loc[(grn.shape[0] - 1) - j, 'pred_l']
            pred_r = grn.loc[(grn.shape[0] - 1) - j, 'pred_r']

            x = csr_to_numpy(data[:, tf].layers[layer_key]).flatten()
            y = csr_to_numpy(data[:, target].layers[layer_key]).flatten()
            labels = data.obs[cell_anno_key].to_numpy()

            x_bool = (x != 0)
            y_bool = (y != 0)
            keep_bool = np.logical_and(x_bool, y_bool)

            x = x[keep_bool]
            y = y[keep_bool]
            labels_plot = labels[keep_bool]

            colors = [label_to_color[label] for label in labels_plot]

            ax = axd[ax_key]

            ax.scatter(
                x,
                y,
                c=colors,
                alpha=0.9,
                edgecolors='none',
                s=10,
            )

            min_x, max_x = x.min(), x.max()

            ax.plot([min_x, threshold], [pred_l, pred_l], color='red', zorder=2)
            ax.scatter([threshold], [pred_l], color='red', marker='o', zorder=3)
            ax.plot([threshold, max_x], [pred_r, pred_r], color='red', zorder=2)
            ax.scatter([threshold], [pred_r], color='red', marker='o', facecolor='white', zorder=3)
            ax.axvline(x=threshold, color='red', linestyle='--', zorder=1)
            ax.set_title(fr'{ds_to_ds_name[dataset]}, $w = {round(weight, 3)}$')
            ax.set_xlabel(tf)
            ax.set_ylabel(target)

            ax.legend(handles=legend_handles)

        i += 1

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 0.95, label, transform=ax.transAxes + trans,
                fontsize=12, va='bottom', fontfamily='sans-serif', fontweight='bold')

    fig.savefig(os.path.join(res_subdir, 'tf_target_scatter_plots.png'), dpi=fig.dpi)


def main_revised_regulon_plot():
    # ### Script for plotting Figure 5

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    import scanpy as sc

    from validation.plotting import plot_enrichr_results

    from switchtfi.plotting import plot_regulon

    save_p = './results/05_revision/revised_regulon_plot'
    os.makedirs(save_p, exist_ok=True)

    # Load transition GRN of beta-cell transition
    bgrn = pd.read_csv('./results/02_switchtfi/endocrine/beta/grn.csv', index_col=0)

    # Load expression data with gene trends
    bdata = sc.read_h5ad('./results/03_validation/anndata/trend_pre-endocrine_beta.h5ad')

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

    # Extract the Ybx1 regulon
    keep_bool = np.isin(bgrn['TF'].to_numpy(), ['Ybx1'])
    regulon = bgrn[keep_bool].copy().reset_index(drop=True)

    # Compute score and sort
    weights = regulon['weight'].to_numpy()
    pvals = regulon['pvals_wy'].to_numpy()
    pvals += np.finfo(np.float64).eps
    regulon['score'] = -np.log10(pvals) * weights
    regulon = regulon.sort_values('score', axis=0, ascending=False)

    # Get top-k targets
    top_k = 20
    top_k_targets = regulon['target'].tolist()[0:top_k]

    # ### Plot results for Beta dataset ### #
    fig = plt.figure(figsize=(13, 12), constrained_layout=True, dpi=300)
    mosaic = '''
        A
        B
        C
    '''


    axd = fig.subplot_mosaic(
        mosaic=mosaic,
        # gridspec_kw={'height_ratios': [5, 6], }
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
        grn=bgrn,
        tf='Ybx1',
        sort_by='score',
        top_k=top_k,
        title=None,
        edge_width=3.5,
        font_size=gene_fs,
        node_size=nodes_size,
        dpi=300,
        ax=axd['A']
    )

    # ### Plot the gene trends for the top 20 targets of Ybx1

    # Subset to the top k targets
    bdata_top_k_targets = bdata[:, top_k_targets].copy()

    # Get expression trends
    expression_trends = bdata_top_k_targets.varm['gam_gene_trends'].T

    # Scale gene expressions trends to [0, 1] for comparability
    mins = np.min(expression_trends, axis=0)
    maxs = np.max(expression_trends, axis=0)
    expression_trends_scaled = (expression_trends - mins) / (maxs - mins)

    # Sort genes such that the ones with the earliest peak come first
    argmaxs = np.argmax(expression_trends_scaled, axis=0)
    sorting_idx = np.argsort(argmaxs)
    expression_trends_scaled = expression_trends_scaled[:, sorting_idx]

    # Also sort the gene names of the top k TFs
    top_k_targets_sorted = np.array(top_k_targets)[sorting_idx].tolist()

    # Get pt-vector
    pt_vec = bdata_top_k_targets.obs['palantir_pseudotime'].to_numpy()

    ax = axd['B']
    trend_plot = ax.imshow(expression_trends_scaled.T, cmap='viridis', aspect='auto', interpolation='none')

    ax.set_xlabel('Pseudotime', fontsize=ax_fs)

    ax.set_frame_on(False)

    ax.set_yticks(ticks=np.arange(top_k), labels=top_k_targets_sorted, fontsize=12)

    x_tick_values = np.round(np.linspace(pt_vec.min(), pt_vec.max(), num=5), decimals=2)
    x_tick_positions = np.linspace(0, expression_trends_scaled.shape[0] - 1, num=5).astype(int)
    ax.set_xticks(x_tick_positions, x_tick_values)

    # ax.set_title("Expression Trends of Ybx1's Targets" , fontsize=ax_fs)

    cbar = plt.colorbar(trend_plot, ax=ax, location='right', pad=-0.1)
    cbar.ax.set_ylabel('Scaled Expression', fontsize=ax_fs)

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
        axs=axd['C']
    )

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=letter_fs, va='bottom', fontfamily='sans-serif', fontweight='bold')

    plt.savefig(os.path.join(save_p, 'hypothesis_gen_ybx1_revised.png'), dpi=fig.dpi)


def main_scalability_compute_edge_fraction():

    import os
    import glob

    import pandas as pd

    from statistics import median, mean


    def aggregate_grns(grns: list[pd.DataFrame]) -> pd.DataFrame:

        # Add auxiliary id column
        grns_id = []
        for i, grn in enumerate(grns):
            grn['grn_id'] = i
            grns_id.append(grn)

        # Concatenate GRNs
        stacked = pd.concat(grns_id, ignore_index=True)

        # Aggregate
        aggregated_grn = (
            stacked
            .groupby(['TF', 'target'], as_index=False)
            .agg(
                scenic_weight=('scenic_weight', 'mean'),
                support=('grn_id', 'nunique')
            )
        )

        return aggregated_grn


    res_p = './results/01_grn_inf'

    datasets = ['alpha', 'beta', 'ery']

    dataset_to_subdir = {
        'alpha': 'endocrine/alpha',
        'beta': 'endocrine/beta',
        'ery': 'hematopoiesis'
    }


    for dataset in datasets:

        grn_files = sorted(glob.glob(os.path.join(res_p, dataset_to_subdir[dataset]) + '/*_pruned_grn.csv'))

        grn_list = []
        for fn in grn_files:
            grn = pd.read_csv(fn, index_col=0)
            grn_list.append(grn)

        # Aggregate
        grn_agg = aggregate_grns(grns=grn_list)

        # Get total number of edges
        # total_edges = grn_agg.shape[0]
        num_edges_per_grn = [grn.shape[0] for grn in grn_list]

        median_num_edges = int(median(num_edges_per_grn))

        # Threshold
        thresholds = [3, 9, 18]
        num_edges_at_threshold = []
        for threshold in thresholds:
            grn_thres = grn_agg[grn_agg['support'] >= threshold]
            num_edges_at_threshold.append(grn_thres.shape[0])

        fracs = [n / median_num_edges for n in num_edges_at_threshold]

        print((
            f'# ### {dataset}:'
            f'\n# Median num edges per GRN: {median_num_edges}'
            f'\n# Thresholds: {thresholds}'
            f'\n# Totals: {num_edges_at_threshold}'
            f'\n# Fractions: {fracs}'
        ))

    # => small = 25 %, medium = 50 %, large = 75 %


def main_scalability_plot_figure():

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    import matplotlib.transforms as mtransforms

    test = False

    vary_num_cells_num_edges = [0.25, 0.5, 0.75]
    vary_num_edges_num_cells = [1000, 10000, 50000] if not test else [50, 100, 150]

    save_p = './results/05_revision/scalability/plots'
    os.makedirs(save_p, exist_ok=True)

    res_p = './results/05_revision/scalability/aggregated_results'

    res_df_n_cells_n_edges = pd.read_csv(os.path.join(res_p, 'aggregated_results_num_cells_num_edges.csv'), index_col=0)
    res_df_n_edges_n_cells = pd.read_csv(os.path.join(res_p, 'aggregated_results_num_edges_num_cells.csv'), index_col=0)

    # print(res_df_n_cells_n_edges)
    # print(res_df_n_edges_n_cells)

    method_to_method_name = {
        'cellrank': 'CellRank', 'splicejac': 'spliceJAC', 'drivaer': 'DrivAER',
        'switchtfi': 'SwitchTFI', 'grn_inf': 'SCENIC'
    }
    res_df_n_cells_n_edges['method'] = res_df_n_cells_n_edges['method'].map(method_to_method_name)
    res_df_n_edges_n_cells['method'] = res_df_n_edges_n_cells['method'].map(method_to_method_name)

    methods = ['CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI', 'SCENIC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    method_to_color = dict(zip(methods, colors))

    def seconds_to_human(x: float, dummy=None) -> str:
        x = int(x)
        if x < 60:
            return f'{x}s'
        elif x < 3600:
            minutes, seconds = divmod(x, 60)
            if seconds == 0:
                return f'{minutes}m'
            return f'{minutes}m {seconds}s'
        else:
            hours, remainder = divmod(x, 3600)
            minutes = int(round(remainder / 60))  # round to nearest minute
            if minutes == 0:
                return f'{hours}h'
            return f'{hours}h {minutes}m'

    def mem_formatter(x, dummy=None):
        if x >= 1024:
            out = x / 1024
            out = np.round(out, 1)
            if out % 1 == 0:
                out = int(out)
            return f'{out} GB'
        return f'{int(x)} MB'

    axtick_fontsize = 8
    plt.rcParams['xtick.labelsize'] = axtick_fontsize
    plt.rcParams['ytick.labelsize'] = axtick_fontsize

    hue_order = ['CellRank', 'spliceJAC', 'DrivAER', 'SwitchTFI']

    fig = plt.figure(figsize=(8, 9), constrained_layout=True, dpi=300)  # (8, 7)
    axd = fig.subplot_mosaic(
        '''
        ABX
        CDE
        FGH
        IJK
        LMN
        '''
    )

    # GRN inf plots
    for subplot_key, mode in zip(list('AB'), ['wall_time', 'mem_peak_cpu']):

        # Subset dataframe
        keep_bool = (res_df_n_cells_n_edges['method'] == 'SCENIC')
        res_df_sub = res_df_n_cells_n_edges[keep_bool].copy()

        ax = axd[subplot_key]
        sns.lineplot(
            data=res_df_sub,
            x='n_cells',
            y=mode,
            hue='method',
            palette=method_to_color,
            marker='o',
            ax=ax
        )

        # Style axis
        ax.set_xscale('log', base=10)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=10))

        ax.set_xlabel('Number of Cells')

        if mode == 'wall_time':
            ax.set_yscale('log', base=60)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_human))
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=60, subs=[], numticks=10))
            ax.yaxis.set_major_locator(ticker.LogLocator(base=60, subs=[0.5, 1.0], numticks=10))
            extra_major = [2 * 3600, 6 * 3600, 12 * 3600, 18 * 3600, 24 * 3600]
            ax.yaxis.set_major_locator(ticker.FixedLocator(
                list(ax.yaxis.get_major_locator()()) + extra_major
            ))
            ax.set_ylabel('Wall time')
        else:
            ymin = res_df_sub[mode].min()
            ymax = res_df_sub[mode].max()
            ax.set_yscale('log', base=2)

            locator = ticker.LogLocator(base=2.0, subs=[1.0], numticks=100)
            ticks = list(locator.tick_values(ymin, ymax))
            ticks = sorted(set(ticks + [ymin, ymax]))

            ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(mem_formatter))

            # ax.yaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=[1.0], numticks=10))
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(mem_formatter))

            ax.set_ylabel('Peak Memory')


    # Runtime plots
    for subplot_key, num_edges, title in zip(
            list('CDE'),
            vary_num_cells_num_edges,
            ['GRN size: small', 'GRN size: medium', 'GRN size: large']
    ):

        # Subset dataframe
        keep_bool = (
                (res_df_n_cells_n_edges['method'] != 'SCENIC')
                & (
                        (res_df_n_cells_n_edges['n_edges_frac'] == num_edges)
                        | res_df_n_cells_n_edges['n_edges_frac'].isna()
                )
        )
        res_df_sub = res_df_n_cells_n_edges[keep_bool].copy()

        ax = axd[subplot_key]
        sns.lineplot(
            data=res_df_sub,
            x='n_cells',
            y='wall_time',
            hue='method',
            hue_order=hue_order,
            palette=method_to_color,
            marker='o',
            ax=ax
        )

        # Style axis
        ax.set_xscale('log', base=10)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=10))

        ax.set_yscale('log', base=60)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_human))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=60, subs=[], numticks=10))

        # ax.yaxis.set_major_locator(ticker.LogLocator(base=60, subs=[0.5, 1.0], numticks=10))
        # extra_major = [2 * 3600, 6 * 3600, 12 * 3600, 18 * 3600, 24 * 3600]
        # ax.yaxis.set_major_locator(ticker.FixedLocator(list(ax.yaxis.get_major_locator()()) + extra_major))

        if subplot_key == 'C':
            major_ticks = [60, 30 * 60, 3600, 6 * 3600]
        elif subplot_key == 'D':
            major_ticks = [60, 30 * 60, 3600, 3 * 3600, 12 * 3600]
        else:  # E
            major_ticks = [60, 30 * 60, 3600, 6 * 3600, 18 * 3600]
        ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))

        ax.set_xlabel('Number of Cells')
        ax.set_ylabel('Wall time')

        ax.set_title(title)


    for subplot_key, num_cells, title in zip(
            list('FGH'),
            vary_num_edges_num_cells,
            ['No. of Cells: 1000', 'No. of Cells: 10000', 'No. of Cells: 50000']
    ):

        # Subset dataframe
        keep_bool = (
                (res_df_n_edges_n_cells['method'] != 'SCENIC')
                & (res_df_n_edges_n_cells['n_cells'] == num_cells)
        )
        res_df_sub = res_df_n_edges_n_cells[keep_bool].copy()

        ax = axd[subplot_key]
        sns.lineplot(
            data=res_df_sub,
            x='n_edges',
            y='wall_time',
            hue='method',
            hue_order=hue_order,
            palette=method_to_color,
            marker='o',
            ax=ax
        )

        # Style axis
        ax.set_xscale('log', base=10)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=10))

        ax.set_yscale('log', base=60)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_human))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=60, subs=[], numticks=10))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=60, subs=[0.5, 1.0], numticks=10))
        extra_major = [2*3600, 6*3600, 12*3600, 18*3600, 24*3600]
        ax.yaxis.set_major_locator(ticker.FixedLocator(
            list(ax.yaxis.get_major_locator()()) + extra_major
        ))

        ax.set_xlabel('Number of Edges')
        ax.set_ylabel('Wall time')

        ax.set_title(title)


    # Memory plots
    for subplot_key, num_edges, title in zip(
            list('IJK'),
            vary_num_cells_num_edges,
            ['GRN size: small', 'GRN size: medium', 'GRN size: large']
    ):
        # Subset dataframe
        keep_bool = (
                (res_df_n_cells_n_edges['method'] != 'SCENIC')
                & (
                        (res_df_n_cells_n_edges['n_edges_frac'] == num_edges)
                        | res_df_n_cells_n_edges['n_edges_frac'].isna()
                )
        )
        res_df_sub = res_df_n_cells_n_edges[keep_bool].copy()

        ax = axd[subplot_key]
        sns.lineplot(
            data=res_df_sub,
            x='n_cells',
            y='mem_peak_cpu',
            hue='method',
            hue_order=hue_order,
            palette=method_to_color,
            marker='o',
            ax=ax
        )

        ax.set_xscale('log', base=10)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=10))
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=[1.0], numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(mem_formatter))
        ax.set_xlabel('Number of Cells')
        ax.set_ylabel('Peak Memory')

        ax.set_title(title)

    for subplot_key, num_cells, title in zip(
            list('LMN'),
            vary_num_edges_num_cells,
            ['No. of Cells: 1000', 'No. of Cells: 10000', 'No. of Cells: 50000']
    ):
        # Subset dataframe
        keep_bool = (
            (res_df_n_edges_n_cells['method'] != 'SCENIC')
            & (res_df_n_edges_n_cells['n_cells'] == num_cells)
        )
        res_df_sub = res_df_n_edges_n_cells[keep_bool].copy()

        ax = axd[subplot_key]
        sns.lineplot(
            data=res_df_sub,
            x='n_edges',
            y='mem_peak_cpu',
            hue='method',
            hue_order=hue_order,
            palette=method_to_color,
            marker='o',
            ax=ax
        )

        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=[1.0], numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(mem_formatter))

        ax.set_xlabel('Number of Edges')
        ax.set_ylabel('Peak Memory')

        ax.set_title(title)

    # for key, ax in axd.items():
    #     ax.legend_.set_title(None)

    # Add legend
    for key, ax in axd.items():
        if key != 'X':
            ax.get_legend().remove()
        ax.grid(True, which='both', axis='y', color='lightgrey', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

    handles0, labels0 = axd['A'].get_legend_handles_labels()
    handles1, labels1 = axd['D'].get_legend_handles_labels()
    handles = handles0 + handles1
    labels = labels0 + labels1
    axd['X'].axis('off')
    axd['X'].legend(
        handles, labels,
        loc='center',
        ncol=1,
        frameon=True
    )

    # y_label_panels = {'A', 'B', 'C', 'F', 'I', 'L'}

    for label, ax in axd.items():

        # if label not in y_label_panels:
        #     ax.set_ylabel('')

        if label != 'X':
            trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
            ax.text(0.0, 0.95, label, transform=ax.transAxes + trans,
                    fontsize=12, va='bottom', fontfamily='sans-serif', fontweight='bold')


    plt.savefig(os.path.join(save_p, 'scalability.png'))


def main_tcell_data_exploration():

    import os

    import matplotlib.pyplot as plt
    import scanpy as sc

    data_dir = './results/05_revision/tcell/data'
    plot_dir = './results/05_revision/tcell/data_exploration'
    os.makedirs(plot_dir, exist_ok=True)

    full_dataset_filename = 'ga_an0602_10x_smarta_doc_arm_liver_spleen_d10_d28_mgd_ts_filtered_int_inf_tp_rp_convert.h5ad'

    load_full_dataset = True

    # ### Load full dataset and subset (tissue spleen, day 10)
    if load_full_dataset:

        tdata = sc.read_h5ad(os.path.join(data_dir, full_dataset_filename))

        tdata.raw = None  # Delete the raw to avoid error when saving

        # Relabel cluster labels from 0-9 to 1-10
        new_labels = [int(label + 1) for label in tdata.obs['cluster']]
        tdata.obs['cluster'] = new_labels

        print(f'# ### AnnData:\n{tdata}')
        print(f'# ### Data matrix:\n{tdata.X}')
        print(f'# ### Clusters:\n{tdata.obs["cluster"]}')
        print(f'# ### Unique cluster labels:\n{set(tdata.obs["cluster"].tolist())}')

        # ### Subset to the tissue and time point:
        # Tissue: spleen = 0, liver = 1
        # Time point: 0 = day 10, 1 = day 28

        tissue_name_to_label = {'spleen': 0, 'liver': 1}
        time_name_to_label = {'d10': 0, 'd28': 1}
        infection_label_to_name = {0: 'chronic (doc)', 1: 'acute (arm)'}

        for tissue in ['spleen', 'liver']:
            for time in ['d10', 'd28']:

                print(f'\n# ### Tissue: {tissue}, Time point: {time}')

                # Subset to tissue and time; save AnnData
                keep_bool_tissue = tdata.obs['tissue'] == tissue_name_to_label[tissue]
                keep_bool_time = tdata.obs['time'] == time_name_to_label[time]
                keep_bool = keep_bool_tissue & keep_bool_time

                tdata_subset = tdata[keep_bool, :].copy()
                tdata_subset.write(os.path.join(plot_dir, f'tdata_{tissue}_{time}.h5ad'))
                tdata_tissue_time = tdata[keep_bool_tissue & keep_bool_time, :].copy()

                # Subset w.r.t. infection status and visualize the population sizes:
                # - docile = chronic = 0
                # - armstrong = acute = 1

                # keep_bool_doc = (tdata.obs['infection'] == 0).to_numpy()
                # tdata_doc = tdata[keep_bool_doc, :].copy()

                # keep_bool_arm = (tdata.obs['infection'] == 1).to_numpy()
                # tdata_arm = tdata[keep_bool_arm, :].copy()

                # Visualize population sizes of clusters 3, 4, 5 depending on infection status
                count_df = tdata_tissue_time.obs.groupby(['cluster', 'infection']).size().unstack(fill_value=0)
                count_df.rename(columns=infection_label_to_name, inplace=True)
                count_df.to_csv(os.path.join(plot_dir, f'population_sizes_{tissue}_{time}.csv'))
                print(f'# Number of cells per cluster:\n{count_df}')

                fig, ax = plt.subplots(dpi=300)
                count_df.plot(kind='bar', ax=ax)
                ax.set_ylabel('# cells')
                ax.set_title(f'Tissue: {tissue}, Time: {time}')
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height,
                                f'{int(height)}',
                                ha='center',
                                va='bottom',
                                fontsize=6
                            )
                fig.savefig(os.path.join(plot_dir, f'population_sizes_{tissue}_{time}.png'), dpi=300)
                plt.close(fig)

                # Normalize the counts by the total cells in the group (chronic vs. acute) across all clusters
                norm_df = count_df.div(count_df.sum(axis=0), axis=1)
                print(f'# Normalized number of cells per cluster:\n{norm_df}')
                norm_df.to_csv(os.path.join(plot_dir, f'relative_population_sizes_{tissue}_{time}.csv'))

                fig, ax = plt.subplots(dpi=300)
                norm_df.plot(kind='bar', ax=ax)
                ax.set_ylabel('# cells (normalized)')
                ax.set_title(f'Tissue: {tissue}, Time: {time}')
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height,
                                f'{round(height, 4)}',
                                ha='center',
                                va='bottom',
                                fontsize=6
                            )
                fig.savefig(
                    os.path.join(plot_dir, f'relative_population_sizes_{tissue}_{time}.png'),
                    dpi=300
                )
                plt.close(fig)

    # Analyze cluster structure of clusters of interest Clusters:
    # - 3 = Th1 progenitors
    # - 4 = Th1 intermediate state
    # - 5 = Th1 terminally differentiated

    prog_off_labels = {3: 'prog', 4: 'prog', 5: 'off'}

    for tissue in ['spleen', 'liver']:
        for time in ['d10', 'd28']:

            # Load data
            tdata = sc.read_h5ad(os.path.join(plot_dir, f'tdata_{tissue}_{time}.h5ad'))

            # Subset to populations of interest
            keep_bool_cluster = tdata.obs['cluster'].isin(list(prog_off_labels.keys()))
            tdata = tdata[keep_bool_cluster, :].copy()

            # Add progenitor, offspring annotations
            cluster_labels = tdata.obs['cluster'].tolist()
            prog_off_anno = [prog_off_labels[cluster] for cluster in cluster_labels]
            tdata.obs['prog_off'] = prog_off_anno

            fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

            for ax, label_key in zip(axs, ['cluster', 'prog_off']):

                labels = tdata.obs[label_key]
                unique_labels = labels.unique()
                cmap=plt.get_cmap('Set2')
                label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
                colors = labels.map(label_to_color).tolist()

                ax.scatter(
                    x=tdata.obsm['X_iumap'][:, 0],
                    y=tdata.obsm['X_iumap'][:, 1],
                    c=colors,
                    s=10,
                    edgecolors='grey',
                    linewidths=0.5,
                )

                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_title(f'Tissue: {tissue}, Time: {time}')

                for label in unique_labels:
                    ax.scatter([], [], color=label_to_color[label], label=label)
                ax.legend(title=label_key)

            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'umap_{tissue}_{time}.png'), dpi=300)
            plt.close(fig)


def main_tcell_data_processing():

    # ### Note:
    # Based on analyzing the relative cell population sizes, it seems like clusters 3, 4 are 5 is the offspring.
    # -> For d10, spleen and liver, cells do not reach cluster 4 in the chronic case.
    # => Run analysis with prog = 3,4 and proj = 3 for d10, spleen and liver

    import os

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scanpy.external as scex

    from scipy.stats import median_abs_deviation

    sp = False
    sp_str = '_sp'

    data_dir = f'./results/05_revision/tcell{sp_str if sp else ""}/data'

    tissues = ['spleen', 'liver']
    time = 'd10'
    infections = ['chronic', 'acute']
    clusters = [[1, 3], [3, 4, 5], [3, 5]]

    cluster_ids_to_prog_off_labels = [
        {1: 'prog', 3: 'off'},
        {3: 'prog', 4: 'prog', 5: 'off'},
        {3: 'prog', 4: 'prog', 5: 'off'}
    ]

    time_name_to_label = {'d10': 0, 'd28': 1}
    tissue_name_to_label = {'spleen': 0, 'liver': 1}
    infection_name_to_label = {'chronic': 0, 'acute': 1}

    # Load the full dataset
    full_dataset_filename = 'ga_an0602_10x_smarta_doc_arm_liver_spleen_d10_d28_mgd_ts_filtered_int_inf_tp_rp_convert.h5ad'
    tdata = sc.read_h5ad(os.path.join(data_dir, full_dataset_filename))

    if sp:
        tdata.X = tdata.raw.X.copy()

    # Delete the raw to avoid error when saving
    tdata.raw = None

    # Relabel cluster labels from 0-9 to 1-10
    new_labels = [int(label + 1) for label in tdata.obs['cluster']]
    tdata.obs['cluster'] = new_labels

    # Convert sparse to dense
    tdata.X = tdata.X.toarray()

    for tissue in tissues:
        for infection in infections:
            for cluster_keys, cid_to_pol in zip(clusters, cluster_ids_to_prog_off_labels):

                # Subset to populations of interest
                keep_bool_time = tdata.obs['time'] == time_name_to_label[time]
                keep_bool_tissue = tdata.obs['tissue'] == tissue_name_to_label[tissue]
                keep_bool_infection = tdata.obs['infection'] == infection_name_to_label[infection]
                keep_bool_cluster = tdata.obs['cluster'].isin(cluster_keys)

                keep_bool = keep_bool_time & keep_bool_tissue & keep_bool_infection & keep_bool_cluster

                tdata_subset = tdata[keep_bool, :].copy()

                # Add progenitor-offspring annotations
                cluster_labels = tdata_subset.obs['cluster'].tolist()
                prog_off_anno = [cid_to_pol[cluster] for cluster in cluster_labels]
                tdata_subset.obs['prog_off'] = prog_off_anno

                if sp:
                    # --- Filter low quality reads
                    pct_counts_mt_threshold = 8.0

                    # Define outliers
                    def is_outlier(
                            adata: sc.AnnData,
                            obs_key_qc_metric: str,
                            nmads: int = 5
                    ) -> pd.Series:

                        m = adata.obs[obs_key_qc_metric]

                        median = np.median(m)
                        mad = median_abs_deviation(m)

                        lower_bound = median - nmads * mad
                        upper_bound = median + nmads * mad

                        outlier = (m < lower_bound) | (m > upper_bound)

                        return outlier

                    # Calculate QC-metrics
                    tdata_subset.var['mt'] = tdata_subset.var_names.str.startswith('mt-')
                    sc.pp.calculate_qc_metrics(
                        tdata_subset,
                        qc_vars=['mt'],
                        inplace=True,
                        percent_top=[20],
                        log1p=True
                    )

                    # Annotate cells that are outliers w.r.t. QC-metrics
                    tdata_subset.obs['outlier'] = (
                            is_outlier(tdata_subset, obs_key_qc_metric='log1p_total_counts', nmads=5)
                            | is_outlier(tdata_subset, obs_key_qc_metric='log1p_n_genes_by_counts', nmads=5)
                            | is_outlier(tdata_subset, obs_key_qc_metric='pct_counts_in_top_20_genes', nmads=5)
                    )

                    tdata_subset.obs['mt_outlier'] = (
                            is_outlier(tdata_subset, obs_key_qc_metric='pct_counts_mt', nmads=3)
                            | (tdata_subset.obs['pct_counts_mt'] > pct_counts_mt_threshold)
                    )

                    # Filter adata based on the identified outliers
                    n_cells_before = tdata_subset.n_obs
                    tdata_subset = tdata_subset[(~tdata_subset.obs.outlier) & (~tdata_subset.obs.mt_outlier)].copy()

                    n_cells_after = tdata_subset.n_obs
                    print(f'# Number of cells before filtering: {n_cells_before}')
                    print(f'# Number of cells after filtering: {n_cells_after}')
                    print(f'# Number of cells removed due to low quality: {n_cells_before - n_cells_after}')

                    # --- Filter uninformative genes
                    min_cells = 10
                    sc.pp.filter_genes(tdata_subset, min_cells=min_cells)

                    # --- Normalize and transform
                    tdata_subset.layers['raw_counts'] = tdata_subset.X.copy()
                    sc.pp.normalize_total(tdata_subset)
                    sc.pp.log1p(tdata_subset)

                else:
                    # Basic count based QC on the gene level
                    sc.pp.filter_genes(tdata_subset, min_cells=20)

                # MAGIC imputation
                tdata_dummy = tdata_subset.copy()
                scex.pp.magic(
                    adata=tdata_dummy,
                    name_list='all_genes',
                    knn=5,
                    decay=1,
                    knn_max=None,
                    t=1,
                    n_pca=100,
                    solver='exact',
                    knn_dist='euclidean',
                    random_state=42,
                    n_jobs=None,
                    verbose=True,
                    copy=None,
                )
                tdata_subset.layers['magic_imputed'] = tdata_dummy.X.copy()

                # Scale to unit variance for GRN inference
                tdata_subset.layers['unit_variance'] = sc.pp.scale(tdata_subset.X.copy(), zero_center=False, copy=True)

                # Save
                cluster_str = ''.join(str(i) for i in cluster_keys)
                id_str = f'{tissue}_{time}_{infection}_{cluster_str}'
                tdata_subset.write(os.path.join(data_dir, f'tdata_{id_str}.h5ad'))


def main_tcell_grn_inference():

    import os
    import glob

    import pandas as pd
    import matplotlib.pyplot as plt
    import scanpy as sc

    from typing import List
    from grn_inf.grn_inference import pyscenic_pipeline

    sp = True
    sp_str = '_sp'

    # Define paths to files where TFs are stored
    tf_file = './data/tf/mus_musculus/allTFs_mm.txt'
    # Define paths to auxiliary annotation files needed for GRN inference with Scenic
    db_file = (
        './data/scenic_aux_data/databases/mouse/mm10/mc_v10_clust/'
        'mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'
    )
    anno_file = './data/scenic_aux_data/motif2tf_annotations/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl'


    tissues = ['spleen', 'liver']
    time = 'd10'
    infections = ['acute', 'chronic']
    clusters = ['13', ]  # ['345', '35']  # Todo
    n_grns = 18
    edge_count_threshold = 9

    with_tox = False
    tox_str = 'wtox' if with_tox else 'notox'
    tox_genes = ['Tox', 'Tox1', 'Tox2', 'Tox3', 'Tox4']

    inference = True

    data_p = f'./results/05_revision/tcell{sp_str if sp else ""}/data'
    base_res_p = f'./results/05_revision/tcell{sp_str if sp else ""}/grn/{tox_str}'
    os.makedirs(base_res_p, exist_ok=True)

    for infection in infections:
        for tissue in tissues:
            for cluster_keys in clusters:

                # Load data
                id_str = f'{tissue}_{time}_{infection}_{cluster_keys}'
                filepath = os.path.join(data_p, f'tdata_{id_str}.h5ad')
                tdata = sc.read_h5ad(filepath)

                res_p = os.path.join(base_res_p, id_str)
                os.makedirs(res_p, exist_ok=True)

                if inference:
                    for i in range(n_grns):
                        print(f'# ### GRN inference, {id_str}, iteration {i}/{n_grns}')

                        pyscenic_pipeline(
                            adata=tdata,
                            additional_tfs=tox_genes if with_tox else None,
                            layer_key='unit_variance',
                            tf_file=tf_file,
                            result_folder=res_p,
                            database_path=db_file,
                            motif_annotations_path=anno_file,
                            grn_inf_method='grnboost2',
                            fn_prefix=f'{i:02d}_',
                            verbosity=1,
                            plot=False
                        )

                # ### Combine the 18 individual Scenic GRNs into one
                # Edges that occur in >= n_occurrence_threshold individual GRNs are retained
                def aggregate_grns(grns: List[pd.DataFrame]) -> pd.DataFrame:

                    # Add auxiliary id column
                    grns_id = []
                    for i, grn in enumerate(grns):
                        grn['grn_id'] = i
                        grns_id.append(grn)

                    # Concatenate GRNs
                    stacked = pd.concat(grns_id, ignore_index=True)

                    # Aggregate
                    if 'scenic_weight' in stacked.columns:
                        aggregated_grn = (
                            stacked
                            .groupby(['TF', 'target'], as_index=False)
                            .agg(
                                scenic_weight=('scenic_weight', 'mean'),
                                importance=('importance', 'mean'),
                                support=('grn_id', 'nunique')
                            )
                        )
                    else:
                        aggregated_grn = (
                            stacked
                            .groupby(['TF', 'target'], as_index=False)
                            .agg(
                                importance=('importance', 'mean'),
                                support=('grn_id', 'nunique')
                            )
                        )

                    return aggregated_grn

                print('### Combining GRNs ...')
                csv_files_grnboost2 = sorted(glob.glob(res_p + '/*_basic_grn.csv'))
                csv_files_scenic = sorted(glob.glob(res_p + '/*_pruned_grn.csv'))
                grn_list_grnboost2 = []
                grn_list_scenic = []
                for fn_grnboost2, fn_scenic in zip(csv_files_grnboost2, csv_files_scenic):

                    grn_grnboost2 = pd.read_csv(fn_grnboost2, sep='\t')
                    grn_scenic = pd.read_csv(fn_scenic, index_col=0)

                    # Add importance column to scenic GRN
                    grn_scenic = grn_scenic.merge(
                        grn_grnboost2,
                        on=['TF', 'target'],
                        how='left'  # Keep all rows of grn_scenic
                    )

                    grn_list_grnboost2.append(grn_grnboost2)
                    grn_list_scenic.append(grn_scenic)

                # Aggregate
                aggregated_grn_grnboost2 = aggregate_grns(grns=grn_list_grnboost2)
                aggregated_grn_grnboost2.to_csv(os.path.join(res_p, 'grnboost2_aggregated_grn.csv'))

                aggregated_grn_scenic = aggregate_grns(grns=grn_list_scenic)
                aggregated_grn_scenic.to_csv(os.path.join(res_p, 'scenic_aggregated_grn.csv'))

                # Threshold
                core_grn_grnboost2 = aggregated_grn_grnboost2[
                    aggregated_grn_grnboost2['support'] >= edge_count_threshold
                    ].copy()
                fn_grnboost2_grn = f'edge_count_threshold_{edge_count_threshold}_grnboost2_aggregated_grn.csv'
                core_grn_grnboost2.to_csv(os.path.join(res_p, fn_grnboost2_grn))

                core_grn_scenic = aggregated_grn_scenic[
                    aggregated_grn_scenic['support'] >= edge_count_threshold
                ].copy()
                fn_scenic_grn = f'edge_count_threshold_{edge_count_threshold}_scenic_aggregated_grn.csv'
                core_grn_scenic.to_csv(os.path.join(res_p, fn_scenic_grn))

                # Plot histograms
                fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
                colors = ['lightblue', 'lightcoral']
                for i, col in enumerate(['importance', 'support']):
                    ax = axs[i]
                    ax.hist(aggregated_grn_grnboost2[col], bins=100, color=colors[i], edgecolor='grey')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    ax.set_title(col.capitalize())
                fig.tight_layout()
                fig.savefig(os.path.join(res_p, 'histograms_grnboost2.png'), dpi=fig.dpi)
                plt.close(fig)

                fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=300)
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                for i, col in enumerate(['importance', 'scenic_weight', 'support']):
                    ax = axs[i]
                    ax.hist(aggregated_grn_scenic[col], bins=100, color=colors[i], edgecolor='grey')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    ax.set_title(col.capitalize())
                fig.tight_layout()
                fig.savefig(os.path.join(res_p, 'histograms_scenic.png'), dpi=fig.dpi)
                plt.close(fig)


def main_tcell_grn_exploration():
    import os

    import numpy as np
    import pandas as pd

    sp = True
    sp_str = '_sp'

    tissues = ['spleen', 'liver']
    time = 'd10'
    infection = 'acute'
    clusters = ['345', '35']

    with_tox = True
    tox_str = 'wtox' if with_tox else 'notox'
    tox_genes = ['Tox', 'Tox2', 'Tox3', 'Tox4']

    grn_p = f'./results/05_revision/tcell{sp_str if sp else ""}/grn/{tox_str}'

    method_to_weight_key = {'grnboost2': 'importance', 'scenic': 'scenic_weight'}

    for tissue in tissues:
        for cluster_keys in clusters:

            print(f'# ### {tissue}, {cluster_keys}, {infection} ### #')

            # Load the precomputed GRNs
            id_str = f'{tissue}_{time}_{infection}_{cluster_keys}'

            for method in ['grnboost2', 'scenic']:

                print(f'# ### {method}')

                grn_path = os.path.join(grn_p, id_str, f'edge_count_threshold_9_{method}_aggregated_grn.csv')
                grn = pd.read_csv(grn_path, index_col=0).reset_index()
                grn = grn.sort_values(by=method_to_weight_key[method], ascending=False).reset_index(drop=True)

                n_edges = grn.shape[0]
                n_vertices = np.unique(grn[['TF', 'target']].to_numpy()).shape[0]

                tfs = set(grn['TF'].tolist())
                targets = set(grn['target'].tolist())

                tf_target_intersection = tfs.intersection(targets)

                print(f'# num nodes: {n_vertices}')
                print(f'# num edges: {n_edges}')
                print(f'# num tfs: {len(tfs)}')
                print(f'# num targets: {len(targets)}')
                print(f'# intersection size: {len(tf_target_intersection)}')

                # print(f'# GRN:\n{grn}')

                if with_tox:
                    tox_tf_counts = grn['TF'].value_counts().reindex(tox_genes, fill_value=0)
                    tox_target_counts = grn['target'].value_counts().reindex(tox_genes, fill_value=0)
                    print(f'# Tox TFs:\n{tox_tf_counts}')
                    print(f'# Tox targets:\n{tox_target_counts}')


def main_tcell_switchtfi():

    import os

    import pandas as pd
    import scanpy as sc

    from switchtfi import fit_model, rank_tfs

    sp = True
    sp_str = '_sp'

    tissues = ['spleen', 'liver']
    time = 'd10'
    infection = 'acute'
    clusters = ['13', ]  # ['345', '35'] Todo

    with_tox = False
    tox_str = 'wtox' if with_tox else 'notox'

    grn_inf_method = 'scenic'  # grnboost2, scenic
    support_threshold = 9 if grn_inf_method == 'scenic' else 18

    data_p = f'./results/05_revision/tcell{sp_str if sp else ""}/data'
    grn_p = f'./results/05_revision/tcell{sp_str if sp else ""}/grn/{tox_str}'
    base_res_p = f'./results/05_revision/tcell{sp_str if sp else ""}/switchtfi/{tox_str}'

    for tissue in tissues:
        for cluster_keys in clusters:

            id_str = f'{tissue}_{time}_{infection}_{cluster_keys}'

            # Load data
            filepath = os.path.join(data_p, f'tdata_{id_str}.h5ad')
            tdata = sc.read_h5ad(filepath)

            # Load the precomputed GRN
            grn_path = os.path.join(grn_p, id_str, f'{grn_inf_method}_aggregated_grn.csv')
            grn = pd.read_csv(grn_path, index_col=0)

            # Subset based on support
            grn = grn[grn['support'] >= support_threshold].copy()

            # Create results directory
            res_p = os.path.join(base_res_p, id_str, grn_inf_method)
            os.makedirs(res_p, exist_ok=True)

            transition_grn, ranked_tfs_pagerank = fit_model(
                adata=tdata,
                grn=grn,
                layer_key='magic_imputed',
                clustering_obs_key='prog_off',
                result_folder=res_p,
                verbosity=2,
                save_intermediate=True
            )

            ranked_tfs_outdegree = rank_tfs(
                grn=transition_grn,
                centrality_measure='out_degree',
                reverse=False,
                weight_key='score',
                result_folder=res_p,
                fn_prefix='outdegree_'
            )


def main_tcell_de_analysis():
    import os
    import anndata2ri

    import pandas as pd
    import scanpy as sc
    import rpy2.robjects as ro

    from typing import Tuple, Any
    from rpy2.robjects import pandas2ri, default_converter
    from rpy2.robjects.conversion import localconverter
    from scipy.sparse import issparse

    # Build a converter that knows about AnnData and Pandas
    converter = default_converter + anndata2ri.converter + pandas2ri.converter

    def zinbwave_edger(
            adata, group_key: str,
            contrast: Tuple[Any, Any],
            epsilon: float = 1e12,
            k: int = 0,
            n_cores: int = 1
    ) -> pd.DataFrame:
        """
        Run zinbwave with observational weights and test groups with edgeR.

        Parameters
        ----------
        adata : AnnData
            AnnData object with raw counts in .X and metadata in .obs
        group_key : str
            Column in adata.obs defining the groups (factor)
        contrast : Tuple[Any, Any]
            Contrast for test (contrast[0] vs contrast[1] = test group vs reference group)
        epsilon : float
            Penalty parameter for zinbwave.
            (https://www.bioconductor.org/packages/release/bioc/vignettes/zinbwave/inst/doc/intro.html:
            "Our evaluations have shown that a value of epsilon=1e12 gives good performance across a range of datasets,
            although this number is still arbitrary.")
        k : int
            Number of latent factors for zinbwave.
            (https://doi.org/10.1186/s13059-018-1406-4:
            To compute the weights for differential expression analysis, ZINB-WaVE was fitted with intercept and
            cell-type covariate in X, V=1 J , K=0 for W, common dispersion, and ε=1012. (Fig. 6)
            The observational weights were computed with the number of unknown covariates K=0, i.e.,
            no latent variables were inferred.)
        n_cores : int
            Number of cores to use, -1 uses all available

        Returns
        -------
        Pandas DataFrame with DE results (logFC, logCPM, LR, PValue, FDR)
        """

        # Set number of cores to use
        n_available_cores = len(os.sched_getaffinity(0))
        if n_cores == -1:
            n_cores = n_available_cores
        else:
            n_cores = min(n_cores, n_available_cores)

        # Minimal AnnData for R
        x = adata.X.copy()
        if issparse(x):
            x = x.toarray()
        adata_input = sc.AnnData(x)
        adata_input.var_names = adata.var_names.copy()
        adata_input.obs_names = adata.obs_names.copy()
        adata_input.obs[group_key] = adata.obs[group_key].copy()

        # Add annotations such that alphabetical order reflects contrast
        # group_anno = [f'0_{lbl}' if lbl == contrast[0] else f'1_{lbl}' for lbl in adata.obs[group_key]]
        # adata_input.obs[group_key] = group_anno

        # Put AnnData into R as a SingleCellExperiment
        with localconverter(converter):
            ro.globalenv['sce'] = adata_input
        ro.globalenv['group_key'] = group_key
        ro.globalenv['zinb_epsilon'] = epsilon
        ro.globalenv['zinb_K'] = k
        ro.globalenv['n_cores'] = n_cores
        ro.globalenv['ref_level'] = str(contrast[1])  # baseline
        ro.globalenv['test_level'] = str(contrast[0])  # test

        ro.r('''
        suppressPackageStartupMessages({
            library(SingleCellExperiment)
            library(zinbwave)
            library(edgeR)
            library(BiocParallel)
        })

        # Ensure group column is factor
        colData(sce)$group <- as.factor(colData(sce)[[group_key]])
        colData(sce)$group <- relevel(colData(sce)$group, ref=ref_level)

        message("# ### Running zinbwave with ", n_cores, " cores ...")

        # Run zinbwave with weights
        sce_zinb <- zinbwave(
            sce, 
            K=zinb_K, 
            epsilon=zinb_epsilon, 
            observationalWeights=TRUE,
            BPPARAM=MulticoreParam(n_cores)
        )

        print("# ### edgeR ..." )

        # edgeR pipeline with weights
        weights <- assay(sce_zinb, "weights")
        dge <- DGEList(assay(sce_zinb))
        dge <- calcNormFactors(dge)  # Normalization

        design <- model.matrix(~group, data=colData(sce_zinb))
        dge$weights <- weights
        dge <- estimateDisp(dge, design)  # Estimate dispersion parameters
        fit <- glmFit(dge, design)  # Fit generalized linear model

        lrt <- glmWeightedF(fit, coef=2)  # Likelihood ration test
        res <- topTags(lrt, n=Inf)$table
        ''')

        # Convert R DataFrame back to pandas
        with localconverter(converter):
            de_res_df = ro.conversion.rpy2py(ro.r('res'))

        return de_res_df

    sp = True
    sp_str = '_sp'

    # Load the full data set
    data_p = f'./results/05_revision/tcell{sp_str if sp else ""}/data/'
    fn_all_data = 'ga_an0602_10x_smarta_doc_arm_liver_spleen_d10_d28_mgd_ts_filtered_int_inf_tp_rp_convert.h5ad'
    tdata = sc.read_h5ad(os.path.join(data_p, fn_all_data))

    # Relabel cluster labels from 0-9 to 1-10
    new_labels = [int(label + 1) for label in tdata.obs['cluster']]
    tdata.obs['cluster'] = new_labels

    # Set raw counts as main data matrix
    tdata.X = tdata.raw.X.copy()
    tdata.raw = None

    # Subset to TFs
    tf_file = './data/tf/mus_musculus/allTFs_mm.txt'
    with open(tf_file) as file:
        all_tfs = [line.strip() for line in file.readlines()]
    tf_bool = tdata.var_names.isin(all_tfs)
    tdata = tdata[:, tf_bool].copy()

    tissues = ['spleen', 'liver']
    clusters = ['13', ]  # ['345', '35']  # Todo
    time = 'd10'

    time_name_to_label = {'d10': 0, 'd28': 1}
    tissue_name_to_label = {'spleen': 0, 'liver': 1}
    infection_name_to_label = {'chronic': 0, 'acute': 1}
    cluster_key_to_progenitor_labels = {'13': [1, ], '345': [3, 4], '35': [3, ]}
    cluster_key_to_cluster_labels = {'13': [1, 3], '345': [3, 4, 5], '35': [3, 5]}
    cluster_key_to_cid_to_pol = {
        '13': {1: 'prog', 3: 'off'},
        '35': {3: 'prog', 5: 'off'},
        '345': {3: 'prog', 4: 'prog', 5: 'off'}
    }

    # Add semantic annotations
    infection_label_to_name = {0: 'chronic', 1: 'acute'}
    tdata.obs['infection_name'] = [infection_label_to_name[lbl] for lbl in tdata.obs['infection']]

    # Subset data to time point d10
    keep_bool_time = (tdata.obs['time'] == time_name_to_label[time])
    tdata = tdata[keep_bool_time, :].copy()

    save_path_base = f'./results/05_revision/tcell{sp_str if sp else ""}/de_analysis'
    os.makedirs(save_path_base, exist_ok=True)

    # ### --- DE analysis --- ### #
    # ### Chronic vs acute in progenitors
    contrasts = ['cva', 'avc']
    for tissue in tissues:
        for cluster_key in clusters:
            for contrast in contrasts:
                print(f'# ###### {tissue}, {cluster_key}, {contrast} ###### #')

                # Subset data to tissue and progenitors
                keep_bool_tissue = (tdata.obs['tissue'] == tissue_name_to_label[tissue])
                keep_bool_progenitor = tdata.obs['cluster'].isin(cluster_key_to_progenitor_labels[cluster_key])
                keep_bool = keep_bool_tissue & keep_bool_progenitor
                tdata_sub = tdata[keep_bool, :].copy()

                # Filter genes with low counts
                sc.pp.filter_genes(tdata_sub, min_cells=5)
                sc.pp.filter_genes(tdata_sub, min_counts=5)

                # Create contrast tuple
                if contrast == 'cva':
                    ct = ('chronic', 'acute')
                else:
                    ct = ('acute', 'chronic')

                res_df = zinbwave_edger(
                    adata=tdata_sub, contrast=ct, group_key='infection_name', n_cores=-1
                )

                print(res_df)

                id_str = f'{tissue}_{time}_{cluster_key}_{contrast}'
                res_df.to_csv(os.path.join(save_path_base, f'{id_str}.csv'))

    # ### Progenitor vs offspring in acute
    contrasts = ['pvo', 'ovp']
    for tissue in tissues:
        for cluster_key in clusters:
            for contrast in contrasts:
                print(f'# ###### {tissue}, {cluster_key}, {contrast} ###### #')

                # Subset data to tissue, acute, and cluster
                keep_bool_tissue = (tdata.obs['tissue'] == tissue_name_to_label[tissue])
                keep_bool_cluster = tdata.obs['cluster'].isin(cluster_key_to_cluster_labels[cluster_key])
                keep_bool_acute = (tdata.obs['infection'] == infection_name_to_label['acute'])
                keep_bool = keep_bool_tissue & keep_bool_cluster & keep_bool_acute
                tdata_sub = tdata[keep_bool, :].copy()

                # Filter genes with low counts
                sc.pp.filter_genes(tdata_sub, min_cells=5)
                sc.pp.filter_genes(tdata_sub, min_counts=5)

                # Add progenitor-offspring annotations
                cid_to_pol = cluster_key_to_cid_to_pol[cluster_key]
                tdata_sub.obs['prog_off'] = [cid_to_pol[cid] for cid in tdata_sub.obs['cluster']]

                # Create contrast tuple
                if contrast == 'pvo':
                    ct = ('prog', 'off')
                else:
                    ct = ('off', 'prog')

                res_df = zinbwave_edger(
                    adata=tdata_sub, contrast=ct, group_key='prog_off', n_cores=-1
                )

                print(res_df)

                id_str = f'{tissue}_{time}_{cluster_key}_acute_{contrast}'
                res_df.to_csv(os.path.join(save_path_base, f'{id_str}.csv'))


def main_tcell_plot_figure():
    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    import matplotlib.patheffects as pe
    import seaborn as sns
    import scanpy as sc

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from adjustText import adjust_text
    from scipy.stats import hypergeom

    sp = True
    sp_str = '_sp'

    # Create directory for results
    res_dir = f'./results/05_revision/tcell{sp_str if sp else ""}/figure'
    os.makedirs(res_dir, exist_ok=True)

    # Set parameters
    tissues = ['Spleen', 'Liver']

    tissue_to_id = {'Spleen': 0, 'Liver': 1}

    preliminary = False
    standard_preprocessing = True

    keep_cluster_ids = [1, 2, 3, 4, 5]

    dim_red = 'UMAP'  # 'UMAP', 'DiffMap'

    if preliminary:

        # Load the full data set
        data_p = f'./results/05_revision/tcell{sp_str if sp else ""}/data/'
        fn_all_data = 'ga_an0602_10x_smarta_doc_arm_liver_spleen_d10_d28_mgd_ts_filtered_int_inf_tp_rp_convert.h5ad'
        tdata = sc.read_h5ad(os.path.join(data_p, fn_all_data))

        # Relabel cluster labels from 0-9 to 1-10
        new_labels = [int(label + 1) for label in tdata.obs['cluster']]
        tdata.obs['cluster'] = new_labels

        # Subset cells to time point d10 and clusters 3, 5
        keep_bool_cluster = tdata.obs['cluster'].isin(keep_cluster_ids)
        keep_bool_time = (tdata.obs['time'] == 0)  # day 10
        keep_bool = keep_bool_time & keep_bool_cluster
        tdata = tdata[keep_bool, :].copy()

        # Add semantic annotations
        infection_label_to_name = {0: 'Chronic', 1: 'Acute'}
        tdata.obs['infection_name'] = [infection_label_to_name[lbl] for lbl in tdata.obs['infection']]

        for tissue in tissues:

            # Subset to tissue
            keep_bool_tissue = tdata.obs['tissue'] == tissue_to_id[tissue]
            tdata_sub = tdata[keep_bool_tissue, :].copy()

            if standard_preprocessing:
                # Use raw and do basic preprocessing
                tdata_sub.X = tdata_sub.raw.X.copy()
                sc.pp.filter_genes(tdata_sub, min_cells=20)
                sc.pp.normalize_total(tdata_sub)
                sc.pp.log1p(tdata_sub)

            # Compute UMAP
            sc.pp.pca(tdata_sub)
            sc.pp.neighbors(tdata_sub, n_neighbors=5)
            sc.tl.umap(tdata_sub)
            sc.tl.diffmap(tdata_sub)

            # Save for later plotting
            tdata_sub.raw = None  # Avoid error
            fn = f'd10_35_{tissue}_tdata_with_umap.h5ad'
            tdata_sub.write_h5ad(os.path.join(res_dir, fn))

    # --- Plotting ---
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # Outer 2×3 grid
    outer = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])

    # A.1 / A.2 (nested)
    gsA = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.05)
    axA1 = fig.add_subplot(gsA[0, 0])
    axA2 = fig.add_subplot(gsA[1, 0])

    # B, C
    axB = fig.add_subplot(outer[0, 1])
    axC = fig.add_subplot(outer[0, 2])

    # D.1 / D.2 (nested)
    gsD = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 0], hspace=0.05)
    axD1 = fig.add_subplot(gsD[0, 0])
    axD2 = fig.add_subplot(gsD[1, 0])

    # E, F
    axE = fig.add_subplot(outer[1, 1])
    axF = fig.add_subplot(outer[1, 2])

    # Dict like subplot_mosaic
    axd = {
        'A.1': axA1,
        'A.2': axA2,
        'B': axB,
        'C': axC,
        'D.1': axD1,
        'D.2': axD2,
        'E': axE,
        'F': axF,
    }

    # --- UMAP or Diffusion map
    fontsize_cluster_anno = 8
    for tissue, subplot_keys in zip(tissues, [['A.1', 'A.2'], ['D.1', 'D.2']]):

        # Load the data
        fn = f'd10_35_{tissue}_tdata_with_umap.h5ad'
        tdata_plot = sc.read_h5ad(os.path.join(res_dir, fn))

        # Define color scheme
        palette = sns.color_palette('Set2', n_colors=len(keep_cluster_ids))
        cluster_id_to_color = dict(zip(keep_cluster_ids, palette))

        for subplot_key, grey_label in zip(subplot_keys, ['Chronic', 'Acute']):

            # Create df for plotting
            dim_red_key = 'X_umap' if dim_red == 'UMAP' else 'X_diffmap'
            plot_df = pd.DataFrame(tdata_plot.obsm[dim_red_key][:, 0: 2].copy(), columns=[f'{dim_red}1', f'{dim_red}2'])
            plot_df['cluster'] = tdata_plot.obs['cluster'].tolist().copy()
            plot_df['infection'] = tdata_plot.obs['infection_name'].tolist().copy()

            plot_df['colors'] = [
                'lightgrey' if inf == grey_label else cluster_id_to_color.get(cid, 'lightgrey')
                for cid, inf in zip(plot_df['cluster'], plot_df['infection'])
            ]

            # Get subplot axes
            ax = axd[subplot_key]

            # Scatterplot
            ax.scatter(
                plot_df[f'{dim_red}1'],
                plot_df[f'{dim_red}2'],
                c=plot_df['colors'],
                s=1.0,
            )

            # Style axes
            ax.set_xlabel(f'{dim_red}1')
            ax.set_ylabel(f'{dim_red}2')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add cluster annotations
            texts = []
            for cid in keep_cluster_ids:

                cluster_points = plot_df[
                    (plot_df['cluster'] == cid) & (plot_df['infection'] != grey_label)
                    ].copy()

                x_med = cluster_points[f'{dim_red}1'].median()
                y_med = cluster_points[f'{dim_red}2'].median()

                text = ax.text(
                    x_med, y_med, f'{cid}',
                    fontsize=fontsize_cluster_anno,
                    color='black',
                    ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=1.0, foreground='white')]
                )

                # if grey_label == 'Acute':
                #     texts.append(text)
                texts.append(text)

            if texts:
                adjust_text(texts, ax=ax)

            # Add annotation for grey cluster
            cluster_points_grey = plot_df[plot_df['infection'] == grey_label].copy()
            x_med_grey = cluster_points_grey[f'{dim_red}1'].median()
            y_med_grey = cluster_points_grey[f'{dim_red}2'].median()
            ax.text(
                x_med_grey, y_med_grey, grey_label,
                fontsize=fontsize_cluster_anno,
                color='black',
                ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=1.0, foreground='white')]
            )

    # Set titles
    axd['A.1'].set_title(tissues[0])
    axd['D.1'].set_title(tissues[1])

    # --- Vulcano
    res_p_base_switchtfi = f'./results/05_revision/tcell{sp_str if sp else ""}/switchtfi'
    res_p_base_de = f'./results/05_revision/tcell{sp_str if sp else ""}/de_analysis'

    pvalue_column = 'FDR'  # PValue, FDR
    lfc_column = 'logFC'

    pval_thresh = 0.05
    lfc_thresh = 0.5

    fontsize_tf_anno = 10

    for tissue, subplot_key in zip(tissues, ['B', 'E']):

        # Load SwitchTFI results
        res_p_switchtfi = os.path.join(res_p_base_switchtfi, 'notox', f'{tissue.lower()}_d10_acute_35', 'scenic')
        ranked_tfs_pr = pd.read_csv(os.path.join(res_p_switchtfi, 'ranked_tfs.csv'), index_col=0)
        tfs = ranked_tfs_pr['gene'].tolist()

        # Load DE results
        res_p_de = os.path.join(res_p_base_de, f'{tissue.lower()}_d10_35_cva.csv')
        res_df_de = pd.read_csv(res_p_de, index_col=0)
        res_df_de['TF'] = res_df_de.index.copy()

        # Avoid log(0) error
        eps = 1e-300
        res_df_de[f'-log10({pvalue_column})'] = -np.log10(res_df_de[pvalue_column].astype('float') + eps)

        # Add color columns
        res_df_de['color'] = 'grey'
        sig_bool_neg = (
                (res_df_de[lfc_column] < -lfc_thresh) &
                (res_df_de[pvalue_column] < pval_thresh)
        )
        res_df_de.loc[sig_bool_neg, 'color'] = 'blue'
        sig_bool_pos = (
                (res_df_de[lfc_column] > lfc_thresh) &
                (res_df_de[pvalue_column] < pval_thresh)
        )
        res_df_de.loc[sig_bool_pos, 'color'] = 'green'

        # Plot
        ax = axd[subplot_key]

        sns.scatterplot(
            data=res_df_de,
            x=lfc_column,
            y=f'-log10({pvalue_column})',
            s=8.0,
            hue='color',
            palette={'grey': 'grey', 'blue': 'blue', 'green': 'green'},
            legend=False,
            ax=ax
        )

        # Add driver TF annotations
        texts = []
        for tf in tfs:
            sub = res_df_de.loc[res_df_de['TF'] == tf]
            if not sub.empty:
                x = sub[lfc_column].to_numpy()[0]
                y = sub[f'-log10({pvalue_column})'].to_numpy()[0]

                ax.scatter(x, y, color='red', s=3.0, zorder=3)
                texts.append(ax.text(
                    x, y, tf,
                    ha='right', va='bottom',
                    fontsize=fontsize_tf_anno,
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground='white')]
                ))

        adjust_text(
            texts, ax=ax, expand=(2.0, 2.0),
            force_text=(1.0, 1.0),
            time_lim=3,
            only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'y'},
            arrowprops=dict(arrowstyle='-', color='red', lw=0.5)
        )

        linewidth_axlines = 1.0
        ax.axvline(-lfc_thresh, color='black', linestyle='--', linewidth=linewidth_axlines)
        ax.axvline(lfc_thresh, color='black', linestyle='--', linewidth=linewidth_axlines)
        ax.axhline(-np.log10(pval_thresh), color='black', linestyle='--', linewidth=linewidth_axlines)

        ax.set_xlabel('log2 FC chronic vs acute')
        ax.set_ylabel(f'-log10(P-val adj)')

        ax.set_title(tissue)

    # --- Geometric distribution
    significance_threshold = 0.05
    for tissue, subplot_key in zip(tissues, ['C', 'F']):
        # Load SwitchTFI results
        res_p_switchtfi = os.path.join(res_p_base_switchtfi, 'notox', f'{tissue.lower()}_d10_acute_35', 'scenic')
        ranked_tfs_pr = pd.read_csv(os.path.join(res_p_switchtfi, 'ranked_tfs.csv'), index_col=0)
        tfs = ranked_tfs_pr['gene'].tolist()

        # Load DE results
        res_p_de = os.path.join(res_p_base_de, f'{tissue.lower()}_d10_35_cva.csv')
        res_df_de = pd.read_csv(res_p_de, index_col=0)
        res_df_de['TF'] = res_df_de.index.copy()

        # Define parameters of the hypergeometric distribution
        num_de_driver_tfs = (res_df_de.loc[tfs, pvalue_column] <= significance_threshold).sum()
        total = res_df_de.shape[0]  # M
        num_de = (res_df_de[pvalue_column] <= significance_threshold).sum()  # n
        num_draws = len(tfs)  # N

        pval_exact = 1 - hypergeom.cdf(num_de_driver_tfs - 1, total, num_de, num_draws)

        ax = axd[subplot_key]

        M = total  # population size
        n = num_de  # number of "successes" in population
        N = num_draws  # number of draws
        k_obs = num_de_driver_tfs

        x = np.arange(0, N + 1)
        pmf = hypergeom.pmf(x, M, n, N)

        mask_tail = x >= k_obs
        mask_base = ~mask_tail

        # --- baseline stems (black) ---
        base = ax.stem(
            x[mask_base], pmf[mask_base],
            linefmt='k-', markerfmt='ko', basefmt=' ',
            use_line_collection=True,  # label='PMF'
        )
        base.markerline.set_markersize(4)

        # --- tail stems (light red) ---
        tail = ax.stem(
            x[mask_tail], pmf[mask_tail],
            linefmt='-', markerfmt='o', basefmt=' ',
            use_line_collection=True
        )
        tail.markerline.set_markersize(4)
        tail.markerline.set_markerfacecolor('lightcoral')
        tail.markerline.set_markeredgecolor('lightcoral')
        tail.stemlines.set_color('lightcoral')  # works for LineCollection

        # --- observed value (light red stem + dot) ---
        obs = ax.stem(
            [k_obs], [hypergeom.pmf(k_obs, M, n, N)],
            linefmt='-', markerfmt='o', basefmt=' ',
            use_line_collection=True, label=fr"$P$ = {pval_exact:.2e}"
        )
        obs.markerline.set_markersize(4)
        obs.markerline.set_markerfacecolor('lightcoral')
        obs.markerline.set_markeredgecolor('lightcoral')
        obs.stemlines.set_color('lightcoral')

        # overlay observed marker with red edge ring
        ax.plot(
            k_obs, hypergeom.pmf(k_obs, M, n, N),
            marker='o', markersize=4,
            mfc='lightcoral', mec='red', mew=1.0, linestyle='None', zorder=3,
            label=f'Observed = {k_obs}'
        )

        # mask = x >= k_obs
        # ax.fill_between(x[mask], 0, pmf[mask],
        #                 color="red", alpha=0.3,
        #                 label=f"p = {pval_exact:.2e}")

        ax.legend(loc='upper right')

        ax.set_xlabel('No. of DE TFs')
        ax.set_ylabel('PMF')

        ax.set_title(tissue)

    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-25 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 0.95, label, transform=ax.transAxes + trans,
                fontsize=12, va='bottom', fontfamily='sans-serif', fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, f'fig.png'), dpi=fig.dpi)


def main_tcell_de_plots():

    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scanpy as sc

    from adjustText import adjust_text
    from scipy.stats import hypergeom

    from switchtfi.plotting import plot_regulon

    sp = True
    sp_str = '_sp'

    tissues = ['spleen', 'liver']
    time = 'd10'
    clusters = ['13', '345', '35']
    tox_strs = ['notox', 'wtox']
    grn_inf_methods = ['scenic', 'grnboost2']

    time_name_to_label = {'d10': 0, 'd28': 1}
    tissue_name_to_label = {'spleen': 0, 'liver': 1}
    infection_name_to_label = {'chronic': 0, 'acute': 1}

    pvalue_column = 'FDR'  # PValue, FDR
    lfc_column = 'logFC'

    # Set paths
    res_p_base_switchtfi = f'./results/05_revision/tcell{sp_str if sp else ""}/switchtfi'
    res_p_base_de = f'./results/05_revision/tcell{sp_str if sp else ""}/de_analysis'
    save_path_base = f'./results/05_revision/tcell{sp_str if sp else ""}/de_plots'

    # Load the full data set
    data_p = f'./results/05_revision/tcell{sp_str if sp else ""}/data/'
    fn_all_data = 'ga_an0602_10x_smarta_doc_arm_liver_spleen_d10_d28_mgd_ts_filtered_int_inf_tp_rp_convert.h5ad'
    tdata = sc.read_h5ad(os.path.join(data_p, fn_all_data))

    # Subset to TFs
    tf_file = './data/tf/mus_musculus/allTFs_mm.txt'
    with open(tf_file) as file:
        all_tfs = [line.strip() for line in file.readlines()]
    tf_bool = tdata.var_names.isin(all_tfs)
    tdata = tdata[:, tf_bool].copy()

    # Relabel cluster labels from 0-9 to 1-10
    new_labels = [int(label + 1) for label in tdata.obs['cluster']]
    tdata.obs['cluster'] = new_labels

    # Add semantic annotations
    infection_label_to_name = {0: 'chronic', 1: 'acute'}
    tdata.obs['infection_name'] = [infection_label_to_name[lbl] for lbl in tdata.obs['infection']]

    # Subset data to time point d10
    keep_bool_time = (tdata.obs['time'] == time_name_to_label[time])
    tdata = tdata[keep_bool_time, :].copy()


    for tox_str in tox_strs:
        for grn_inf_method in grn_inf_methods:
            for tissue in tissues:
                for cluster_keys in clusters:

                    # Set identifier
                    id_str = f'{tissue}_{time}_acute_{cluster_keys}'

                    # Load SwitchTFI results
                    res_p_switchtfi = os.path.join(res_p_base_switchtfi, tox_str, id_str, grn_inf_method)
                    try:
                        grn = pd.read_csv(os.path.join(res_p_switchtfi, 'grn.csv'), index_col=0)
                        ranked_tfs_pr = pd.read_csv(os.path.join(res_p_switchtfi, 'ranked_tfs.csv'), index_col=0)
                        tfs = ranked_tfs_pr['gene'].tolist()
                    except FileNotFoundError:
                        print(f'No results found for: {res_p_switchtfi}, skipping')
                        continue

                    # Generate dir for results
                    save_path = os.path.join(save_path_base, tox_str, id_str, grn_inf_method)
                    os.makedirs(save_path, exist_ok=True)

                    # Subset data to tissue
                    keep_bool_tissue = tdata.obs['tissue'] == tissue_name_to_label[tissue]
                    tdata_sub = tdata[keep_bool_tissue, :].copy()

                    # --- (1) DE chronic vs acute in progenitors
                    # Subset data to progenitors
                    prog_clusters = [3, 4] if cluster_keys == '345' else [3, ]
                    keep_bool_progenitor = tdata_sub.obs['cluster'].isin(prog_clusters)
                    tdata_sub_1 = tdata_sub[keep_bool_progenitor, :].copy()

                    # Add semantic annotations
                    tdata_sub_1.obs['infection_name'] = tdata_sub_1.obs['infection'].map({0: 'chronic', 1: 'acute'})

                    # Load DE results
                    fn_de_1 = f'{tissue}_{time}_{cluster_keys}_cva.csv'
                    res_df_de_1 = pd.read_csv(os.path.join(res_p_base_de, fn_de_1), index_col=0)
                    res_df_de_1['TF'] = res_df_de_1.index.copy()

                    # --- (2) DE prog vs. off in acute
                    # Subset data to clusters 3, 4, 5 and acute
                    cluster_list = [3, 4, 5] if cluster_keys == '345' else [3, 5]
                    keep_bool_cluster = tdata_sub.obs['cluster'].isin(cluster_list)
                    keep_bool_acute = (tdata_sub.obs['infection'] == infection_name_to_label['acute'])
                    keep_bool_cluster_acute = keep_bool_cluster & keep_bool_acute
                    tdata_sub_2 = tdata_sub[keep_bool_cluster_acute, :].copy()

                    # Add progenitor-offspring annotations
                    tdata_sub_2.obs['prog_off'] = tdata_sub_2.obs['cluster'].map({3: 'prog', 4: 'prog', 5: 'off'})

                    # Load DE results
                    fn_de_2 = f'{tissue}_{time}_{cluster_keys}_acute_pvo.csv'
                    res_df_de_2 = pd.read_csv(os.path.join(res_p_base_de, fn_de_2), index_col=0)
                    res_df_de_2['TF'] = res_df_de_2.index.copy()


                    # Plot
                    point_size = 12
                    point_size_tf = 6
                    fontsize_tf = 12
                    linewidth_tf = 1.0
                    linewidth_axlines = 1.0
                    fontsize_axlabels = 14
                    fontsize_title = 16
                    tick_label_fontsize = 6

                    fig = plt.figure(figsize=(12, 9), constrained_layout=True, dpi=300)
                    axd = fig.subplot_mosaic(
                        '''
                        AB
                        CD
                        '''
                    )

                    # Volcano
                    for subplot_key, de_results, title in zip(
                            ['A', 'C'],
                            [res_df_de_1, res_df_de_2],
                            [f'Chronic vs. Acute | Progenitors ({cluster_keys})', 'Prog vs. Off | Acute']
                    ):

                        ax = axd[subplot_key]

                        de_results = de_results[de_results[lfc_column].abs() <= 10].copy()

                        eps = 1e-300
                        de_results[f'-log10({pvalue_column})'] = -np.log10(de_results[pvalue_column].astype('float') + eps)

                        pval_thresh = 0.05
                        lfc_thresh = 0.5
                        de_results['color'] = 'grey'
                        sig_bool_neg = (
                                (de_results[lfc_column] < -lfc_thresh) &
                                (de_results[pvalue_column] < pval_thresh)
                        )
                        de_results.loc[sig_bool_neg, 'color'] = 'blue'
                        sig_bool_pos = (
                                (de_results[lfc_column] > lfc_thresh) &
                                (de_results[pvalue_column] < pval_thresh)
                        )
                        de_results.loc[sig_bool_pos, 'color'] = 'green'

                        sns.scatterplot(
                            data=de_results,
                            x=lfc_column,
                            y=f'-log10({pvalue_column})',
                            s=point_size,
                            hue='color',
                            palette={'grey': 'grey', 'blue': 'blue', 'green': 'green'},
                            legend=False,
                            ax=ax
                        )

                        texts = []
                        for tf in tfs:
                            sub = de_results.loc[de_results['TF'] == tf]
                            if not sub.empty:
                                x = sub[lfc_column].to_numpy()[0]
                                y = sub[f'-log10({pvalue_column})'].to_numpy()[0]

                                ax.scatter(x, y, color='red', s=point_size_tf, zorder=3)
                                texts.append(ax.text(x, y, tf, fontsize=fontsize_tf, ha='right', va='bottom'))

                        adjust_text(
                            texts, ax=ax, expand=(3.0, 3.0),
                            arrowprops=dict(arrowstyle='-', color='red', lw=linewidth_tf)
                        )

                        ax.axvline(-lfc_thresh, color='black', linestyle='--', linewidth=linewidth_axlines)
                        ax.axvline(lfc_thresh, color='black', linestyle='--', linewidth=linewidth_axlines)
                        ax.axhline(-np.log10(pval_thresh), color='black', linestyle='--', linewidth=linewidth_axlines)

                        ax.set_xlabel('log2 FC', fontsize=fontsize_axlabels)
                        ax.set_ylabel(f'-log10(p_adj)', fontsize=fontsize_axlabels)

                        ax.set_title(title, fontsize=fontsize_title)

                        ax.xaxis.set_tick_params(labelsize=tick_label_fontsize)
                        ax.yaxis.set_tick_params(labelsize=tick_label_fontsize)

                    # Expression
                    for subplot_key, tdata_sub, de_results, group_key, title in zip(
                            ['B', 'D'],
                            [tdata_sub_1, tdata_sub_2],
                            [res_df_de_1, res_df_de_2],
                            ['infection_name', 'prog_off'],
                            [f'Chronic vs. Acute | Progenitors ({cluster_keys})', 'Prog vs. Off | Acute']
                    ):

                        plot_df = (
                            tdata_sub[:, tfs].to_df()
                            .join(tdata_sub.obs[group_key])
                            .melt(id_vars=group_key, var_name='TF', value_name='Expression')
                        )

                        if group_key == 'prog_off':
                            hue_order = ['prog', 'off']
                        elif group_key == 'infection_name':
                            hue_order = ['chronic', 'acute']
                        else:
                            hue_order = None

                        ax = axd[subplot_key]

                        plot_types = {'box', }  # 'violin', 'strip', 'box', 'boxen'

                        if 'violin' in plot_types:
                            sns.violinplot(
                                data=plot_df,
                                x='TF',
                                y='Expression',
                                hue=group_key,
                                hue_order=hue_order,
                                split=True,
                                inner='quart',
                                cut=0,
                                density_norm='width',
                                ax=ax,
                            )

                        if 'strip' in plot_types:
                            sns.stripplot(
                                data=plot_df,
                                x='TF',
                                y='Expression',
                                hue=group_key,
                                hue_order=hue_order,
                                dodge=True,
                                size=2,
                                ax=ax
                            )

                        if 'box' in plot_types:
                            sns.boxplot(
                                data=plot_df,
                                x='TF',
                                y='Expression',
                                hue=group_key,
                                hue_order=hue_order,
                                ax=ax,
                                showcaps=True,
                            )

                        if 'boxen' in plot_types:
                            sns.boxenplot(
                                data=plot_df,
                                x='TF',
                                y='Expression',
                                hue=group_key,
                                hue_order=hue_order,
                                linewidth=3.0,
                                ax=ax
                            )

                        fig.canvas.draw()
                        tf_to_pval = dict(zip(de_results['TF'], de_results[pvalue_column]))
                        tick_positions = ax.get_xticks()
                        tick_labels = [tick_label.get_text() for tick_label in ax.get_xticklabels()]
                        new_labels = []
                        for lbl in tick_labels:
                            if lbl in tf_to_pval:
                                pval = tf_to_pval[lbl]
                                p_label = 'p_adj' if pvalue_column == 'FDR' else 'p'
                                if pval < 1e-3:
                                    pval_str = f'{p_label}=\n{pval:.1e}'
                                else:
                                    pval_str = f'{p_label}=\n{pval:.3f}'
                                new_labels.append(f'{lbl}\n{pval_str}')
                            else:
                                new_labels.append(lbl)

                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels(new_labels)

                        ax.set_xlabel('')
                        ax.set_ylabel('Expression', fontsize=fontsize_axlabels)

                        ax.set_title(title, fontsize=fontsize_title)

                        tick_label_fontsize_x = (
                            8 if tissue == 'liver' and cluster_keys == '35' else tick_label_fontsize
                        )
                        ax.xaxis.set_tick_params(labelsize=tick_label_fontsize_x)
                        ax.yaxis.set_tick_params(labelsize=tick_label_fontsize)

                    fig.savefig(os.path.join(save_path, f'de_plot.png'), dpi=fig.dpi)
                    plt.close(fig)

                    # For each TF plot its regulon
                    for tf in tfs:
                        fig, ax = plt.subplots(dpi=300)
                        plot_regulon(
                            grn=grn,
                            tf=tf,
                            top_k=30,
                            sort_by='score',
                            ax=ax
                        )
                        fig.tight_layout()
                        fig.savefig(
                            os.path.join(save_path, f'regulon_{tf}.png'),
                            bbox_inches='tight', pad_inches=0.0, dpi=fig.dpi
                        )
                        plt.close(fig)


def main_tcell_explore_results():

    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scanpy as sc

    from switchtfi.utils import load_grn_json, csr_to_numpy


    tissues = ['spleen', ]  # 'liver']
    time = 'd10'
    infection = 'acute'
    clusters = ['345', '35']

    data_p = './results/05_revision/tcell/data/'
    switchtfi_res_p = './results/02_switchtfi/tcell/'

    save_path = './results/05_revision/tcell/results_analysis'
    os.makedirs(save_path, exist_ok=True)

    fn_all_data = 'ga_an0602_10x_smarta_doc_arm_liver_spleen_d10_d28_mgd_ts_filtered_int_inf_tp_rp_convert.h5ad'

    # ### --- First look at results (number of driver TFs, transition GRN size, scatter plots ...) --- ### #
    for tissue in tissues:
        for cluster_keys in clusters:

            id_str = f'{tissue}_{time}_{infection}_{cluster_keys}'
            os.makedirs(os.path.join(save_path, id_str), exist_ok=True)

            # Load the data
            filepath = os.path.join(data_p, f'tdata_{id_str}.h5ad')
            tdata = sc.read_h5ad(filepath)

            # Load the results
            transition_grn = pd.read_csv(os.path.join(switchtfi_res_p, id_str, 'grn.csv'), index_col=0)
            ranked_tfs_pr = pd.read_csv(os.path.join(switchtfi_res_p, id_str, 'ranked_tfs.csv'), index_col=0)
            ranked_tfs_od = pd.read_csv(os.path.join(switchtfi_res_p, id_str, 'outdegree_ranked_tfs.csv'), index_col=0)

            print(f'# ### {id_str} ### #')
            print(f'# ### Transition GRN:\n{transition_grn}')
            print(f'# ### TFs PageRank:\n{ranked_tfs_pr}')
            print(f'# ### TFs outdegree:\n{ranked_tfs_od}')

            full_grn = load_grn_json(os.path.join(switchtfi_res_p, id_str, 'grn.json')).drop(columns=['index'])
            full_grn = full_grn.sort_values(by='weight', ascending=False, ignore_index=True)  # .reset_index(drop=True)

            fig = plt.figure(figsize=(8, 5), constrained_layout=True, dpi=300)
            axd = fig.subplot_mosaic(
                """
                ABC
                DEF
                """
            )

            num_plots_per_row = 3
            n_edges = full_grn.shape[0]
            plot_indices = (
                    list(range(0, num_plots_per_row))
                    + list(range(n_edges - 1, n_edges - 1 - num_plots_per_row, -1))
            )
            subplot_keys = list('ABCDEF')

            layer_key = 'magic_imputed'

            label_to_color = {
                'prog': '#fdae6b',  # warmer orange-peach
                'off': '#a1d99b'  # fresher light green
            }

            legend_handles = [
                plt.Line2D([], [], marker='o', color='w', label=label.capitalize(), markerfacecolor=color, markersize=6)
                for label, color in label_to_color.items()
            ]

            for j, plot_idx in enumerate(plot_indices):

                tf = full_grn.loc[plot_idx, 'TF']
                target = full_grn.loc[plot_idx, 'target']
                weight = full_grn.loc[plot_idx, 'weight']
                threshold = full_grn.loc[plot_idx, 'threshold']
                pred_l = full_grn.loc[plot_idx, 'pred_l']
                pred_r = full_grn.loc[plot_idx, 'pred_r']

                labels = tdata.obs['prog_off'].to_numpy()
                x = csr_to_numpy(tdata[:, tf].layers[layer_key]).flatten()
                y = csr_to_numpy(tdata[:, target].layers[layer_key]).flatten()

                x_bool = (x != 0)
                y_bool = (y != 0)
                keep_bool = np.logical_and(x_bool, y_bool)

                x = x[keep_bool]
                y = y[keep_bool]
                labels_plot = labels[keep_bool]

                colors = [label_to_color[label] for label in labels_plot]

                ax = axd[subplot_keys[j]]

                ax.scatter(
                    x,
                    y,
                    c=colors,
                    alpha=0.9,
                    edgecolors='none',
                    s=10,
                )

                min_x, max_x = x.min(), x.max()

                ax.plot([min_x, threshold], [pred_l, pred_l], color='red', zorder=2)
                ax.scatter([threshold], [pred_l], color='red', marker='o', zorder=3)
                ax.plot([threshold, max_x], [pred_r, pred_r], color='red', zorder=2)
                ax.scatter([threshold], [pred_r], color='red', marker='o', facecolor='white', zorder=3)
                ax.axvline(x=threshold, color='red', linestyle='--', zorder=1)
                ax.set_title(fr'$w = {round(weight, 3)}$')
                ax.set_xlabel(tf)
                ax.set_ylabel(target)

                ax.legend(handles=legend_handles)

            fig.savefig(os.path.join(save_path, id_str, 'scatter_plots.png'), dpi=fig.dpi)


    # ### --- DE analysis for driver TFs --- ### #
    tdata = sc.read_h5ad(os.path.join(data_p, fn_all_data))
    tdata.raw = None

    # Relabel cluster labels from 0-9 to 1-10
    new_labels = [int(label + 1) for label in tdata.obs['cluster']]
    tdata.obs['cluster'] = new_labels

    time_name_to_label = {'d10': 0, 'd28': 1}
    tissue_name_to_label = {'spleen': 0, 'liver': 1}
    infection_name_to_label = {'chronic': 0, 'acute': 1}

    clusters_data = ['35', '345']
    clusters_de = [[3, 5], [3, 4, 5], [3, 4 ], [3, ], [4, ], [5, ]]

    save_p_de = os.path.join(save_path, 'de')
    os.makedirs(save_p_de, exist_ok=True)

    for tissue in tissues:
        for cluster_config_data in clusters_data:
            for cluster_keys_de in clusters_de:

                if (cluster_config_data == '35') and 4 in cluster_keys_de:
                    continue

                # Load the SwitchTFI result
                id_str_switchtfi = f'{tissue}_{time}_{infection}_{cluster_config_data}'
                ranked_tfs_pr = pd.read_csv(
                    os.path.join(switchtfi_res_p, id_str_switchtfi, 'ranked_tfs.csv'), index_col=0
                )
                ranked_tfs_od = pd.read_csv(
                    os.path.join(switchtfi_res_p, id_str_switchtfi, 'outdegree_ranked_tfs.csv'), index_col=0
                )

                ranked_tfs_pr = ranked_tfs_pr['gene'].tolist()
                ranked_tfs_od = ranked_tfs_od['gene'].tolist()
                candidate_tfs = list(set(ranked_tfs_pr + ranked_tfs_od))

                # Subset data
                keep_bool_time = tdata.obs['time'] == time_name_to_label[time]
                keep_bool_tissue = tdata.obs['tissue'] == tissue_name_to_label[tissue]
                keep_bool_cluster_de = tdata.obs['cluster'].isin(cluster_keys_de)

                keep_bool = keep_bool_time & keep_bool_tissue & keep_bool_cluster_de

                tdata_sub = tdata[keep_bool, :].copy()

                # DE analysis
                tdata_sub.obs['infection_name'] = tdata_sub.obs['infection'].map({0: 'chronic', 1: 'acute'})
                sc.tl.rank_genes_groups(
                    tdata_sub,
                    groupby='infection_name',
                    method='wilcoxon',
                )

                for group in ['chronic', 'acute']:

                    group_str = 'cva' if group == 'chronic' else 'avc'

                    cluster_str = ''.join(str(i) for i in cluster_keys_de)
                    id_str = f'{tissue}_{time}_switchtficlust_{cluster_config_data}_declust_{cluster_str}'
                    save_path_de_current = os.path.join(save_p_de, id_str)
                    os.makedirs(save_path_de_current, exist_ok=True)

                    de_results = sc.get.rank_genes_groups_df(tdata_sub, group=group)  # !!! chronic vs acute

                    de_results_tfs = de_results[de_results['names'].isin(candidate_tfs)]

                    de_results.to_csv(os.path.join(save_path_de_current, f'de_results_all_{group_str}.csv'))
                    de_results_tfs.to_csv(os.path.join(save_path_de_current, f'de_results_tfs_{group_str}.csv'))

                    print(de_results_tfs)

                    # Plot results
                    # Heatmap of top DE genes
                    sc.pl.rank_genes_groups_heatmap(
                        tdata_sub,
                        groupby='infection_name',
                        n_genes=20,
                        standard_scale='var',
                        show=False,
                        save=False,
                    )
                    fig = plt.gcf()
                    for ax in fig.axes:
                        for label in ax.get_xticklabels():
                            if label.get_text() in candidate_tfs:
                                label.set_color('red')
                    fig.savefig(os.path.join(save_path_de_current, f'heatmap_{group_str}.png'), dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
                    plt.close(fig)

                    # Dotplot of top DE genes
                    sc.pl.rank_genes_groups_dotplot(
                        tdata_sub,
                        groupby='infection_name',
                        n_genes=20,
                        standard_scale='var',
                        show=False,
                        save=False,
                    )
                    fig = plt.gcf()
                    for ax in fig.axes:
                        for label in ax.get_xticklabels():
                            if label.get_text() in candidate_tfs:
                                label.set_color('red')
                    fig.savefig(os.path.join(save_path_de_current, f'dotplot_scaled_mean_expression_{group_str}.png'), dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
                    plt.close(fig)

                    # Dotplot of top DE genes
                    sc.pl.rank_genes_groups_dotplot(
                        tdata_sub,
                        groupby='infection_name',
                        n_genes=20,
                        values_to_plot='logfoldchanges', cmap='bwr',
                        vmin=-4, vmax=4,
                        colorbar_title='logfoldchange',
                        show=False,
                        save=False,
                    )
                    fig = plt.gcf()
                    for ax in fig.axes:
                        for label in ax.get_xticklabels():
                            if label.get_text() in candidate_tfs:
                                label.set_color('red')
                    fig.savefig(os.path.join(save_path_de_current, f'dotplot_lfc_{group_str}.png'), dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
                    plt.close(fig)

                    # Dotplot candidate driver TFs
                    var_names = {'driver_tfs': candidate_tfs}
                    sc.pl.rank_genes_groups_dotplot(
                        tdata_sub,
                        var_names=var_names,
                        values_to_plot='logfoldchanges', cmap='bwr',
                        vmin=-4, vmax=4,
                        colorbar_title='logfoldchange',
                        show=False,
                        save=False,
                    )
                    fig = plt.gcf()
                    for ax in fig.axes:
                        for label in ax.get_xticklabels():
                            if label.get_text() in candidate_tfs:
                                label.set_color('red')
                    fig.savefig(os.path.join(save_path_de_current, f'dotplot_tfs_lfc_{group_str}.png'), dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
                    plt.close(fig)

                    # Ranked DE genes
                    sc.pl.rank_genes_groups(
                        tdata_sub,
                        groups=None,
                        n_genes=20,
                        show=False,
                        save=False,
                    )
                    fig = plt.gcf()
                    for ax in fig.axes:
                        for label in ax.get_xticklabels():
                            if label.get_text() in candidate_tfs:
                                label.set_color('red')
                    fig.savefig(os.path.join(save_path_de_current, f'de_ranking_{group_str}.png'), dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
                    plt.close(fig)

                    # Volcano plot
                    de_results['-logQ'] = -np.log(de_results['pvals'].astype('float'))
                    de_results = de_results[de_results['logfoldchanges'].abs() <= 10].copy()

                    lowqval_de = de_results.loc[abs(de_results['logfoldchanges']) > 1.5]
                    other_de = de_results.loc[abs(de_results['logfoldchanges']) <= 1.5]

                    fig, ax = plt.subplots(dpi=300)
                    sns.regplot(
                        x=other_de['logfoldchanges'],
                        y=other_de['-logQ'],
                        fit_reg=False,
                        scatter_kws={'s': 6},
                    )
                    sns.regplot(
                        x=lowqval_de['logfoldchanges'],
                        y=lowqval_de['-logQ'],
                        fit_reg=False,
                        scatter_kws={'s': 6},
                    )

                    for tf in candidate_tfs:
                        sub = de_results.loc[de_results['names'] == tf]
                        if not sub.empty:
                            x = sub["logfoldchanges"].to_numpy()[0]
                            y = sub["-logQ"].to_numpy()[0]

                            ax.scatter(x, y, color='red', s=6, zorder=3)
                            ax.text(x, y, tf, fontsize=6, ha='right', va='bottom')

                    ax.set_xlabel('log2 FC')
                    ax.set_ylabel('-log Q-value')

                    ax.set_title(group_str)
                    fig.savefig(os.path.join(save_path_de_current, f'volcanoe_{group_str}.png'), dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)




if __name__ == '__main__':

    # main_no_imputation_results()

    # main_tf_ranking_similarity_old()

    # main_targets_enrichment_analysis()

    # main_targets_enrichment_plots()

    # main_tf_ranking_similarity()

    # main_additional_tf_target_scatter_plots()

    # main_revised_regulon_plot()

    # main_scalability_compute_edge_fraction()

    # main_scalability_plot_figure()

    # main_tcell_data_exploration()

    # main_tcell_data_processing()

    # main_tcell_grn_inference()

    # main_tcell_grn_exploration()

    # main_tcell_switchtfi()

    # main_tcell_de_analysis()

    main_tcell_de_plots()

    # main_tcell_plot_figure()

    # main_tcell_explore_results()

    print('done')
