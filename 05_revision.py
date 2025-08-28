

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


def main_tcell_data_exploration():

    import os

    import matplotlib.pyplot as plt
    import scanpy as sc

    data_dir = './results/05_revision/tcell/data'
    plot_dir = './results/05_revision/tcell/plots/data_exploration'
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

    import scanpy as sc
    import scanpy.external as scex

    data_dir = './results/05_revision/tcell/data'

    tissues = ['spleen', 'liver']
    time = 'd10'
    infections = ['chronic', 'acute']
    clusters = [[3, 4, 5], [3, 5]]

    cluster_ids_to_prog_off_labels = {3: 'prog', 4: 'prog', 5: 'off'}

    time_name_to_label = {'d10': 0, 'd28': 1}
    tissue_name_to_label = {'spleen': 0, 'liver': 1}
    infection_name_to_label = {'chronic': 0, 'acute': 1}

    # Load the full dataset
    full_dataset_filename = 'ga_an0602_10x_smarta_doc_arm_liver_spleen_d10_d28_mgd_ts_filtered_int_inf_tp_rp_convert.h5ad'
    tdata = sc.read_h5ad(os.path.join(data_dir, full_dataset_filename))

    # Delete the raw to avoid error when saving
    tdata.raw = None

    # Relabel cluster labels from 0-9 to 1-10
    new_labels = [int(label + 1) for label in tdata.obs['cluster']]
    tdata.obs['cluster'] = new_labels

    # Convert sparse to dense
    tdata.X = tdata.X.toarray()

    for tissue in tissues:
        for infection in infections:
            for cluster_keys in clusters:

                # Subset to populations of interest
                keep_bool_time = tdata.obs['time'] == time_name_to_label[time]
                keep_bool_tissue = tdata.obs['tissue'] == tissue_name_to_label[tissue]
                keep_bool_infection = tdata.obs['infection'] == infection_name_to_label[infection]
                keep_bool_cluster = tdata.obs['cluster'].isin(cluster_keys)

                keep_bool = keep_bool_time & keep_bool_tissue & keep_bool_infection & keep_bool_cluster

                tdata_subset = tdata[keep_bool, :].copy()

                # Add progenitor-offspring annotations
                cluster_labels = tdata_subset.obs['cluster'].tolist()
                prog_off_anno = [cluster_ids_to_prog_off_labels[cluster] for cluster in cluster_labels]
                tdata_subset.obs['prog_off'] = prog_off_anno

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
    clusters = ['345', '35']
    n_grns = 18
    edge_count_threshold = 9

    inference = True

    data_p = './results/05_revision/tcell/data'
    base_res_p = './results/05_revision/tcell/grn'
    os.makedirs(base_res_p, exist_ok=True)

    for infection in infections:
        for tissue in tissues:
            for cluster_keys in clusters:

                # Todo
                # t = (tissue, infection, cluster_keys)
                # skip = {('spleen', 'acute', '345'), ('spleen', 'acute', '35'), ('spleen', 'chronic', '345')}
                # if t in skip:
                #     continue

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

    tissues = ['spleen', 'liver']
    time = 'd10'
    infection = 'acute'
    clusters = ['345', '35']

    grn_p = './results/05_revision/tcell/grn'

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


def main_tcell_switchtfi():

    import os

    import pandas as pd
    import scanpy as sc

    from switchtfi import fit_model, rank_tfs

    tissues = ['spleen', 'liver']
    time = 'd10'
    infection = 'acute'
    clusters = ['345', '35']

    data_p = './data/anndata/tcell'
    grn_p = './results/01_grn_inf/tcell'
    base_res_p = './results/02_switchtfi/tcell'

    for tissue in tissues:
        for cluster_keys in clusters:

            id_str = f'{tissue}_{time}_{infection}_{cluster_keys}'

            # Load data
            filepath = os.path.join(data_p, f'tdata_{id_str}.h5ad')
            tdata = sc.read_h5ad(filepath)

            # Load the precomputed GRN
            grn_path = os.path.join(grn_p, id_str, 'edge_count_threshold_9_scenic_aggregated_grn.csv')
            grn = pd.read_csv(grn_path, index_col=0).reset_index()

            # Create results directory
            res_p = os.path.join(base_res_p, 'grnboost2', id_str)  # Todo
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


def main_tcell_explore_results():

    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scanpy as sc

    from switchtfi.utils import load_grn_json, csr_to_numpy


    tissues = ['spleen', ]  # 'liver']
    time = 'd10'
    infection = 'acute'
    clusters = ['345', '35']

    data_p = './data/anndata/tcell'
    base_res_p = './results/02_switchtfi/tcell'

    for tissue in tissues:
        for cluster_keys in clusters:

            id_str = f'{tissue}_{time}_{infection}_{cluster_keys}'

            # Load the data
            filepath = os.path.join(data_p, f'tdata_{id_str}.h5ad')
            tdata = sc.read_h5ad(filepath)

            # Load the results
            transition_grn = pd.read_csv(os.path.join(base_res_p, id_str, 'grn.csv'), index_col=0)
            ranked_tfs_pr = pd.read_csv(os.path.join(base_res_p, id_str, 'ranked_tfs.csv'), index_col=0)
            ranked_tfs_od = pd.read_csv(os.path.join(base_res_p, id_str, 'outdegree_ranked_tfs.csv'), index_col=0)

            print(f'# ### {id_str} ### #')
            print(f'# ### Transition GRN:\n{transition_grn}')
            print(f'# ### TFs PageRank:\n{ranked_tfs_pr}')
            print(f'# ### TFs outdegree:\n{ranked_tfs_od}')

            full_grn = load_grn_json(os.path.join(base_res_p, id_str, 'grn.json')).drop(columns=['index'])
            full_grn = full_grn.sort_values(by='weight', ascending=False, ignore_index=True)  # .reset_index(drop=True)

            print(full_grn)

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

            fig.savefig(os.path.join(base_res_p, id_str, 'scatter_plots.png'), dpi=fig.dpi)


if __name__ == '__main__':

    # main_no_imputation_results()

    # main_tf_ranking_similarity_old()



    # main_tf_ranking_similarity()

    # main_additional_tf_target_scatter_plots()

    # main_revised_regulon_plot()

    # main_tcell_data_exploration()

    # main_tcell_data_processing()

    # main_tcell_grn_inference()

    # main_tcell_grn_exploration()

    # main_tcell_switchtfi()

    # main_tcell_explore_results()  # todo: de testing, check if edges present in chronic grn


    load = False
    if load:
        import os
        import scanpy as sc
        import cellrank as cr

        data_dir = './data_dummy'
        os.makedirs(data_dir, exist_ok=True)

        adata = cr.datasets.reprogramming_morris(os.path.join(data_dir, 'reprogramming_morris.h5ad'), subset='full')
        print(f'# ### Reprogramming Morris: {adata.n_obs} cells, {adata.n_vars} genes')
        print(adata)
        # print(adata.X)
        print(type(adata.X))
        print(type(adata.X[0, 0]))
        print(adata.obs['pseudotime'])
        print(adata.obs['timecourse'])
        print(adata.obs['cell_type'])


    print('done')
