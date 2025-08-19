

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


def main_tcell_data_exploration():

    import os

    import matplotlib.pyplot as plt
    import scanpy as sc

    data_dir = './data/anndata/tcell'

    plot_dir = './data/anndata/tcell/plots'
    os.makedirs(plot_dir, exist_ok=True)

    load_full_dataset = True

    # ### Load full dataset and subset (tissue spleen, day 10)
    if load_full_dataset:

        full_dataset_filename = 'ga_an0602_10x_smarta_doc_arm_liver_spleen_d10_d28_mgd_ts_filtered_int_inf_tp_rp_convert.h5ad'
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
                tdata_subset.write(os.path.join(data_dir, f'tdata_{tissue}_{time}.h5ad'))
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
            tdata = sc.read_h5ad(os.path.join(data_dir, f'tdata_{tissue}_{time}.h5ad'))

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

    import matplotlib.pyplot as plt
    import scanpy as sc
    import scanpy.external as scex

    data_dir = './data/anndata/tcell'

    tissues = ['spleen', 'liver']
    time = 'd10'
    clusters = [[3, 4, 5], [3, 5]]
    prog_off_labels = {3: 'prog', 4: 'prog', 5: 'off'}


    for tissue in tissues:
        for cluster_keys in clusters:

            # Load data
            tdata = sc.read_h5ad(os.path.join(data_dir, f'tdata_{tissue}_{time}.h5ad'))

            # Subset to populations of interest
            keep_bool_cluster = tdata.obs['cluster'].isin(cluster_keys)
            tdata = tdata[keep_bool_cluster, :].copy()

            # Add progenitor, offspring annotations
            cluster_labels = tdata.obs['cluster'].tolist()
            prog_off_anno = [prog_off_labels[cluster] for cluster in cluster_labels]
            tdata.obs['prog_off'] = prog_off_anno

            # Convert sparse to dense
            tdata.X = tdata.X.toarray()

            # Basic count based QC on the gene level
            sc.pp.filter_genes(tdata, min_cells=20)

            # MAGIC imputation
            tdata_dummy = tdata.copy()
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
            tdata.layers['magic_imputed'] = tdata_dummy.X.copy()

            # Scale to unit variance for GRN inference
            tdata.layers['unit_variance'] = sc.pp.scale(tdata.X.copy(), zero_center=False, copy=True)

            # Save
            cluster_str = ''.join(str(i) for i in cluster_keys)
            tdata.write(os.path.join(data_dir, f'tdata_{tissue}_{time}_{cluster_str}.h5ad'))


def main_tcell_grn_inference():

    import scanpy as sc

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

    data_filenames = [
        'toxdata_tissue_spleen_time_d10_genotype_wt_cluster_12_processed.h5ad',
        # 'toxdata_tissue_spleen_time_d10_genotype_ko_cluster_12_processed.h5ad'
    ]

    base_res_p = './results/01_grn_inf/tox'
    n_grns = 18
    n_occurrence_threshold = 9

    for filename in data_filenames:
        # ### Load AnnData wist scRNA-seq data
        toxdata = sc.read_h5ad(os.path.join('./data/anndata', filename))

        # ### Set parameters and perform GRN inference with Scenic
        res_p = os.path.join(base_res_p, filename[:-5])
        os.makedirs(res_p, exist_ok=True)

        gene_names = toxdata.var_names.tolist()
        tox_genes = [v for v in gene_names if v.startswith('Tox') or v.startswith('tox')]
        print('# ### Genes of the Tox family that are expressed in the dataset are added to the TFs:\n', tox_genes)

        for i in range(n_grns):
            print(f'# ### GRN inference, iteration {i}')

            pyscenic_pipeline(
                adata=toxdata.copy(),
                layer_key='scaled_log1p_norm',
                tf_file=tf_file,
                result_folder=res_p,
                database_path=db_file,
                motif_annotations_path=anno_file,
                grn_inf_method='grnboost2',
                fn_prefix=f'{i:02d}_tox_',
                verbosity=1,
                plot=False
            )

        # ### Combine the 18 individual Scenic GRNs into one
        # Edges that occur in >= n_occurrence_threshold individual GRNs are retained
        print('### Combining Pyscenic GRNs')
        grn_list = []
        # Get list of paths to csv files
        csv_files = glob.glob(res_p + '/*_tox_pruned_grn.csv')
        for csv_file in csv_files:
            grn_list.append(pd.read_csv(csv_file, index_col=[0]))

        combined_grn_scenic = combine_grns(
            grn_list=grn_list,
            n_occurrence_thresh=n_occurrence_threshold,
            result_folder=res_p,
            verbosity=1,
            fn_prefix=f'ngrnthresh{n_occurrence_threshold}_tox_pyscenic_'
        )

        # ### Combine GrnBoost2 GRNs into one (not needed afterwards)
        print('### Combining Pyscenic GRNs')
        grn_list = []
        csv_files = glob.glob(res_p + '/*_tox_basic_grn.csv')

        for csv_file in csv_files:
            grn = pd.read_csv(csv_file, sep='\t')
            # Extract top 1% of important edges
            top_1_perc = math.ceil(grn.shape[0] * 0.01)
            grn_list.append(grn[0:top_1_perc])

        combined_grn_grnboost2 = combine_grns(
            grn_list=grn_list,
            n_occurrence_thresh=n_occurrence_threshold,
            result_folder=res_p,
            verbosity=1,
            fn_prefix=f'ngrnthresh{n_occurrence_threshold}_tox_grnboost_'
        )

        # Check whether Tox family genes appear as TFs or targets
        res_dfs = []
        gene_col = []
        for tox_gene in tox_genes:
            tfs_scenic = combined_grn_scenic['TF'].tolist()
            n_occ_tf_scenic = sum([1 if gene == tox_gene else 0 for gene in tfs_scenic])
            targets_scenic = combined_grn_scenic['target'].tolist()
            n_occ_target_scenic = sum([1 if gene == tox_gene else 0 for gene in targets_scenic])

            tfs_grnboost2 = combined_grn_grnboost2['TF'].tolist()
            n_occ_tf_grnboost2 = sum([1 if gene == tox_gene else 0 for gene in tfs_grnboost2])
            targets_grnboost2 = combined_grn_grnboost2['target'].tolist()
            n_occ_target_grnboost2 = sum([1 if gene == tox_gene else 0 for gene in targets_grnboost2])

            res_df = pd.DataFrame(
                data=[[n_occ_tf_scenic, n_occ_tf_grnboost2], [n_occ_target_scenic, n_occ_target_grnboost2]],
                index=['TF', 'target'],
                columns=['SCENIC', 'GRNboost2']
            )

            res_dfs.append(res_df)
            gene_col.extend([tox_gene, tox_gene])

            print(f'# ### Gene {tox_gene} occurs in the output GRN:\n{res_df}')

        res_df = pd.concat(res_dfs)
        res_df['Tox gene'] = gene_col
        res_df.to_csv(os.path.join(res_p, 'tox_genes_in_grns.csv'))



if __name__ == '__main__':

    # main_no_imputation_results()

    # main_additional_tf_target_scatter_plots()

    main_tcell_data_exploration()

    main_tcell_data_processing()






    '''
    data_dir = './data_dummy'
    os.makedirs(data_dir, exist_ok=True)

    scv.settings.data_path = data_dir

    adata = scv.datasets.pancreas(os.path.join(data_dir, 'endocrinogenesis_day15.h5ad'))
    print(f'# ### Pancreas: {adata.n_obs} cells, {adata.n_vars} genes')

    adata = scv.datasets.dentategyrus(os.path.join(data_dir, 'dentategyrus.h5ad'))
    print(f'# ### Dentategyrus: {adata.n_obs} cells, {adata.n_vars} genes')

    # adata = scv.datasets.forebrain()  # os.path.join(data_dir, 'forebrain.loom'))
    # print(f'# ### Forebrain: {adata.n_obs} cells, {adata.n_vars} genes')

    # adata = scv.datasets.dentategyrus_lamanno()  # os.path.join(data_dir, 'dentategyrus_lamanno.h5ad'))
    # print(f'# ### Dentategyrus_lamanno: {adata.n_obs} cells, {adata.n_vars} genes')

    adata = scv.datasets.gastrulation(os.path.join(data_dir, 'gastrulation.h5ad'))
    print(f'# ### Gastrulation: {adata.n_obs} cells, {adata.n_vars} genes')

    adata = scv.datasets.gastrulation_e75(os.path.join(data_dir, 'gastrulation_e75.h5ad'))
    print(f'# ### Gastrulation_e75: {adata.n_obs} cells, {adata.n_vars} genes')

    adata = scv.datasets.gastrulation_erythroid(os.path.join(data_dir, 'gastrulation_erythroid.h5ad'))
    print(f'# ### Gastrulation_erythroid: {adata.n_obs} cells, {adata.n_vars} genes')

    adata = scv.datasets.bonemarrow(os.path.join(data_dir, 'bonemarrow.h5ad'))
    print(f'# ### Bonemarrow: {adata.n_obs} cells, {adata.n_vars} genes')

    adata = scv.datasets.pbmc68k(os.path.join(data_dir, 'pbmc68k.h5ad'))
    print(f'# ### pbmc68k: {adata.n_obs} cells, {adata.n_vars} genes')

    simdata = scv.datasets.simulation(n_obs=1000, n_vars=10000, switches=3, random_seed=42)

    print(simdata)

    # print(simdata.X)

    sc.pp.pca(simdata)
    sc.pp.neighbors(simdata)

    # Step 3: Compute UMAP
    sc.tl.umap(simdata)

    # Step 4: Plot UMAP
    sc.pl.umap(simdata, color='true_t')
    '''

    print('done')

