
import os
import argparse
import time
import psutil
import threading
import subprocess


import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from pathlib import Path
from typing import Callable, Dict, Tuple, Union, Any

TEST = False

if Path('/home/woody/iwbn/iwbn107h').is_dir():
    SAVE_PATH = Path('/home/woody/iwbn/iwbn107h/scalability')
else:
    SAVE_PATH = Path('./results/05_revision/scalability')

INTERM_RES_SUBDIR = 'intermediate_results'
os.makedirs(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR), exist_ok=True)

# --- Number of genes always fix
NUM_GENES = 10000 if not TEST else 300

# --- Vary number of cells, vary number of edges (GRN: small, medium, large)
VARY_NUM_CELLS_NUM_CELLS = [100, 500, 1000, 5000, 10000, 50000, 85010] if not TEST else [50, 100, 150]
VARY_NUM_CELLS_NUM_EDGES = [0.05, 0.10, 0.5]

# --- Vary number of edges, vary number of cells (low, medium, high)
VARY_NUM_EDGES_NUM_EDGES = [100, 500, 1000, 5000, 10000, 50000] if not TEST else [100, 200, 300]
VARY_NUM_EDGES_NUM_CELLS = [1000, 10000, 50000] if not TEST else [50, 100, 150]

GRN_INF_METHOD_INPUT = 'scenic' if not TEST else 'grnboost2'

TRACKING_INTERVAL = 0.1


def get_cpu_memory_mb(process: psutil.Process) -> float:
    total_mem = 0
    try:
        with process.oneshot():
            children = process.children(recursive=True)
            all_procs = [process] + children
            for proc in all_procs:

                try:
                    if proc.is_running():
                        total_mem += proc.memory_info().rss

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

    except Exception as e:
        print(f'CPU memory tracking failed with error:\n{e}')

    total_mem /= 1024 ** 2

    return total_mem


def track_memory_cpu(interval=0.1):
    """
    Tracks total memory (RSS) of the current process + children.
    Returns a list of memory samples (in MB).
    """

    process = psutil.Process(os.getpid())
    memory_samples = [get_cpu_memory_mb(process=process)]
    stop_event = threading.Event()

    # Initial sample
    def poll():
        while not stop_event.is_set():
            mem = get_cpu_memory_mb(process=process)
            memory_samples.append(mem)
            stop_event.wait(interval)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()

    return memory_samples, stop_event, thread


def track_memory_gpu(interval=0.1):
    """
    Tracks GPU 0 memory usage over time in a background thread.
    Returns (samples_list, stop_event, thread).
    """
    memory_samples = []
    stop_event = threading.Event()

    interval_ms = max(1, int(interval * 1000))

    # Start a persistent nvidia-smi process
    try:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i", "0",  # Pin 1st GPU
                f"-lms", str(interval_ms),  # Sampling interval in ms
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
    except TypeError:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i", "0",
                f"-lms", str(interval_ms),
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,  # instead of text=True
            bufsize=1
        )

    # Get first sample
    first = 0
    if proc.stdout is not None:
        try:
            first_line = proc.stdout.readline().strip()

            if first_line:
                first = int(first_line)

        except Exception as e:
            pass

    memory_samples.append(first)

    def poll():
        try:
            for line in proc.stdout:
                try:
                    mem = int(line.strip())
                except ValueError:
                    continue

                memory_samples.append(mem)

                if stop_event.is_set():
                    break
        finally:
            # Clean up process when stopping
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                pass

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()

    return memory_samples, stop_event, thread


def scalability_wrapper(
        function: Callable,
        function_params: Union[Dict[str, Any], None]= None,
        track_gpu: bool = False,
        res_dir: Union[str, None] = None,
        res_filename: Union[str, None] = None,
) -> Tuple[pd.DataFrame, Any]:

    # Start memory tracking
    memory_samples_cpu, stop_event_cpu, tracker_thread_cpu = track_memory_cpu(interval=TRACKING_INTERVAL)
    if track_gpu:
        memory_samples_gpu, stop_event_gpu, tracker_thread_gpu = track_memory_gpu(interval=TRACKING_INTERVAL)

    wall_start = time.perf_counter()

    try:

        if function_params is not None:
            function_output = function(**function_params)
        else:
            function_output = function()

    finally:

        wall_end = time.perf_counter()

        # Stop memory tracker
        stop_event_cpu.set()
        tracker_thread_cpu.join()

        if track_gpu:
            stop_event_gpu.set()
            tracker_thread_gpu.join()

    # Analyze results
    wall_time = wall_end - wall_start

    memory_peak_cpu = max(memory_samples_cpu)
    memory_average_cpu = sum(memory_samples_cpu) / len(memory_samples_cpu)

    if track_gpu:
        memory_peak_gpu = max(memory_samples_gpu)
        memory_average_gpu = sum(memory_samples_gpu) / len(memory_samples_gpu)
    else:
        memory_peak_gpu = None
        memory_average_gpu = None

    res = {
        'wall_time': wall_time,
        'mem_peak_cpu': memory_peak_cpu,
        'mem_avg_cpu': memory_average_cpu,
        'samples_cpu': len(memory_samples_cpu),
        'mem_peak_gpu': memory_peak_gpu,
        'mem_avg_gpu': memory_average_gpu,
        'samples_gpu': len(memory_samples_gpu) if track_gpu else None,
    }
    res_df = pd.DataFrame([res])

    if res_dir is not None:
        if res_filename is None:
            res_filename = 'scalability_results.csv'
        res_df.to_csv(os.path.join(res_dir, res_filename))

    return res_df, function_output


def process_data():

    import cellrank as cr

    save_path = SAVE_PATH / 'data'
    os.makedirs(save_path, exist_ok=True)

    # Check whether data generation was run beforehand
    existing_files = [
        f.name for f in save_path.iterdir() if not (f.name.startswith('.') or f.name == 'reprogramming_morris.h5ad')
    ]
    if existing_files:
        raise RuntimeError(
            f'Data processing was already run. '
            f'Remove existing files {existing_files} in "{save_path}" before running again.'
        )

    # Download data
    adata = cr.datasets.reprogramming_morris(os.path.join(save_path, 'reprogramming_morris.h5ad'), subset='85k')

    # Filter genes
    sc.pp.filter_genes(adata, min_cells=20)

    # Subset to the top 10,000 hvg genes
    adata_proc = adata.copy()
    sc.pp.normalize_total(adata_proc)
    sc.pp.log1p(adata_proc)
    sc.pp.highly_variable_genes(adata_proc)
    top_genes = adata_proc.var['dispersions_norm'].nsmallest(NUM_GENES).index
    adata_hvg = adata[:, top_genes].copy()

    # Add progenitor-offspring annotations based on reprogramming day
    rpd_to_anno = {
        '0': 'prog', '3': 'prog', '6': 'off', '9': 'off', '12': 'off', '15': 'off', '21': 'off', '28': 'off'
    }
    prog_off_anno = [rpd_to_anno[rpd] for rpd in adata_hvg.obs['reprogramming_day']]
    adata_hvg.obs['prog_off'] = prog_off_anno

    # Save AnnData, individual data matrices and relevant annotations
    adata_hvg.write_h5ad(os.path.join(save_path, 'reprogramming_morris_hvg.h5ad'))
    x_unspliced = adata_hvg.layers['unspliced']
    x_spliced = adata_hvg.layers['spliced']
    sp.save_npz(os.path.join(save_path, f'x_unspliced.npz'), x_unspliced)
    sp.save_npz(os.path.join(save_path, f'x_spliced.npz'), x_spliced)

    np.save(os.path.join(save_path, f'cell_names.npy'), adata_hvg.obs_names.to_numpy())
    np.save(os.path.join(save_path, f'gene_names.npy'), adata_hvg.var_names.to_numpy())
    np.save(os.path.join(save_path, f'prog_off_anno.npy'), adata_hvg.obs['prog_off'].to_numpy())


def load_data(n_obs: Union[int, None] = None, seed: int = 42) -> sc.AnnData:

    save_path = SAVE_PATH / 'data'

    if not (save_path / 'reprogramming_morris_hvg.h5ad').exists():
        raise RuntimeError(
            f"Missing expected file 'reprogramming_morris_hvg.h5ad'. Make sure process_data() has been run first."
        )

    # Load npy files to avoid errors caused by incompatible Scanpy versions
    x_unspliced = sp.load_npz(os.path.join(save_path, f'x_unspliced.npz')).toarray().astype(np.float32)
    x_spliced = sp.load_npz(os.path.join(save_path, f'x_spliced.npz')).toarray().astype(np.float32)
    cell_names = np.load(os.path.join(save_path, f'cell_names.npy'), allow_pickle=True)
    gene_names = np.load(os.path.join(save_path, f'gene_names.npy'), allow_pickle=True)
    prog_off_anno = np.load(os.path.join(save_path, f'prog_off_anno.npy'), allow_pickle=True)

    # Create anndata
    adata = sc.AnnData(X=x_spliced)
    adata.obs_names = cell_names
    adata.var_names = gene_names
    adata.layers['unspliced'] = x_unspliced
    adata.layers['spliced'] = x_spliced
    adata.obs['prog_off'] = prog_off_anno

    # Subsample to the desired number of cells
    if n_obs is None:
        n_obs = adata.n_obs
    np.random.seed(seed)
    n_total = adata.n_obs
    idx = np.random.choice(n_total, size=n_obs, replace=False)
    adata_sub = adata[idx, :].copy()

    return adata_sub


def load_grn(
        n_obs: int,
        n_edges: Union[int, float, None] = None,
        grn_inf_method: str = 'scenic'
) -> pd.DataFrame:

    if grn_inf_method not  in {'scenic', 'grnboost2'}:
        raise ValueError('grn_inf_method must be "scenic" or "grnboost2"')

    # Load full GRN
    fn_grn = f'grn_{grn_inf_method}_num_cells_{n_obs}.csv'
    grn_path = os.path.join(SAVE_PATH, 'grn_inf', fn_grn)
    grn = pd.read_csv(grn_path, index_col=0)

    # Subset GRN
    if n_edges is not None:

        if isinstance(n_edges, float):
            n_edges = int(n_edges * grn.shape[0])

        grn = grn.iloc[:n_edges].copy()

    return grn


def scalability_grn_inf():

    import pickle
    from arboreto.algo import grnboost2

    from grn_inf.grn_inference import (
        load_tf_names, check_tf_gene_set_intersection, modules_from_grn, prune_grn, pyscenic_result_df_to_grn
    )

    # Define path where results are saved
    save_path = SAVE_PATH / 'grn_inf'
    os.makedirs(save_path, exist_ok=True)

    # Define paths to auxiliary files
    tf_file = './data/tf/mus_musculus/allTFs_mm.txt'
    db_file = (
            './data/scenic_aux_data/databases/mouse/mm10/mc_v10_clust'
            '/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'
    )
    anno_file = './data/scenic_aux_data/motif2tf_annotations/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl'

    # Run GRN inference varying numbers of cells
    for num_cells in VARY_NUM_CELLS_NUM_CELLS:

        # Load the data and do basic processing
        adata = load_data(n_obs=num_cells)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

        adata_df = adata.to_df(layer=None)

        # Load the list of TFs
        tf_names = load_tf_names(path=str(tf_file))
        check_tf_gene_set_intersection(
            tf_names=np.array(tf_names),
            gene_names=adata.var_names.to_numpy(),
            verbosity=1
        )

        # Infer a basic GRN with GRNboost2
        fn_kwargs_grnboost2 = {
            'expression_data': adata_df,
            'gene_names': None,
            'tf_names': tf_names,
            'seed': 42,
            'verbose': True,
        }
        res_df_grnboost2, grn_grnboost2 = scalability_wrapper(
            function=grnboost2,
            function_params=fn_kwargs_grnboost2,
            track_gpu=False,
        )

        # Derive potential regulons from co-expression modules
        res_df_modules, modules = scalability_wrapper(
            function=modules_from_grn,
            function_params={
                'adjacencies': grn_grnboost2,
                'expression_matrix': adata_df,
                'result_folder': None,
            },
            track_gpu=False,
        )

        # Prune modules for targets with cis regulatory footprints (RcisTarget)
        res_df_pruning, scenic_result = scalability_wrapper(
            function=prune_grn,
            function_params={
                'modules': modules,
                'database_path': db_file,
                'motif_annotations_path': anno_file,
                'result_folder': None,
            },
            track_gpu=False,
        )

        # Extract pruned GRN from Scenic results dataframe
        res_df_scenic_to_grn, grn_scenic = scalability_wrapper(
            function=pyscenic_result_df_to_grn,
            function_params={
                'pyscenic_result_df': scenic_result,
                'result_folder': None,
            },
            track_gpu=False,
        )

        # Save (intermediate) results
        grn_grnboost2 = grn_grnboost2.sort_values(by='importance', ascending=False).reset_index(drop=True)
        grn_grnboost2.to_csv(os.path.join(save_path, f'grn_grnboost2_num_cells_{num_cells}.csv'))

        modules_p = os.path.join(save_path, f'modules_num_cells_{num_cells}.pkl')
        with open(modules_p, 'wb') as f:
            pickle.dump(modules, f)

        scenic_result.to_csv(os.path.join(save_path, f'scenic_result_num_cells_{num_cells}.csv'))

        grn_scenic = grn_scenic.sort_values(by='scenic_weight', ascending=False).reset_index(drop=True)
        grn_scenic.to_csv(os.path.join(save_path, f'grn_scenic_num_cells_{num_cells}.csv'))

        # Save scalability results
        res_dfs_sub = [res_df_grnboost2, res_df_modules, res_df_pruning, res_df_scenic_to_grn]
        res_df = pd.concat(res_dfs_sub, axis=0, ignore_index=True)
        res_df['n_cells'] = [num_cells] * 4
        res_df['alg_step'] = ['grnboost2', 'modules', 'pruning', 'scenic_to_grn']

        fn_fg = f'grn_inf_fine_grained_num_cells_{num_cells}.csv'
        res_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn_fg))

        summary_df = (
            res_df
            .drop(columns=['alg_step'])
            .groupby('n_cells', as_index=False)
            .agg({
                'wall_time': 'sum',
                'mem_peak_cpu': 'max', 'mem_avg_cpu': 'mean', 'samples_cpu': 'sum',
                'mem_peak_gpu': 'max', 'mem_avg_gpu': 'mean', 'samples_gpu': 'sum',
            })
        )

        fn = f'grn_inf_num_cells_{num_cells}.csv'
        summary_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn))


def scalability_switchtfi():

    import scanpy.external as sce
    from switchtfi.fit import align_anndata_grn
    from switchtfi.weight_fitting import calculate_weights
    from switchtfi.pvalue_calculation import compute_corrected_pvalues, remove_insignificant_edges
    from switchtfi.tf_ranking import rank_tfs

    # Run cellrank inference on varying numbers of cells
    for num_cells in VARY_NUM_CELLS_NUM_CELLS:
        for num_edges in VARY_NUM_CELLS_NUM_EDGES:

            # Load the data and do basic processing
            adata = load_data(n_obs=num_cells)
            sc.pp.normalize_per_cell(adata)
            sc.pp.log1p(adata)

            # Load the corresponding GRN
            grn = load_grn(n_obs=num_cells, n_edges=num_edges, grn_inf_method=GRN_INF_METHOD_INPUT)

            # Runs step-wise analysis
            res_df_align, (adata_aligned, grn_aligned) = scalability_wrapper(
                function=align_anndata_grn,
                function_params={'adata': adata, 'grn': grn},
            )

            res_df_imputation, adata_imputed = scalability_wrapper(
                function=sce.pp.magic,
                function_params={
                    'adata': adata_aligned,
                    'name_list': 'all_genes',
                    'knn': 5,
                    'decay': 1,
                    'knn_max': None,
                    't': 1,
                    'n_pca': 100,
                    'solver': 'exact',
                    'knn_dist': 'euclidean',
                    'random_state': 42,
                    'n_jobs': 1,
                    'verbose': True,
                    'copy': True,
                },
            )

            res_df_weights, grn_weighted = scalability_wrapper(
                function=calculate_weights,
                function_params={
                    'adata': adata_imputed,
                    'grn': grn_aligned,
                    'layer_key': None,
                    'n_cell_pruning_params': None,
                    'clustering_obs_key': 'prog_off'
                },
            )

            res_df_pvalues, grn_pval = scalability_wrapper(
                function=compute_corrected_pvalues,
                function_params={
                    'adata': adata_aligned,
                    'grn': grn_weighted,
                    'method': 'wy',
                    'clustering_obs_key': 'prog_off',
                },
            )

            res_df_pruning, transition_grn = scalability_wrapper(
                function=remove_insignificant_edges,
                function_params={
                    'grn': grn_pval,
                    'alpha': 0.05,
                    'p_value_key': 'pvals_wy',
                },
            )

            res_df_tf_ranking, ranked_tfs = scalability_wrapper(
                function=rank_tfs,
                function_params={
                    'grn': transition_grn,
                    'centrality_measure': 'pagerank',
                },
            )

            res_dfs_sub = [
                res_df_align, res_df_imputation, res_df_weights, res_df_pvalues, res_df_pruning, res_df_tf_ranking
            ]
            res_df = pd.concat(res_dfs_sub, axis=0, ignore_index=True)

            res_df['n_cells'] = [num_cells] * 6
            res_df['n_edges_frac'] = [num_edges] * 6
            res_df['n_edges'] = [grn.shape[0]] * 6

            res_df['alg_step'] = ['align', 'impute', 'weight', 'pvalue', 'prune', 'rank_tfs']

            num_edges_str = str(num_edges).replace('.', '_')
            fn_fg = f'switchtfi_fine_grained_num_cells_{num_cells}_num_edges_{num_edges_str}.csv'
            res_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn_fg))

            summary_df = (
                res_df
                .drop(columns=['alg_step'])
                .groupby('n_cells', as_index=False)
                .agg({
                    'wall_time': 'sum',
                    'mem_peak_cpu': 'max', 'mem_avg_cpu': 'mean', 'samples_cpu': 'sum',
                    'mem_peak_gpu': 'max', 'mem_avg_gpu': 'mean', 'samples_gpu': 'sum',
                    'n_edges_frac': 'first',
                    'n_edges': 'first',
                })
            )

            fn = f'switchtfi_num_cells_{num_cells}_num_edges_{num_edges_str}.csv'
            summary_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn))


def scalability_cellrank():

    import scvelo as scv
    import cellrank as cr


    def compute_rna_velocity(data: sc.AnnData) -> sc.AnnData:

        sc.tl.pca(data)
        sc.pp.neighbors(data, n_pcs=30, n_neighbors=30, random_state=42)

        # Compute moments
        scv.pp.moments(data, n_pcs=None, n_neighbors=None)
        # Estimate model parameters
        scv.tl.recover_dynamics(data, n_jobs=None)  # TODO: increase n_jobs if velocity estimation takes overly long
        # Compute velocities
        scv.tl.velocity(data, mode='dynamical')

        return data


    def compute_rna_velo_transition_matrix(data: sc.AnnData) -> cr.kernels.Kernel:

        # Set up velocity kernel
        vk = cr.kernels.VelocityKernel(data)
        # Compute (cel-cell) transition matrix
        vk.compute_transition_matrix()

        return vk


    def identify_initial_terminal_states(cr_kernel: cr.kernels.Kernel) -> cr.estimators.GPCCA:

        # Initialize estimator
        gpcca = cr.estimators.GPCCA(cr_kernel)

        gpcca.compute_schur()

        use_cluster_info = True
        if use_cluster_info:
            gpcca.compute_macrostates(cluster_key='prog_off', n_states=2)
            gpcca.set_initial_states(states='prog')
            gpcca.set_terminal_states(states='off')
        else:
            # Cannot use cluster annotations since CellRank will crash
            gpcca.compute_macrostates()
            gpcca.predict_initial_states(allow_overlap=True)
            gpcca.predict_terminal_states(allow_overlap=True)

        return gpcca


    def estimate_fate_probabilities(cr_estimator: cr.estimators.GPCCA) -> cr.estimators.GPCCA:

        # Initial and terminal states must have been identified beforehand ..
        cr_estimator.compute_fate_probabilities()

        return cr_estimator


    def uncover_driver_genes(cr_estimator: cr.estimators.GPCCA) -> Tuple[pd.DataFrame, cr.estimators.GPCCA]:
        cr_estimator.compute_eigendecomposition()
        driver_genes = cr_estimator.compute_lineage_drivers(cluster_key='clusters')

        return driver_genes, cr_estimator


    # Warmup run to compile functions before the initial run
    adata_warmup = load_data(n_obs=200)
    sc.pp.normalize_per_cell(adata_warmup)
    sc.pp.log1p(adata_warmup)

    adata_warmup_velo = compute_rna_velocity(data=adata_warmup)
    velo_kernel = compute_rna_velo_transition_matrix(data=adata_warmup_velo)
    estimator = identify_initial_terminal_states(cr_kernel=velo_kernel)
    estimator_prob = estimate_fate_probabilities(cr_estimator=estimator)
    uncover_driver_genes(cr_estimator=estimator_prob)

    # Run cellrank inference on varying numbers of cells
    for num_cells in VARY_NUM_CELLS_NUM_CELLS:

        # Load the data and do basic processing
        adata = load_data(n_obs=num_cells)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

        # Runs step-wise analysis
        res_df_rna_velo, adata_rna_velo = scalability_wrapper(
            function=compute_rna_velocity,
            function_params={'data': adata},
        )

        res_df_trans_matrix, velocity_kernel = scalability_wrapper(
            function=compute_rna_velo_transition_matrix,
            function_params={'data': adata_rna_velo},
        )

        res_df_terminal_states, cr_estim = scalability_wrapper(
            function=identify_initial_terminal_states,
            function_params={'cr_kernel': velocity_kernel},
        )

        res_df_fate_probs, cr_estimator_fate_probs = scalability_wrapper(
            function=estimate_fate_probabilities,
            function_params={'cr_estimator': cr_estim},
        )

        res_df_driver_genes, _ = scalability_wrapper(
            function=uncover_driver_genes,
            function_params={'cr_estimator': cr_estimator_fate_probs},
        )

        res_dfs_sub = [
            res_df_rna_velo, res_df_trans_matrix, res_df_terminal_states, res_df_fate_probs, res_df_driver_genes
        ]
        res_df = pd.concat(res_dfs_sub, axis=0, ignore_index=True)

        res_df['n_cells'] = [num_cells] * 5

        res_df['alg_step'] = ['rna_velo', 'trans_matrix', 'terminal_states', 'fate_probs', 'driver_genes']

        fn_fg = f'cellrank_fine_grained_num_cells_{num_cells}.csv'
        res_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn_fg))

        summary_df = (
            res_df
            .drop(columns=['alg_step'])
            .groupby('n_cells', as_index=False)
            .agg({
                'wall_time': 'sum',
                'mem_peak_cpu': 'max', 'mem_avg_cpu': 'mean', 'samples_cpu': 'sum',
                'mem_peak_gpu': 'max', 'mem_avg_gpu': 'mean', 'samples_gpu': 'sum',
            })
        )

        fn = f'cellrank_num_cells_{num_cells}.csv'
        summary_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn))


def scalability_splicejac():

    import scvelo as scv
    import splicejac as sj


    def compute_hvgs_subset(data: sc.AnnData) -> sc.AnnData:

        # Set number of genes based on smallest cluster size
        min_cluster_size = data.obs['clusters'].value_counts().min()
        num_genes = min(data.shape[1], int(min_cluster_size * 0.9))

        # Compute highly variable genes
        sc.pp.highly_variable_genes(data)

        # Select exactly top num_genes by dispersion
        ranked = data.var.sort_values('dispersions_norm', ascending=False)
        top_genes = ranked.index[:num_genes]

        data_hvg = data[:, top_genes].copy()

        return data_hvg


    def infer_splicejac_grn(data: sc.AnnData) -> sc.AnnData:

        sj.tl.estimate_jacobian(
            data,
            n_top_genes=data.shape[1], # Pass to avoid spliceJAC error (default is 20)
            filter_and_norm=False  # No further gene filtering
        )

        return data


    def get_splicejac_transition_genes(data: sc.AnnData) -> Tuple[sc.AnnData, pd.DataFrame]:

        sc.tl.rank_genes_groups(data, 'clusters', method='t-test')

        sj.tl.transition_genes(data, 'prog', 'off')  # , top_DEG=num_genes, top_TG=num_genes)

        # Extract ranked list of genes, see splicejac git - plot_trans_genes()
        # Get splicejac transition weights
        splicejac_weights = data.uns['transitions']['prog' + '-' + 'off']['weights']
        genes = list(data.var_names)

        driver_genes = pd.DataFrame({'gene': genes, 'splicejac_weight': splicejac_weights})
        driver_genes.sort_values(by='splicejac_weight', ascending=False, inplace=True, ignore_index=True)

        return data, driver_genes


    # Run SpliceJAC inference on varying numbers of cells
    for num_cells in VARY_NUM_CELLS_NUM_CELLS:

        # Load the data and do basic processing
        adata = load_data(n_obs=num_cells)
        scv.pp.filter_genes(adata, min_shared_counts=20)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.obs['clusters'] = adata.obs['prog_off'].copy()

        # Runs step-wise analysis
        res_df_hvg, adata_hvg = scalability_wrapper(
            function=compute_hvgs_subset,
            function_params={'data': adata},
        )

        res_df_grn_inf, adata_grn = scalability_wrapper(
            function=infer_splicejac_grn,
            function_params={'data': adata_hvg},
        )

        res_df_transition, _ = scalability_wrapper(
            function=get_splicejac_transition_genes,
            function_params={'data': adata_grn},
        )

        res_dfs_sub = [
            res_df_hvg, res_df_grn_inf, res_df_transition
        ]
        res_df = pd.concat(res_dfs_sub, axis=0, ignore_index=True)

        res_df['n_cells'] = [num_cells] * 3

        res_df['alg_step'] = ['hvg_subset', 'grn_inf', 'transition']
        gpu_cols = ['mem_peak_gpu', 'mem_avg_gpu', 'samples_gpu']
        res_df[gpu_cols] = res_df[gpu_cols].astype('float64')

        res_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, f'splicejac_fine_grained_num_cells_{num_cells}.csv'))

        summary_df = (
            res_df
            .drop(columns=['alg_step'])
            .groupby('n_cells', as_index=False)
            .agg({
                'wall_time': 'sum',
                'mem_peak_cpu': 'max', 'mem_avg_cpu': 'mean', 'samples_cpu': 'sum',
                'mem_peak_gpu': 'max', 'mem_avg_gpu': 'mean', 'samples_gpu': 'sum',
            })
        )

        summary_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, f'splicejac_num_cells_{num_cells}.csv'))


def scalability_drivaer():

    import DrivAER as dv

    from validation.drivaer_workflow import get_tf_target_pdseries

    def drivaer_inference(data: sc.AnnData, grn: pd.DataFrame) -> pd.DataFrame:

        tf_to_target_list = get_tf_target_pdseries(grn=grn, tf_target_keys=('TF', 'target'))

        low_dim_rep, relevance, genes = dv.calc_relevance(
            count=data,
            pheno=data.obs['prog_off'],
            tf_targets=tf_to_target_list,
            min_targets=10,
            ae_type='nb-conddisp',
            epochs=50,
            early_stop=3,
            hidden_size=(8, 2, 8),
            verbose=True
        )

        driver_genes = pd.DataFrame({'gene': genes, 'relevance': relevance})
        driver_genes.sort_values(by='relevance', ascending=False, inplace=True, ignore_index=True)

        return driver_genes


    # Run DrivAER inference on varying numbers of cells
    for num_cells in VARY_NUM_CELLS_NUM_CELLS:
        for num_edges in VARY_NUM_CELLS_NUM_EDGES:

            # Load the data
            adata = load_data(n_obs=num_cells)

            # Load the corresponding GRN
            input_grn = load_grn(n_obs=num_cells, n_edges=num_edges, grn_inf_method=GRN_INF_METHOD_INPUT)

            # Run DrivAER analysis
            res_df, _ = scalability_wrapper(
                function=drivaer_inference,
                function_params={'data': adata, 'grn': input_grn},
                track_gpu=False,
            )

            res_df['n_cells'] = [num_cells, ]
            res_df['n_edges_frac'] = [num_edges, ]
            res_df['n_edges'] = [input_grn.shape[0], ]

            num_edges_str = str(num_edges).replace('.', '_')
            fn = f'drivaer_num_cells_{num_cells}_num_edges_{num_edges_str}.csv'
            res_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn))


def scalability_switchtfi_grn():

    import scanpy.external as sce
    from switchtfi.fit import align_anndata_grn
    from switchtfi.weight_fitting import calculate_weights
    from switchtfi.pvalue_calculation import compute_corrected_pvalues, remove_insignificant_edges
    from switchtfi.tf_ranking import rank_tfs

    for num_edges in VARY_NUM_EDGES_NUM_EDGES:
        for num_cells in VARY_NUM_EDGES_NUM_CELLS:

            # Load the data and do basic processing
            adata = load_data(n_obs=num_cells)
            sc.pp.normalize_per_cell(adata)
            sc.pp.log1p(adata)

            # Load the corresponding GRN
            grn = load_grn(n_obs=num_cells, n_edges=num_edges, grn_inf_method=GRN_INF_METHOD_INPUT)

            # Runs step-wise analysis
            res_df_align, (adata_aligned, grn_aligned) = scalability_wrapper(
                function=align_anndata_grn,
                function_params={'adata': adata, 'grn': grn},
            )

            res_df_imputation, adata_imputed = scalability_wrapper(
                function=sce.pp.magic,
                function_params={
                    'adata': adata_aligned,
                    'name_list': 'all_genes',
                    'knn': 5,
                    'decay': 1,
                    'knn_max': None,
                    't': 1,
                    'n_pca': 100,
                    'solver': 'exact',
                    'knn_dist': 'euclidean',
                    'random_state': 42,
                    'n_jobs': 1,
                    'verbose': True,
                    'copy': True,
                },
            )

            res_df_weights, grn_weighted = scalability_wrapper(
                function=calculate_weights,
                function_params={
                    'adata': adata_imputed,
                    'grn': grn_aligned,
                    'layer_key': None,
                    'n_cell_pruning_params': None,
                    'clustering_obs_key': 'prog_off'
                },
            )

            res_df_pvalues, grn_pval = scalability_wrapper(
                function=compute_corrected_pvalues,
                function_params={
                    'adata': adata_aligned,
                    'grn': grn_weighted,
                    'method': 'wy',
                    'clustering_obs_key': 'prog_off',
                },
            )

            res_df_pruning, transition_grn = scalability_wrapper(
                function=remove_insignificant_edges,
                function_params={
                    'grn': grn_pval,
                    'alpha': 0.05,
                    'p_value_key': 'pvals_wy',
                },
            )

            res_df_tf_ranking, ranked_tfs = scalability_wrapper(
                function=rank_tfs,
                function_params={
                    'grn': transition_grn,
                    'centrality_measure': 'pagerank',
                },
            )

            res_dfs_sub = [
                res_df_align, res_df_imputation, res_df_weights, res_df_pvalues, res_df_pruning, res_df_tf_ranking
            ]
            res_df = pd.concat(res_dfs_sub, axis=0, ignore_index=True)

            res_df['n_edges'] = [num_edges] * 6
            res_df['n_cells'] = [num_cells] * 6

            res_df['alg_step'] = ['align', 'impute', 'weight', 'pvalue', 'prune', 'rank_tfs']

            fn_fg = f'switchtfi_fine_grained_num_edges_{num_edges}_num_cells_{num_cells}.csv'
            res_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn_fg))

            summary_df = (
                res_df
                .drop(columns=['alg_step'])
                .groupby('n_cells', as_index=False)
                .agg({
                    'wall_time': 'sum',
                    'mem_peak_cpu': 'max', 'mem_avg_cpu': 'mean', 'samples_cpu': 'sum',
                    'mem_peak_gpu': 'max', 'mem_avg_gpu': 'mean', 'samples_gpu': 'sum',
                    'n_edges': 'first',
                })
            )

            fn = f'switchtfi_num_edges_{num_edges}_num_cells_{num_cells}.csv'
            summary_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn))


def scalability_drivaer_grn():

    import DrivAER as dv
    from validation.drivaer_workflow import get_tf_target_pdseries

    def drivaer_inference(data: sc.AnnData, grn: pd.DataFrame) -> pd.DataFrame:
        tf_to_target_list = get_tf_target_pdseries(grn=grn, tf_target_keys=('TF', 'target'))

        low_dim_rep, relevance, genes = dv.calc_relevance(
            count=data,
            pheno=data.obs['prog_off'],
            tf_targets=tf_to_target_list,
            min_targets=2,  # Set min targets to 2 here for fair comparison (cannot be 1)
            ae_type='nb-conddisp',
            epochs=50,
            early_stop=3,
            hidden_size=(8, 2, 8),
            verbose=True
        )

        driver_genes = pd.DataFrame({'gene': genes, 'relevance': relevance})
        driver_genes.sort_values(by='relevance', ascending=False, inplace=True, ignore_index=True)

        return driver_genes

    for num_edges in VARY_NUM_EDGES_NUM_EDGES:
        for num_cells in VARY_NUM_EDGES_NUM_CELLS:

            # Load the data and do basic processing
            adata = load_data(n_obs=num_cells)

            # Load the corresponding GRN
            input_grn = load_grn(n_obs=num_cells, n_edges=num_edges, grn_inf_method=GRN_INF_METHOD_INPUT)

            # Run DrivAER analysis
            res_df, _ = scalability_wrapper(
                function=drivaer_inference,
                function_params={'data': adata, 'grn': input_grn},
                track_gpu=False,
            )

            res_df['n_edges'] = [num_edges]
            res_df['n_cells'] = [num_cells]

            fn = f'drivaer_num_edges_{num_edges}_num_cells_{num_cells}.csv'
            res_df.to_csv(os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn))


def aggregate_results():

    save_path = os.path.join(SAVE_PATH, 'aggregated_results')
    os.makedirs(save_path, exist_ok=True)

    methods = ['grn_inf', 'switchtfi', 'cellrank', 'splicejac', 'drivaer']

    # Aggregate: --- Vary number of cells, vary number of edges (GRN: small, medium, large)
    res_dfs = []
    res_dfs_fine_grained = []
    for method in methods:

        print('\n###############################################################')
        print('###### ', method, ' ######')
        print('###############################################################')

        res_dfs_method = []
        res_dfs_method_fine_grained = []

        if method in {'switchtfi', 'drivaer'}:
            # These methods vary both num_cells and num_edges
            for num_cells in VARY_NUM_CELLS_NUM_CELLS:
                for num_edges in VARY_NUM_CELLS_NUM_EDGES:

                    num_edges_str = str(num_edges).replace('.', '_')
                    fn = f'{method}_num_cells_{num_cells}_num_edges_{num_edges_str}.csv'
                    fp = os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn)

                    try:
                        res_df_current = pd.read_csv(fp, index_col=0)
                        res_dfs_method.append(res_df_current)
                    except FileNotFoundError:
                        print(f'File {fn} not found')
                        continue

                    if method == 'switchtfi':
                        fn_fg = f'{method}_fine_grained_num_cells_{num_cells}_num_edges_{num_edges_str}.csv'
                        fp_fg = os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn_fg)
                        res_df_current_fine_grained = pd.read_csv(fp_fg, index_col=0)
                        res_dfs_method_fine_grained.append(res_df_current_fine_grained)

        else:
            # These methods vary only num_cells
            for num_cells in VARY_NUM_CELLS_NUM_CELLS:

                fn = f'{method}_num_cells_{num_cells}.csv'
                fp = os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn)

                try:
                    res_df_current = pd.read_csv(fp, index_col=0)
                    res_dfs_method.append(res_df_current)
                except FileNotFoundError:
                    print(f'File {fn} not found')
                    continue

                fn_fg = f'{method}_fine_grained_num_cells_{num_cells}.csv'
                fp_fg = os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn_fg)
                res_df_current_fine_grained = pd.read_csv(fp_fg, index_col=0)
                res_dfs_method_fine_grained.append(res_df_current_fine_grained)

        if res_dfs_method:
            res_df_method = pd.concat(res_dfs_method, axis=0, ignore_index=True)
            res_df_method['method'] = method
            res_dfs.append(res_df_method)

            fn = f'{method}_num_cells_num_edges.csv'
            res_df_method.to_csv(os.path.join(save_path, fn))

            print(res_df_method)

        if res_dfs_method_fine_grained:
            res_df_method_fine_grained = pd.concat(res_dfs_method_fine_grained, axis=0, ignore_index=True)
            res_df_method_fine_grained['method'] = method
            res_dfs_fine_grained.append(res_df_method_fine_grained)

            fn_fg = f'{method}_fine_grained_num_cells_num_edges.csv'
            res_df_method_fine_grained.to_csv(os.path.join(save_path, fn_fg))

            print(res_df_method_fine_grained)


    res_df = pd.concat(res_dfs, axis=0, ignore_index=True)
    fn = f'aggregated_results_num_cells_num_edges.csv'
    res_df.to_csv(os.path.join(save_path, fn))

    res_dfs_fine_grained = pd.concat(res_dfs_fine_grained, axis=0, ignore_index=True)
    fn_fg = f'aggregated_results_fine_grained_num_cells_num_edges.csv'
    res_dfs_fine_grained.to_csv(os.path.join(save_path, fn_fg))

    # Aggregate: --- Vary number of edges, vary number of cells (low, medium, high)
    methods_num_edges = ['switchtfi', 'drivaer']

    res_dfs = []
    res_dfs_fine_grained = []
    for method in methods_num_edges:

        print('\n###############################################################')
        print('###### ', method, ' ######')
        print('###############################################################')

        res_dfs_method = []
        res_dfs_method_fine_grained = []

        for num_edges in VARY_NUM_EDGES_NUM_EDGES:
            for num_cells in VARY_NUM_EDGES_NUM_CELLS:

                fn = f'{method}_num_edges_{num_edges}_num_cells_{num_cells}.csv'
                fp = os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn)

                try:
                    res_df_current = pd.read_csv(fp, index_col=0)
                    res_dfs_method.append(res_df_current)
                except FileNotFoundError:
                    print(f'File {fn} not found')
                    continue

                if method != 'drivaer':

                    fn_fg = f'{method}_fine_grained_num_edges_{num_edges}_num_cells_{num_cells}.csv'
                    fp_fg = os.path.join(SAVE_PATH, INTERM_RES_SUBDIR, fn_fg)
                    res_df_current_fine_grained = pd.read_csv(fp_fg, index_col=0)
                    res_dfs_method_fine_grained.append(res_df_current_fine_grained)

        if res_dfs_method:

            res_df_method = pd.concat(res_dfs_method, axis=0, ignore_index=True)
            res_df_method['method'] = method
            res_dfs.append(res_df_method)

            fn = f'{method}_num_edges_num_cells.csv'
            res_df_method.to_csv(os.path.join(save_path, fn))

            print(res_df_method)

        if res_dfs_method_fine_grained:

            res_df_method_fine_grained = pd.concat(res_dfs_method_fine_grained, axis=0, ignore_index=True)
            res_df_method_fine_grained['method'] = method
            res_dfs_fine_grained.append(res_df_method_fine_grained)

            fn_fg = f'{method}_fine_grained_num_edges_num_cells.csv'
            res_df_method_fine_grained.to_csv(os.path.join(save_path, fn_fg))

            print(res_df_method_fine_grained)

    res_df = pd.concat(res_dfs, axis=0, ignore_index=True)
    fn = f'aggregated_results_num_edges_num_cells.csv'
    res_df.to_csv(os.path.join(save_path, fn))

    res_dfs_fine_grained = pd.concat(res_dfs_fine_grained, axis=0, ignore_index=True)
    fn_fg = f'aggregated_results_fine_grained_num_edges_num_cells.csv'
    res_dfs_fine_grained.to_csv(os.path.join(save_path, fn_fg))


if __name__ == '__main__':

    # ### Perform scalability analysis

    parser = argparse.ArgumentParser(description='Run scalability analysis for selected method.')

    parser.add_argument(
        '-m', '--method',
        type=str,
        choices=[
            'data',
            'grn_inf', 'switchtfi', 'cellrank', 'splicejac', 'drivaer',
            'switchtfi_grn', 'drivaer_grn',
            'results'
        ],
        default='switchtfi',
        help=(
            'Method for which to run the analysis for: '
            '"grn_inf", "switchtfi", "splicejac", "drivaer", "switchtfi_grn", "drivaer_grn", or "results"'
        ),
    )

    parser.add_argument(
        '-vncnc', '--vary-num-cells-num-cells',
        type=int,
        default=None,
        help='Number of cells to include. Optional, overwrites the global VARY_NUM_CELLS_NUM_CELLS variable.'
    )

    parser.add_argument(
        '-vncne', '--vary-num-cells-num-edges',
        type=float,
        default=None,
        help='Fraction of edges to include. Optional, overwrites the global VARY_NUM_CELLS_NUM_EDGES variable.'
    )

    parser.add_argument(
        '-vnene', '--vary-num-edges-num-edges',
        type=int,
        default=None,
        help='Number of edges to include. Optional, overwrites the global VARY_NUM_EDGES_NUM_EDGES variable.'
    )

    parser.add_argument(
        '-vnenc', '--vary-num-edges-num-cells',
        type=int,
        default=None,
        help='Number of cells to include. Optional, overwrites the global VARY_NUM_EDGES_NUM_CELLS variable.'
    )

    args = parser.parse_args()

    m = args.method
    vncnc = args.vary_num_cells_num_cells
    vncne = args.vary_num_cells_num_edges
    vnene = args.vary_num_edges_num_edges
    vnenc = args.vary_num_edges_num_cells

    if m in {'grn_inf', 'switchtfi', 'cellrank', 'splicejac', 'drivaer', 'switchtfi_grn', 'drivaer_grn'}:
        p = SAVE_PATH / 'data'
        if (
                not p.exists()
                or not p.is_dir()
                or not (any(f.name[0] != '.' for f in p.iterdir()))
        ):
            raise RuntimeError(f'Run data generation before running "{m}"')

    if m in {'switchtfi', 'drivaer', 'switchtfi_grn', 'drivaer_grn'}:
        p = SAVE_PATH / 'grn_inf'
        if (
                not p.exists()
                or not p.is_dir()
                or not (any(f.name[0] != '.' for f in p.iterdir()))
        ):
            raise RuntimeError(f'Run GRN inference before running "{m}"')

    if vncnc is not None:
        VARY_NUM_CELLS_NUM_CELLS = [vncnc, ]

    if vncne is not None:
        VARY_NUM_CELLS_NUM_EDGES = [vncne, ]

    if vnene is not None:
        VARY_NUM_EDGES_NUM_EDGES = [vnene, ]

    if vnenc is not None:
        VARY_NUM_EDGES_NUM_CELLS = [vnenc, ]

    if m == 'data':
        process_data()
    elif m == 'grn_inf':
        scalability_grn_inf()
    elif m == 'switchtfi':
        scalability_switchtfi()
    elif m == 'cellrank':
        scalability_cellrank()
    elif m == 'splicejac':
        scalability_splicejac()
    elif m == 'drivaer':
        scalability_drivaer()
    elif m == 'switchtfi_grn':
        scalability_switchtfi_grn()
    elif m == 'drivaer_grn':
        scalability_drivaer_grn()
    else:  # 'results'
        aggregate_results()

    print('done')


