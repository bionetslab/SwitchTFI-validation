

# Todo:
#  - Comparison strategy:
#    - Define grid of n_cells
#    - Fix n_genes to sensible value, e.g. 10000 (typical number of genes that remain after peprocessing)
#    - For each n_cells run grn_inf, cellrank, spliceJAC, DrivAER, SwitchTFI
#      (for DrivAER, SwitchTFI use GRN inferred for this n_cells)
#    - SpliceJAC can only be run on # cells in cluster many genes -> run cellrank, DrivAER, SwitchTFI also on this many genes
#    - Additional study: SwitchTFI, DrivAER scales in n-edges (GRN-size), pick fixed number of cells (e.g. 10000) and remove random edges
#  - Run everything on HPC

# Todo: three comparisons
#  (1) Vary number of cells, fix number of genes (except for spliceJAC), use output GRNs
#  (2) Vary number of cells and number of genes (for comparison to spliceJAC)
#  (3) Compare SwitchTFI and DrivAER on GRNs of different size for fixed n cells and n genes


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


if Path('/home/woody/iwbn/iwbn107h').is_dir():
    SAVE_PATH = Path('/home/woody/iwbn/iwbn107h/scalability')
else:
    SAVE_PATH = Path.cwd().parent / 'results/05_revision/scalability'
os.makedirs(SAVE_PATH, exist_ok=True)

TEST = True

NUM_GENES = 10000 if not TEST else 1000
NUM_CELLS = [100, 500, 1000, 5000, 10000, 50000, 100000] if not TEST else [100, 200, 300]


def process_data():

    import cellrank as cr

    save_path = SAVE_PATH / 'data'
    os.makedirs(save_path, exist_ok=True)

    print(save_path)

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
    adata = cr.datasets.reprogramming_morris(os.path.join(save_path, 'reprogramming_morris.h5ad'), subset='full')

    # Subset to the top 10,000 hvg genes
    adata_proc = adata.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata_proc)
    sc.pp.highly_variable_genes(adata_proc, n_top_genes=NUM_GENES)

    adata_hvg = adata[:, adata_proc.var['highly_variable']].copy()

    # Save AnnData, individual data matrices and relevant annotations
    adata_hvg.write_h5ad(os.path.join(save_path, 'reprogramming_morris_hvg.h5ad'))
    x_unspliced = adata_hvg.layers['unspliced']
    x_spliced = adata_hvg.layers['spliced']
    sp.save_npz(os.path.join(save_path, f'x_unspliced.npz'), x_unspliced)
    sp.save_npz(os.path.join(save_path, f'x_spliced.npz'), x_spliced)

    np.save(os.path.join(save_path, f'cell_names.npy'), adata_hvg.obs_names.to_numpy())
    np.save(os.path.join(save_path, f'gene_names.npy'), adata_hvg.var_names.to_numpy())


def load_data(n_obs: int, seed: int = 42) -> sc.AnnData:

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

    # Create anndata
    adata = sc.AnnData(X=x_spliced)
    adata.obs_names = cell_names
    adata.var_names = gene_names
    adata.layers['unspliced'] = x_unspliced
    adata.layers['spliced'] = x_spliced

    # Subsample to the desired number of cells
    np.random.seed(seed)
    n_total = adata.n_obs
    idx = np.random.choice(n_total, size=n_obs, replace=False)
    adata_sub = adata[idx, :].copy()

    # Add progenitor offspring annotations
    if n_obs % 2 != 0:
        n_prog = n_obs // 2
        n_off = n_obs - n_prog
    else:
        n_prog = int(n_obs / 2)
        n_off = int(n_obs / 2)

    prog_off_anno = ['prog'] * n_prog + ['off'] * n_off
    adata_sub.obs['prog_off'] = prog_off_anno

    return adata_sub


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


# def get_gpu_memory_mb() -> int:
#     """Get current memory usage for GPU 0 in MB using nvidia-smi."""
#     try:
#         output = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
#             stderr=subprocess.DEVNULL
#         )
#         mem = int(output.decode().split('\n')[0].strip())
#         return mem
#     except Exception as e:
#         print(f'GPU memory tracking failed with error:\n{e}')
#         return 0  # fallback if nvidia-smi fails or no GPU present


# def track_memory_gpu(interval=0.1):
#     """
#     Tracks GPU 0 memory usage over time in a background thread.
#     Returns (samples_list, stop_event, thread).
#     """
#     memory_samples = [get_gpu_memory_mb()]
#     stop_event = threading.Event()
#
#     def poll():
#         while not stop_event.is_set():
#             mem = get_gpu_memory_mb()
#             memory_samples.append(mem)
#             stop_event.wait(interval)
#
#     thread = threading.Thread(target=poll)
#     thread.start()
#
#     return memory_samples, stop_event, thread


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
    memory_samples_cpu, stop_event_cpu, tracker_thread_cpu = track_memory_cpu(interval=0.1)
    if track_gpu:
        memory_samples_gpu, stop_event_gpu, tracker_thread_gpu = track_memory_gpu(interval=0.1)

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


def scalability_grn_inf():

    from arboreto.algo import grnboost2

    # Define path where results are saved
    save_path = SAVE_PATH / 'grn_inf'
    os.makedirs(save_path, exist_ok=True)

    # Run GRN inference varying numbers of cells
    res_dfs = []
    for i, n in enumerate(NUM_CELLS):

        # Load the data
        adata = load_data(n_obs=n)
        adata_df = adata.to_df(layer=None)

        # Define the 1500 first gene names as TFs
        n_tfs = min(1500, adata_df.shape[0])
        tf_names = adata.var_names.tolist()[0:n_tfs]

        fn_kwargs = {
            'expression_data': adata_df,
            'gene_names': None,
            'tf_names': tf_names,
            'seed': 42,
            'verbose': True,
        }

        res_df, grn = scalability_wrapper(
            function=grnboost2,
            function_params=fn_kwargs,
            track_gpu=False,
            res_dir=None,
            res_filename=None,
        )

        res_df['n_cells'] = [n] * res_df.shape[0]

        res_dfs.append(res_df)

        grn.to_csv(os.path.join(save_path, f'grn_num_cells_{n}.csv'))

        res_df_joint = pd.concat(res_dfs, axis=0, ignore_index=True)

        res_df_joint.to_csv(os.path.join(SAVE_PATH, f'grn_inf.csv'))

        print(res_df_joint)


def scalability_switchtfi():

    import sys
    sys.path.append(os.path.abspath('..'))
    from switchtfi.fit import align_anndata_grn
    from switchtfi.weight_fitting import calculate_weights
    from switchtfi.pvalue_calculation import compute_corrected_pvalues, remove_insignificant_edges
    from switchtfi.tf_ranking import rank_tfs

    # Run cellrank inference on varying numbers of cells
    res_dfs = []
    for i, n in enumerate(NUM_CELLS):

        # Load the data
        adata = load_data(n_obs=n)

        # Load the corresponding GRN
        grn_path = os.path.join(SAVE_PATH, 'grn_inf', f'grn_num_cells_{n}.csv')
        grn_num_cells = pd.read_csv(grn_path, index_col=0)
        grn_num_cells[['TF', 'target']] = grn_num_cells[['TF', 'target']].astype(str)

        # Runs step-wise analysis
        res_df_align, (adata_aligned, grn_aligned) = scalability_wrapper(
            function=align_anndata_grn,
            function_params={'adata': adata, 'grn': grn_num_cells},
        )

        # Todo: imputation

        res_df_weights, grn_weighted = scalability_wrapper(
            function=calculate_weights,
            function_params={
                'adata': adata_aligned,
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

        res_df_subs = [
            res_df_align, res_df_weights, res_df_pvalues, res_df_pruning, res_df_tf_ranking
        ]
        res_df_sub = pd.concat(res_df_subs, axis=0, ignore_index=True)

        res_df_sub['n_cells'] = [n] * 5

        res_df_sub['alg_step'] = ['align', 'weight', 'pvalue', 'pruning', 'tf_ranking']

        res_dfs.append(res_df_sub)

        res_df = pd.concat(res_dfs, axis=0, ignore_index=True)

        res_df.to_csv(os.path.join(SAVE_PATH, 'switchtfi_fine_grained.csv'))

        summary_df = (
            res_df
            .drop(columns=['alg_step'])
            .groupby('n_cells', as_index=False)
            .sum(min_count=1)
        )

        summary_df.to_csv(os.path.join(SAVE_PATH, 'switchtfi.csv'))

        print(res_df)

        print(summary_df)


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

        # Cannot use arbitrary annotations since CellRank will crash
        # gpcca.compute_macrostates(cluster_key='prog_off', n_states=2)
        # gpcca.set_initial_states(states='prog')
        # gpcca.set_terminal_states(states='off')

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

    adata_warmup_velo = compute_rna_velocity(data=adata_warmup)
    velo_kernel = compute_rna_velo_transition_matrix(data=adata_warmup_velo)
    estimator = identify_initial_terminal_states(cr_kernel=velo_kernel)
    estimator_prob = estimate_fate_probabilities(cr_estimator=estimator)
    uncover_driver_genes(cr_estimator=estimator_prob)

    # Run cellrank inference on varying numbers of cells
    res_dfs = []

    for i, n in enumerate(NUM_CELLS):

        # Load the data and do basic processing
        adata = load_data(n_obs=n)
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

        res_df_subs = [
            res_df_rna_velo, res_df_trans_matrix, res_df_terminal_states, res_df_fate_probs, res_df_driver_genes
        ]
        res_df_sub = pd.concat(res_df_subs, axis=0, ignore_index=True)

        res_df_sub['n_cells'] = [n] * 5

        res_df_sub['alg_step'] = ['rna_velo', 'trans_matrix', 'terminal_states', 'fate_probs', 'driver_genes']

        res_dfs.append(res_df_sub)

        res_df = pd.concat(res_dfs, axis=0, ignore_index=True)

        res_df.to_csv(os.path.join(SAVE_PATH, 'cellrank_fine_grained.csv'))

        summary_df = (
            res_df
            .drop(columns=['alg_step'])
            .groupby('n_cells', as_index=False)
            .sum(min_count=1)
        )

        summary_df.to_csv(os.path.join(SAVE_PATH, 'cellrank.csv'))

        print(res_df)

        print(summary_df)


def scalability_splicejac():

    import scvelo as scv
    import splicejac as sj


    def compute_hvgs_subset(data: sc.AnnData) -> sc.AnnData:

        # Set number of genes based on smallest cluster size
        min_cluster_size = data.obs['clusters'].value_counts().min()
        num_genes = min(data.shape[1], int(min_cluster_size * 0.9))

        # Compute highly variable genes
        sc.pp.highly_variable_genes(data, n_top_genes=num_genes)

        data_hvg = adata[:, data.var['highly_variable']].copy()

        return data_hvg


    def compute_rna_velocity(data: sc.AnnData) -> sc.AnnData:

        # Compute velocities
        scv.tl.velocity(data)
        scv.tl.velocity_graph(data)

        data.uns['neighbors']['distances'] = data.obsp['distances']
        data.uns['neighbors']['connectivities'] = data.obsp['connectivities']

        return data


    def infer_splicejac_grn(data: sc.AnnData) -> sc.AnnData:

        sj.tl.estimate_jacobian(data, filter_and_norm=False)  # No further gene filtering here

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
    res_dfs = []
    for i, n in enumerate(NUM_CELLS):

        # Load the data and do basic processing
        adata = load_data(n_obs=n)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.obs['clusters'] = adata.obs['prog_off'].copy()

        # Runs step-wise analysis
        res_df_hvg, adata_hvg_subset = scalability_wrapper(
            function=compute_hvgs_subset,
            function_params={'data': adata},
        )

        res_df_rna_velo, adata_rna_velo = scalability_wrapper(
            function=compute_rna_velocity,
            function_params={'data': adata_hvg_subset},
        )

        res_df_grn_inf, adata_grn = scalability_wrapper(
            function=infer_splicejac_grn,
            function_params={'data': adata_rna_velo},
        )

        res_df_transition, _ = scalability_wrapper(
            function=get_splicejac_transition_genes,
            function_params={'data': adata_grn},
        )

        res_df_subs = [
            res_df_hvg, res_df_rna_velo, res_df_grn_inf, res_df_transition
        ]
        res_df_sub = pd.concat(res_df_subs, axis=0, ignore_index=True)

        res_df_sub['n_cells'] = [n] * 4

        res_df_sub['alg_step'] = ['hvg_subset', 'rna_velo', 'grn_inf', 'transition']

        res_dfs.append(res_df_sub)

        res_df = pd.concat(res_dfs, axis=0, ignore_index=True)

        gpu_cols = ['mem_peak_gpu', 'mem_avg_gpu', 'samples_gpu']
        res_df[gpu_cols] = res_df[gpu_cols].astype('float64')

        res_df.to_csv(os.path.join(SAVE_PATH, 'splicejac_fine_grained.csv'))

        summary_df = (
            res_df
            .drop(columns=['alg_step'])
            .groupby('n_cells', as_index=False)
            .sum(min_count=1)
        )

        summary_df.to_csv(os.path.join(SAVE_PATH, 'splicejac.csv'))

        print(res_df)

        print(summary_df)


def scalability_drivaer():

    import DrivAER as dv

    from drivaer_workflow import get_tf_target_pdseries

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
    res_dfs = []
    for i, n in enumerate(NUM_CELLS):

        # Load the data
        adata = load_data(n_obs=n)

        # Load the corresponding GRN
        grn_path = os.path.join(SAVE_PATH, 'grn_inf', f'grn_num_cells_{n}.csv')
        grn_num_cells = pd.read_csv(grn_path, index_col=0)
        grn_num_cells[['TF', 'target']] = grn_num_cells[['TF', 'target']].astype(str)

        # Runs DrivAER analysis
        current_res_df, _ = scalability_wrapper(
            function=drivaer_inference,
            function_params={'data': adata, 'grn': grn_num_cells},
            track_gpu=True,
        )

        current_res_df['n_cells'] = [n]

        res_dfs.append(current_res_df)

        res_df = pd.concat(res_dfs, axis=0, ignore_index=True)

        res_df.to_csv(os.path.join(SAVE_PATH, 'drivaer.csv'))


def scalability_switchtfi_grn():
    # Todo
    pass


def scalability_drivaer_grn():
    # Todo
    pass

if __name__ == '__main__':

    # ### Perform scalability analysis

    parser = argparse.ArgumentParser(description='Run scalability analysis for selected method.')

    parser.add_argument(
        '-m', '--method',
        type=str,
        choices=['data', 'grn_inf', 'switchtfi', 'cellrank', 'splicejac', 'drivaer', 'switchtfi_grn', 'drivaer_grn'],
        default='switchtfi',
        help=(
            'Method for which to run the analysis for: '
            '"grn_inf", "switchtfi", "splicejac", "drivaer", "switchtfi_grn", or "drivaer_grn"'
        ),
    )

    args = parser.parse_args()

    if args.method in {'grn_inf', 'switchtfi', 'cellrank', 'splicejac', 'drivaer', 'switchtfi_grn', 'drivaer_grn'}:
        p = SAVE_PATH / 'data'
        if (
                not p.exists()
                or not p.is_dir()
                or not (any(f.name[0] != '.' for f in p.iterdir()))
        ):
            raise RuntimeError(f'Run data generation before running "{args.method}"')

    if args.method in {'switchtfi', 'drivaer', 'switchtfi_grn', 'drivaer_grn'}:
        p = SAVE_PATH / 'grn_inf'
        if (
                not p.exists()
                or not p.is_dir()
                or not (any(f.name[0] != '.' for f in p.iterdir()))
        ):
            raise RuntimeError(f'Run GRN inference before running "{args.method}"')

    if args.method == 'data':
        process_data()
    elif args.method == 'grn_inf':
        scalability_grn_inf()
    elif args.method == 'switchtfi':
        scalability_switchtfi()
    elif args.method == 'cellrank':
        scalability_cellrank()
    elif args.method == 'splicejac':
        scalability_splicejac()
    elif args.method == 'drivaer':
        scalability_drivaer()
    elif args.method == 'switchtfi_grn':
        scalability_switchtfi_grn()
    else:  # 'drivaer_grn'
        scalability_drivaer_grn()


    print('done')

