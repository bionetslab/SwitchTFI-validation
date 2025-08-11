

# Todo:
#  - Comparison strategy:
#    - Define grid of n_cells
#    - Fix n_genes to sensible value, e.g. 10000 (typical number of genes that remain after peprocessing)
#    - For each n_cells run grn_inf, cellrank, spliceJAC, DrivAER, SwitchTFI
#      (for DrivAER, SwitchTFI use GRN inferred for this n_cells)
#    - SpliceJAC can only be run on # cells in cluster many genes -> run cellrank, DrivAER, SwitchTFI also on this many genes
#    - Additional study: SwitchTFI, DrivAER scales in n-edges (GRN-size), pick fixed number of cells (e.g. 10000) and remove random edges
#  - Run everything on HPC


import os
import argparse
import time
import psutil
import threading
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from typing import Callable, Dict, Tuple, Union, Any


SAVE_PATH = Path.cwd().parent / 'results/05_revision/scalability'
os.makedirs(SAVE_PATH, exist_ok=True)

NUM_CELLS_MAX = 200000
NUM_GENES = 10000

NUM_CELLS = [1000, 5000, 10000, 50000, 100000, 200000]


def generate_data():

    import scvelo as scv

    save_path = SAVE_PATH / 'data'
    os.makedirs(save_path, exist_ok=True)

    # Generate data
    n_obs = NUM_CELLS_MAX
    n_vars = NUM_GENES

    simdata = scv.datasets.simulation(n_obs=n_obs, n_vars=n_vars, random_seed=42)

    # Save AnnData object
    simdata.write_h5ad(os.path.join(save_path, 'simdata.h5ad'))

    # Save individual data matrices and relevant annotations
    np.save(os.path.join(save_path, 'unspliced.npy'), simdata.layers['unspliced'])
    np.save(os.path.join(save_path, 'spliced.npy'), simdata.layers['spliced'])
    np.save(os.path.join(save_path, 'cell_names.npy'), simdata.obs_names.to_numpy())
    np.save(os.path.join(save_path, 'gene_names.npy'), simdata.var_names.to_numpy())


def load_data() -> sc.AnnData:

    save_path = SAVE_PATH / 'data'

    if not (save_path / 'simdata.h5ad').exists():
        raise RuntimeError(
            f"Missing expected file 'simdata.h5ad'. Make sure generate_data() has been run first."
        )

    # Load npy files to avoid errors caused by incompatible Scanpy versions
    x_unspliced = np.load(os.path.join(save_path, 'unspliced.npy'))
    x_spliced = np.load(os.path.join(save_path, 'spliced.npy'))
    cell_names = np.load(os.path.join(save_path, 'cell_names.npy'), allow_pickle=True)
    gene_names = np.load(os.path.join(save_path, 'gene_names.npy'), allow_pickle=True)

    simdata = sc.AnnData(X=x_spliced)

    simdata.obs_names = cell_names
    simdata.var_names = gene_names

    simdata.layers['unspliced'] = x_unspliced
    simdata.layers['spliced'] = x_spliced

    return simdata


def add_prog_off_annotations(simdata: sc.AnnData) -> sc.AnnData:

    n_obs = simdata.n_obs

    # Add progenitor offspring annotations
    if n_obs % 2 != 0:
        n_prog = n_obs // 2
        n_off = n_obs - n_prog
    else:
        n_prog = int(n_obs / 2)
        n_off = int(n_obs / 2)

    prog_off_anno = ['prog'] * n_prog + ['off'] * n_off
    simdata.obs['prog_off'] = prog_off_anno

    return  simdata


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
        function_params: Dict[str, Any],
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

        function_output = function(**function_params)

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

    # Load the simulated data
    simdata = load_data()
    simdata_df = simdata.to_df(layer=None)

    # Define the 1500 gene names as TFs
    tf_names = simdata.var_names.tolist()[0:1500]

    # Run GRN inference varying numbers of cells
    res_dfs = []
    for n in NUM_CELLS:

        # Subset the data
        simdata_df_subset = simdata_df.iloc[0:n, :].copy()

        fn_kwargs = {
            'expression_data': simdata_df_subset,
            'gene_names': None,
            'tf_names': tf_names,
            'seed': 42,
            'verbose': False,
        }

        res_df, grn = scalability_wrapper(
            function=grnboost2,
            function_params=fn_kwargs,
            track_gpu=False,
            res_dir=None,
            res_filename=None,
        )

        res_dfs.append(res_df)

        grn.to_csv(os.path.join(save_path, f'grn_num_cells_{n}.csv'))

    res_df = pd.concat(res_dfs, axis=0, ignore_index=True)
    res_df.index=NUM_CELLS
    res_df.index.name = 'num_cells'

    res_df.to_csv(os.path.join(SAVE_PATH, f'grn_inf.csv'))

    print(res_df)


def scalability_switchtfi():
    pass


def scalability_cellrank():
    pass


def scalability_splicejac():
    pass


def scalability_drivaer():
    pass


if __name__ == '__main__':

    preliminary = True

    # ### Generate the simulated data
    if preliminary:

        generate_data()

    # ### Perform scalability analysis
    else:

        parser = argparse.ArgumentParser(description='Run scalability analysis for selected method.')

        parser.add_argument(
            '-m', '--method',
            type=str,
            choices=['grn_inf', 'switchtfi', 'cellrank', 'splicejac', 'drivaer'],
            default='switchtfi',
            help='Method for which to run the analysis for: "grn_inf", "switchtfi", "splicejac", or "drivaer"'
        )

        args = parser.parse_args()

        if args.method in {'switchtfi', 'drivaer'}:
            p = SAVE_PATH / 'grn_inf'
            if (
                    not p.exists()
                    or not p.is_dir()
                    or not (any(f.name[0] != '.' for f in p.iterdir()))
            ):
                raise RuntimeError(f'Run GRN inference before running "{args.method}"')


        if args.method == 'grn_inf':
            scalability_grn_inf()
        elif args.method == 'switchtfi':
            scalability_switchtfi()
        elif args.method == 'cellrank':
            scalability_cellrank()
        elif args.method == 'splicejac':
            scalability_splicejac()
        else:
            scalability_drivaer()


    print('done')

