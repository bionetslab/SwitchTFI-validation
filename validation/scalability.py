

# Todo:
#  - Define grid of (n_cells, n_genes)
#  - For each grid entry run grn_inf, cellrank , and splicejac
#  - For a selection of GRN sizes (n_edges for switchtfi, n_tfs for drivaer; or just n_edges) run switchtfi and drivaer
#  - Sensible Track runtime, cpu time (are any of them parallelized?), memory
#  - Run on HPC

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


SAVE_PATH = Path.cwd().parent / 'results/05_revision/scalability'
os.makedirs(SAVE_PATH, exist_ok=True)

NUM_CELLS_MAX = 50  # 200000
NUM_GENES = 100  # 10000

NUM_CELLS = [20, 50]  # [1000, 5000, 10000, 50000, 100000, 200000]


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


def track_memory(interval=0.1):
    """
    Tracks total memory (RSS) of the current process + children.
    Returns a list of memory samples (in MB).
    """

    memory_samples = []
    process = psutil.Process(os.getpid())
    stop_event = threading.Event()

    def poll():
        while not stop_event.is_set():
            try:
                children = process.children(recursive=True)
                all_procs = [process] + children
                total_mem = sum(p.memory_info().rss for p in all_procs if p.is_running()) / 1024**2
                memory_samples.append(total_mem)
            except psutil.NoSuchProcess:
                pass
            time.sleep(interval)

    thread = threading.Thread(target=poll)
    thread.start()
    return memory_samples, stop_event, thread


def get_gpu_memory_mb() -> int:
    """Get current memory usage for GPU 0 in MB using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            stderr=subprocess.DEVNULL
        )
        mem = int(output.decode().split('\n')[0].strip())
        return mem
    except Exception as e:
        print(f'GPU memory tracking failed with error:\n{e}')
        return 0  # fallback if nvidia-smi fails or no GPU present


def track_memory_gpu(interval=0.1):
    """
    Tracks GPU 0 memory usage over time in a background thread.
    Returns (samples_list, stop_event, thread).
    """
    memory_samples = []
    stop_event = threading.Event()

    def poll():
        while not stop_event.is_set():
            mem = get_gpu_memory_mb()
            memory_samples.append(mem)
            time.sleep(interval)

    thread = threading.Thread(target=poll)
    thread.start()

    return memory_samples, stop_event, thread


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
    wall_times = []
    mem_peaks_cpu = []
    mem_peaks_gpu = []
    mem_avgs_cpu = []
    mem_avgs_gpu = []

    for n in NUM_CELLS:

        # Subset the data
        simdata_df_subset = simdata_df.iloc[0:n, :].copy()

        # Start memory tracking
        memory_samples_cpu, stop_event_cpu, tracker_thread_cpu = track_memory(interval=0.01)
        memory_samples_gpu, stop_event_gpu, tracker_thread_gpu = track_memory_gpu(interval=0.01)

        wall_start = time.perf_counter()

        # Run GRN inference
        grn = grnboost2(
            expression_data=simdata_df_subset,
            gene_names=None,
            tf_names=tf_names,
            seed=42,
            verbose=False,
        )

        wall_end = time.perf_counter()

        # Stop memory tracker
        stop_event_cpu.set()
        tracker_thread_cpu.join()
        stop_event_gpu.set()
        tracker_thread_gpu.join()

        # Analyze results
        wall_time = wall_end - wall_start

        memory_peak_cpu = max(memory_samples_cpu)
        memory_average_cpu = sum(memory_samples_cpu) / len(memory_samples_cpu)

        memory_peak_gpu = max(memory_samples_gpu)
        memory_average_gpu = sum(memory_samples_gpu) / len(memory_samples_gpu)

        wall_times.append(wall_time)
        mem_peaks_cpu.append(memory_peak_cpu)
        mem_peaks_gpu.append(memory_peak_gpu)
        mem_avgs_cpu.append(memory_average_cpu)
        mem_avgs_gpu.append(memory_average_gpu)

        grn.to_csv(os.path.join(save_path, f'grn_num_cells_{n}.csv'))

    res_df = pd.DataFrame(
        {
            'wall_time': wall_times,
            'mem_peak_cpu': mem_peaks_cpu,
            'mem_avg_cpu': mem_avgs_cpu,
            'mem_peak_gpu': mem_peaks_gpu,
            'mem_avg_gpu': mem_avgs_gpu,
        },
        index=NUM_CELLS
    )

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

    preliminary = False

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

