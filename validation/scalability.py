

# Todo:
#  - Define grid of (n_cells, n_genes)
#  - For each grid entry run grn_inf, cellrank , and splicejac
#  - For a selection of GRN sizes (n_edges for switchtfi, n_tfs for drivaer; or just n_edges) run switchtfi and drivaer
#  - Sensible Track runtime, cpu time (are any of them parallelized?), memory
#  - Run on HPC

def data_processing():
    # Todo: download and process data if not done before ...
    pass

def scalability_grn_inf():
    pass


def scalability_switchtfi():
    pass


def scalability_cellrank():
    pass


def scalability_splicejac():
    pass


def scalability_drivaer():
    pass


if __name__ == '__main__':

    import os
    import argparse

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
        # Todo: check if grn inf was run
        pass

    res_dir = './results/03_validation/scalability'
    os.makedirs(res_dir, exist_ok=True)

    data_processing()

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

